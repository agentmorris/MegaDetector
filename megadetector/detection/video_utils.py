"""

video_utils.py

Utilities for splitting, rendering, and assembling videos.

"""

#%% Constants, imports, environment

import os
import cv2
import glob
import json

from collections import defaultdict
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial

from megadetector.utils import path_utils    
from megadetector.visualization import visualization_utils as vis_utils

default_fourcc = 'h264'


#%% Path utilities

VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg')

def is_video_file(s,video_extensions=VIDEO_EXTENSIONS):
    """
    Checks a file's extension against a set of known video file
    extensions to determine whether it's a video file.  Performs a
    case-insensitive comparison.
    
    Args:
        s (str): filename to check for probable video-ness
        video_extensions (list, optional): list of video file extensions
    
    Returns:
        bool: True if this looks like a video file, else False
    """
    
    ext = os.path.splitext(s)[1]
    return ext.lower() in video_extensions


def find_video_strings(strings):
    """
    Given a list of strings that are potentially video file names, looks for
    strings that actually look like video file names (based on extension).
    
    Args:
        strings (list): list of strings to check for video-ness
    
    Returns:
        list: a subset of [strings] that looks like they are video filenames
    """
    
    return [s for s in strings if is_video_file(s.lower())]


def find_videos(dirname, 
                recursive=False,
                convert_slashes=True,
                return_relative_paths=False):
    """
    Finds all files in a directory that look like video file names.
    
    Args:
        dirname (str): folder to search for video files
        recursive (bool, optional): whether to search [dirname] recursively
        convert_slashes (bool, optional): forces forward slashes in the returned files,
            otherwise uses the native path separator
        return_relative_paths (bool, optional): forces the returned filenames to be 
            relative to [dirname], otherwise returns absolute paths
    
    Returns:
        A list of filenames within [dirname] that appear to be videos
    """
    
    if recursive:
        files = glob.glob(os.path.join(dirname, '**', '*.*'), recursive=True)
    else:
        files = glob.glob(os.path.join(dirname, '*.*'))
        
    if return_relative_paths:
        files = [os.path.relpath(fn,dirname) for fn in files]

    if convert_slashes:
        files = [fn.replace('\\', '/') for fn in files]
    
    return find_video_strings(files)


#%% Function for rendering frames to video and vice-versa

# http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html

def frames_to_video(images, Fs, output_file_name, codec_spec=default_fourcc):
    """
    Given a list of image files and a sample rate, concatenates those images into
    a video and writes to a new video file.
    
    Args:
        images (list): a list of frame file names to concatenate into a video
        Fs (float): the frame rate in fps
        output_file_name (str): the output video file, no checking is performed to make
            sure the extension is compatible with the codec
        codec_spec (str, optional):  codec to use for encoding; h264 is a sensible default 
            and generally works on Windows, but when this fails (which is around 50% of the time 
            on Linux), mp4v is a good second choice
    """
    
    if codec_spec is None:
        codec_spec = 'h264'
        
    if len(images) == 0:
        return

    # Determine the width and height from the first image
    frame = cv2.imread(images[0])
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, Fs, (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def get_video_fs(input_video_file):
    """
    Retrieves the frame rate of [input_video_file].
    
    Args:
        input_video_file (str): video file for which we want the frame rate
        
    Returns:
        float: the frame rate of [input_video_file]
    """
    
    assert os.path.isfile(input_video_file), 'File {} not found'.format(input_video_file)    
    vidcap = cv2.VideoCapture(input_video_file)
    Fs = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return Fs


def _frame_number_to_filename(frame_number):
    """
    Ensures that frame images are given consistent filenames.
    """
    
    return 'frame{:06d}.jpg'.format(frame_number)


def video_to_frames(input_video_file, output_folder, overwrite=True, 
                    every_n_frames=None, verbose=False):
    """
    Renders frames from [input_video_file] to a .jpg in [output_folder].
    
    With help from:
        
    https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    
    Args:
        input_video_file (str): video file to split into frames
        output_folder (str): folder to put frame images in
        overwrite (bool, optional): whether to overwrite existing frame images
        every_n_frames (int, optional): sample every Nth frame starting from the first frame;
            if this is None or 1, every frame is extracted
        verbose (bool, optional): enable additional debug console output
    
    Returns:
        tuple: length-2 tuple containing (list of frame filenames,frame rate)
    """
    
    assert os.path.isfile(input_video_file), 'File {} not found'.format(input_video_file)
    
    vidcap = cv2.VideoCapture(input_video_file)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    Fs = vidcap.get(cv2.CAP_PROP_FPS)
    
    # If we're not over-writing, check whether all frame images already exist
    if overwrite == False:
        
        missing_frame_number = None
        frame_filenames = []
        
        for frame_number in range(0,n_frames):
            
            if every_n_frames is not None:
                if frame_number % every_n_frames != 0:
                    continue
            
            frame_filename = _frame_number_to_filename(frame_number)
            frame_filename = os.path.join(output_folder,frame_filename)
            frame_filenames.append(frame_filename)
            if os.path.isfile(frame_filename):
                continue
            else:
                missing_frame_number = frame_number
                break
    
        # OpenCV seems to over-report the number of frames by 1 in some cases, or fails
        # to read the last frame; either way, I'm allowing one missing frame.
        allow_last_frame_missing = True
        
        if missing_frame_number is None or \
            (allow_last_frame_missing and (missing_frame_number == n_frames-1)):
            if verbose:
                print('Skipping video {}, all output frames exist'.format(input_video_file))
            return frame_filenames,Fs
        else:
            pass
            # print("Rendering video {}, couldn't find frame {}".format(
            #    input_video_file,missing_frame_number))
    
    # ...if we need to check whether to skip this video entirely
        
    if verbose:
        print('Reading {} frames at {} Hz from {}'.format(n_frames,Fs,input_video_file))

    frame_filenames = []

    # for frame_number in tqdm(range(0,n_frames)):
    for frame_number in range(0,n_frames):

        success,image = vidcap.read()
        if not success:
            assert image is None
            if verbose:
                print('Read terminating at frame {} of {}'.format(frame_number,n_frames))
            break

        if every_n_frames is not None:
            if frame_number % every_n_frames != 0:
                continue
            
        frame_filename = _frame_number_to_filename(frame_number)
        frame_filename = os.path.join(output_folder,frame_filename)
        frame_filenames.append(frame_filename)
        
        if overwrite == False and os.path.isfile(frame_filename):
            # print('Skipping frame {}'.format(frame_filename))
            pass            
        else:
            try:
                if frame_filename.isascii():
                    cv2.imwrite(os.path.normpath(frame_filename),image)
                else:
                    is_success, im_buf_arr = cv2.imencode('.jpg', image)
                    im_buf_arr.tofile(frame_filename)
                assert os.path.isfile(frame_filename), \
                    'Output frame {} unavailable'.format(frame_filename)
            except KeyboardInterrupt:
                vidcap.release()
                raise
            except Exception as e:
                print('Error on frame {} of {}: {}'.format(frame_number,n_frames,str(e)))

    if verbose:
        print('\nExtracted {} of {} frames'.format(len(frame_filenames),n_frames))

    vidcap.release()    
    return frame_filenames,Fs

# ...def video_to_frames(...)


def _video_to_frames_for_folder(relative_fn,input_folder,output_folder_base,every_n_frames,overwrite,verbose):
    """
    Internal function to call video_to_frames in the context of video_folder_to_frames; 
    makes sure the right output folder exists, then calls video_to_frames.
    """    
    
    input_fn_absolute = os.path.join(input_folder,relative_fn)
    assert os.path.isfile(input_fn_absolute),\
        'Could not find file {}'.format(input_fn_absolute)

    # Create the target output folder
    output_folder_video = os.path.join(output_folder_base,relative_fn)
    os.makedirs(output_folder_video,exist_ok=True)

    # Render frames
    # input_video_file = input_fn_absolute; output_folder = output_folder_video
    frame_filenames,fs = video_to_frames(input_fn_absolute,output_folder_video,
                                         overwrite=overwrite,every_n_frames=every_n_frames,
                                         verbose=verbose)
    
    return frame_filenames,fs


def video_folder_to_frames(input_folder, output_folder_base, 
                           recursive=True, overwrite=True,
                           n_threads=1, every_n_frames=None,
                           verbose=False, parallelization_uses_threads=True):
    """
    For every video file in input_folder, creates a folder within output_folder_base, and 
    renders frame of that video to images in that folder.
    
    Args:
        input_folder (str): folder to process
        output_folder_base (str): root folder for output images; subfolders will be
            created for each input video
        recursive (bool, optional): whether to recursively process videos in [input_folder]
        overwrite (bool, optional): whether to overwrite existing frame images
        n_threads (int, optional): number of concurrent workers to use; set to <= 1 to disable
            parallelism
        every_n_frames (int, optional): sample every Nth frame starting from the first frame;
            if this is None or 1, every frame is extracted
        verbose (bool, optional): enable additional debug console output
        parallelization_uses_threads (bool, optional): whether to use threads (True) or
            processes (False) for parallelization; ignored if n_threads <= 1
            
    Returns:
        tuple: a length-3 tuple containing:
            - list of lists of frame filenames; the Nth list of frame filenames corresponds to 
              the Nth video
            - list of video frame rates; the Nth value corresponds to the Nth video
            - list of video filenames    
    """
    
    # Recursively enumerate video files
    input_files_full_paths = find_videos(input_folder,recursive=recursive)
    print('Found {} videos in folder {}'.format(len(input_files_full_paths),input_folder))
    if len(input_files_full_paths) == 0:
        return [],[],[]
    
    input_files_relative_paths = [os.path.relpath(s,input_folder) for s in input_files_full_paths]
    input_files_relative_paths = [s.replace('\\','/') for s in input_files_relative_paths]
    
    os.makedirs(output_folder_base,exist_ok=True)    
    
    frame_filenames_by_video = []
    fs_by_video = []
    
    if n_threads == 1:
        # For each video
        #
        # input_fn_relative = input_files_relative_paths[0]
        for input_fn_relative in tqdm(input_files_relative_paths):
        
            frame_filenames,fs = \
                _video_to_frames_for_folder(input_fn_relative,input_folder,output_folder_base,
                                            every_n_frames,overwrite,verbose)
            frame_filenames_by_video.append(frame_filenames)
            fs_by_video.append(fs)
    else:
        if parallelization_uses_threads:
            print('Starting a worker pool with {} threads'.format(n_threads))
            pool = ThreadPool(n_threads)
        else:
            print('Starting a worker pool with {} processes'.format(n_threads))
            pool = Pool(n_threads)
        process_video_with_options = partial(_video_to_frames_for_folder, 
                                             input_folder=input_folder,
                                             output_folder_base=output_folder_base,
                                             every_n_frames=every_n_frames,
                                             overwrite=overwrite,
                                             verbose=verbose)
        results = list(tqdm(pool.imap(
            partial(process_video_with_options),input_files_relative_paths), 
                            total=len(input_files_relative_paths)))
        frame_filenames_by_video = [x[0] for x in results]
        fs_by_video = [x[1] for x in results]
        
    return frame_filenames_by_video,fs_by_video,input_files_full_paths
  
# ...def video_folder_to_frames(...)


class FrameToVideoOptions:
    """
    Options controlling the conversion of frame-level results to video-level results via
    frame_results_to_video_results()    
    """
    
    #: One-indexed indicator of which frame-level confidence value to use to determine detection confidence
    #: for the whole video, i.e. "1" means "use the confidence value from the highest-confidence frame"
    nth_highest_confidence = 1
    
    #: What to do if a file referred to in a .json results file appears not to be a 
    #: video; can be 'error' or 'skip_with_warning'
    non_video_behavior = 'error'
    
    
def frame_results_to_video_results(input_file,output_file,options=None):
    """
    Given an MD results file produced at the *frame* level, corresponding to a directory 
    created with video_folder_to_frames, maps those frame-level results back to the 
    video level for use in Timelapse.
    
    Preserves everything in the input .json file other than the images.
    
    Args:
        input_file (str): the frame-level MD results file to convert to video-level results
        output_file (str): the .json file to which we should write video-level results
        options (FrameToVideoOptions, optional): parameters for converting frame-level results
            to video-level results, see FrameToVideoOptions for details            
    """

    if options is None:
        options = FrameToVideoOptions()
        
    # Load results
    with open(input_file,'r') as f:
        input_data = json.load(f)

    images = input_data['images']
    detection_categories = input_data['detection_categories']
    
    ## Break into videos
    
    video_to_frames = defaultdict(list) 
    
    # im = images[0]
    for im in tqdm(images):
        
        fn = im['file']
        video_name = os.path.dirname(fn)
        if not is_video_file(video_name):
            if options.non_video_behavior == 'error':
                raise ValueError('{} is not a video file'.format(video_name))
            elif options.non_video_behavior == 'skip_with_warning':
                print('Warning: {} is not a video file'.format(video_name))
                continue
            else:
                raise ValueError('Unrecognized non-video handling behavior: {}'.format(
                    options.non_video_behavior))
        video_to_frames[video_name].append(im)
    
    print('Found {} unique videos in {} frame-level results'.format(
        len(video_to_frames),len(images)))
    
    output_images = []
    
    ## For each video...
    
    # video_name = list(video_to_frames.keys())[0]
    for video_name in tqdm(video_to_frames):
        
        frames = video_to_frames[video_name]
        
        all_detections_this_video = []
        
        # frame = frames[0]
        for frame in frames:
            if frame['detections'] is not None:
                all_detections_this_video.extend(frame['detections'])
            
        # At most one detection for each category for the whole video
        canonical_detections = []
            
        # category_id = list(detection_categories.keys())[0]
        for category_id in detection_categories:
            
            category_detections = [det for det in all_detections_this_video if \
                                   det['category'] == category_id]
            
            # Find the nth-highest-confidence video to choose a confidence value
            if len(category_detections) >= options.nth_highest_confidence:
                
                category_detections_by_confidence = sorted(category_detections, 
                                                           key = lambda i: i['conf'],reverse=True)
                canonical_detection = category_detections_by_confidence[options.nth_highest_confidence-1]
                canonical_detections.append(canonical_detection)
                                      
        # Prepare the output representation for this video
        im_out = {}
        im_out['file'] = video_name
        im_out['detections'] = canonical_detections
        
        # 'max_detection_conf' is no longer included in output files by default
        if False:
            im_out['max_detection_conf'] = 0
            if len(canonical_detections) > 0:
                confidences = [d['conf'] for d in canonical_detections]
                im_out['max_detection_conf'] = max(confidences)
        
        output_images.append(im_out)
        
    # ...for each video
    
    output_data = input_data
    output_data['images'] = output_images
    s = json.dumps(output_data,indent=1)
    
    # Write the output file
    with open(output_file,'w') as f:
        f.write(s)
    
# ...def frame_results_to_video_results(...)


#%% Test driver

if False:

    #%% Constants
    
    Fs = 30.01
    confidence_threshold = 0.75
    input_folder = 'z:\\'
    frame_folder_base = r'e:\video_test\frames'
    detected_frame_folder_base = r'e:\video_test\detected_frames'
    rendered_videos_folder_base = r'e:\video_test\rendered_videos'
    
    results_file = r'results.json'
    os.makedirs(detected_frame_folder_base,exist_ok=True)
    os.makedirs(rendered_videos_folder_base,exist_ok=True)
    
    
    #%% Split videos into frames
        
    frame_filenames_by_video,fs_by_video,video_filenames = \
        video_folder_to_frames(input_folder,frame_folder_base,recursive=True)
    
    
    #%% List image files, break into folders
    
    frame_files = path_utils.find_images(frame_folder_base,True)
    frame_files = [s.replace('\\','/') for s in frame_files]
    print('Enumerated {} total frames'.format(len(frame_files)))
    
    Fs = 30.01
    # Find unique folders
    folders = set()
    # fn = frame_files[0]
    for fn in frame_files:
        folders.add(os.path.dirname(fn))
    folders = [s.replace('\\','/') for s in folders]
    print('Found {} folders for {} files'.format(len(folders),len(frame_files)))
    
        
    #%% Load detector output
    
    with open(results_file,'r') as f:
        detection_results = json.load(f)
    detections = detection_results['images']
    detector_label_map = detection_results['detection_categories']
    for d in detections:
        d['file'] = d['file'].replace('\\','/').replace('video_frames/','')


    #%% Render detector frames
    
    # folder = list(folders)[0]
    for folder in folders:
        
        frame_files_this_folder = [fn for fn in frame_files if folder in fn]
        folder_relative = folder.replace((frame_folder_base + '/').replace('\\','/'),'')
        detection_results_this_folder = [d for d in detections if folder_relative in d['file']]
        print('Found {} detections in folder {}'.format(len(detection_results_this_folder),folder))
        assert len(frame_files_this_folder) == len(detection_results_this_folder)
        
        rendered_frame_output_folder = os.path.join(detected_frame_folder_base,folder_relative)
        os.makedirs(rendered_frame_output_folder,exist_ok=True)
        
        # d = detection_results_this_folder[0]
        for d in tqdm(detection_results_this_folder):
            
            input_file = os.path.join(frame_folder_base,d['file'])
            output_file = os.path.join(detected_frame_folder_base,d['file'])
            os.makedirs(os.path.dirname(output_file),exist_ok=True)
            vis_utils.draw_bounding_boxes_on_file(input_file,output_file,d['detections'],
                                                  confidence_threshold)
        
        # ...for each file in this folder
            
    # ...for each folder


    #%% Render output videos
            
    # folder = list(folders)[0]
    for folder in tqdm(folders):
        
        folder_relative = folder.replace((frame_folder_base + '/').replace('\\','/'),'')
        rendered_detector_output_folder = os.path.join(detected_frame_folder_base,folder_relative)
        assert os.path.isdir(rendered_detector_output_folder)
        
        frame_files_relative = os.listdir(rendered_detector_output_folder)
        frame_files_absolute = [os.path.join(rendered_detector_output_folder,s) \
                                for s in frame_files_relative]
        
        output_video_filename = os.path.join(rendered_videos_folder_base,folder_relative)
        os.makedirs(os.path.dirname(output_video_filename),exist_ok=True)
        
        original_video_filename = output_video_filename.replace(
            rendered_videos_folder_base,input_folder)
        assert os.path.isfile(original_video_filename)
        Fs = get_video_fs(original_video_filename)
                
        frames_to_video(frame_files_absolute, Fs, output_video_filename)

    # ...for each video
