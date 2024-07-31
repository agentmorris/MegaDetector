"""

process_video.py

Splits a video (or folder of videos) into frames, runs the frames through run_detector_batch.py,
and optionally stitches together results into a new video with detection boxes.

Operates by separating the video into frames, typically sampling every Nth frame, and writing
those frames to disk, before running MD.  This approach clearly has a downside: it requires
a bunch more disk space, compared to extracting frames and running MD on them without ever
writing them to disk.  The upside, though, is that this approach allows you to run repeat
detection elimination after running MegaDetector, and it allows allows more efficient re-use
of frames if you end up running MD more than once, or running multiple versions of MD.

"""

#%% Imports

import os
import sys
import tempfile
import argparse
import itertools
import json
import shutil
import getpass

from uuid import uuid1

from megadetector.detection import run_detector_batch
from megadetector.visualization import visualize_detector_output
from megadetector.utils.ct_utils import args_to_object
from megadetector.utils.path_utils import insert_before_extension, clean_path
from megadetector.detection.video_utils import video_to_frames
from megadetector.detection.video_utils import run_callback_on_frames
from megadetector.detection.video_utils import run_callback_on_frames_for_folder
from megadetector.detection.video_utils import frames_to_video
from megadetector.detection.video_utils import frame_results_to_video_results
from megadetector.detection.video_utils import _add_frame_numbers_to_results
from megadetector.detection.video_utils import video_folder_to_frames
from megadetector.detection.video_utils import default_fourcc
from megadetector.detection.run_detector import load_detector


#%% Classes

class ProcessVideoOptions:
    """
    Options controlling the behavior of process_video()
    """
    
    def __init__(self):
        
        #: Can be a model filename (.pt or .pb) or a model name (e.g. "MDV5A")
        #: 
        #: Use the string "no_detection" to indicate that you only want to extract frames,
        #: not run a model.  If you do this, you almost definitely want to set 
        #: keep_extracted_frames to "True", otherwise everything in this module is a no-op.
        #: I.e., there's no reason to extract frames, do nothing with them, then delete them.
        self.model_file = 'MDV5A'
        
        #: Video (of folder of videos) to process
        self.input_video_file = ''
    
        #: .json file to which we should write results
        self.output_json_file = None
        
        #: File to which we should write a video with boxes, only relevant if 
        #: render_output_video is True
        self.output_video_file = None
        
        #: Folder to use for extracted frames; will use a folder in system temp space
        #: if this is None
        self.frame_folder = None
        
        #: Folder to use for rendered frames (if rendering output video); will use a folder 
        #: in system temp space if this is None
        self.frame_rendering_folder = None
        
        #: Should we render a video with detection boxes?
        #:
        #: If processing a folder, this renders each input video to a separate
        #: video with detection boxes.
        self.render_output_video = False
        
        #: If we are rendering boxes to a new video, should we keep the temporary
        #: rendered frames?
        self.keep_rendered_frames = False
        
        #: Should we keep the extracted frames?
        self.keep_extracted_frames = False
        
        #: Should we delete the entire folder the extracted frames are written to?
        #:
        #: By default, we delete the frame files but leave the (probably-empty) folder in place, 
        #: for no reason other than being paranoid about deleting folders.
        self.force_extracted_frame_folder_deletion = False
        
        #: Should we delete the entire folder the rendered frames are written to?
        #:
        #: By default, we delete the frame files but leave the (probably-empty) folder in place,
        #: for no reason other than being paranoid about deleting folders.
        self.force_rendered_frame_folder_deletion = False
         
        #: If we've already run MegaDetector on this video or folder of videos, i.e. if we 
        #: find a corresponding MD results file, should we re-use it?  Defaults to reprocessing.
        self.reuse_results_if_available = False
        
        #: If we've already split this video or folder of videos into frames, should we 
        #: we re-use those extracted frames?  Defaults to reprocessing.
        self.reuse_frames_if_available = False
        
        #: If [input_video_file] is a folder, should we search for videos recursively?
        self.recursive = False 
        
        #: Enable additional debug console output
        self.verbose = False
        
        #: fourcc code to use for writing videos; only relevant if render_output_video is True
        self.fourcc = None
    
        #: force a specific frame rate for output videos; only relevant if render_output_video 
        #: is True
        self.rendering_fs = None
        
        #: Confidence threshold to use for writing videos with boxes, only relevant if
        #: if render_output_video is True.  Defaults to choosing a reasonable threshold
        #: based on the model version.
        self.rendering_confidence_threshold = None
        
        #: Detections below this threshold will not be included in the output file.
        self.json_confidence_threshold = 0.005
        
        #: Sample every Nth frame; set to None (default) or 1 to sample every frame.  Typically
        #: we sample down to around 3 fps, so for typical 30 fps videos, frame_sample=10 is a 
        #: typical value.  Mutually exclusive with [frames_to_extract].
        self.frame_sample = None
        
        #: Extract a specific set of frames (list of ints, or a single int).  Mutually exclusive with
        #: [frame_sample].
        self.frames_to_extract = None
        
        #: Number of workers to use for parallelization; set to <= 1 to disable parallelization
        self.n_cores = 1
    
        #: For debugging only, stop processing after a certain number of frames.
        self.debug_max_frames = -1
        
        #: For debugging only, force on-disk frame extraction, even if it wouldn't otherwise be
        #: necessary
        self.force_on_disk_frame_extraction = False
        
        #: File containing non-standard categories, typically only used if you're running a non-MD
        #: detector.
        self.class_mapping_filename = None
        
        #: JPEG quality for frame output, from 0-100.  Use None or -1 to let opencv decide.
        self.quality = 90
        
        #: Resize frames so they're at most this wide
        self.max_width = None
        
        #: Run the model at this image size (don't mess with this unless you know what you're
        #: getting into)
        self.image_size = None
        
        #: Enable image augmentation
        self.augment = False
        
        #: By default, a video with no frames (or no frames retrievable with the current parameters)
        #: is an error, this makes it a warning.  This would apply if you request, e.g., the 100th
        #: frame from each video, but a video only has 50 frames.
        self.allow_empty_videos = False
    
# ...class ProcessVideoOptions


#%% Functions

def _select_temporary_output_folders(options):
    """
    Choose folders in system temp space for writing temporary frames.  Does not create folders,
    just defines them.
    """
    
    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    
    # If we create a folder like "process_camera_trap_video" in the system temp dir, it may
    # be the case that no one else can write to it, even to create user-specific subfolders.
    # If we create a uuid-named folder in the system temp dir, we make a mess.  
    #
    # Compromise with "process_camera_trap_video-[user]".
    user_tempdir = tempdir + '-' + getpass.getuser()
    
    # I don't know whether it's possible for a username to contain characters that are
    # not valid filename characters, but just to be sure...
    user_tempdir = clean_path(user_tempdir)
    
    frame_output_folder = os.path.join(
        user_tempdir, os.path.basename(options.input_video_file) + '_frames_' + str(uuid1()))
    
    rendering_output_folder = os.path.join(
        tempdir, os.path.basename(options.input_video_file) + '_detections_' + str(uuid1()))
        
    temporary_folder_info = \
    {
        'temp_folder_base':user_tempdir,
        'frame_output_folder':frame_output_folder,
        'rendering_output_folder':rendering_output_folder
    }
    
    return temporary_folder_info

# ...def _create_frame_output_folders(...)


def _clean_up_rendered_frames(options,rendering_output_folder,detected_frame_files):
    """
    If necessary, delete rendered frames and/or the entire rendering output folder.
    """
    
    if rendering_output_folder is None:
        return
    
    caller_provided_rendering_output_folder = (options.frame_rendering_folder is not None)
    
    # (Optionally) delete the temporary directory we used for rendered detection images
    if not options.keep_rendered_frames:
        
        try:
            
            # If (a) we're supposed to delete the temporary rendering folder no
            # matter where it is and (b) we created it in temp space, delete the 
            # whole tree
            if options.force_rendered_frame_folder_deletion and \
               (not caller_provided_rendering_output_folder):
                   
                if options.verbose:
                    print('Recursively deleting rendered frame folder {}'.format(
                        rendering_output_folder))
                    
                shutil.rmtree(rendering_output_folder)
                
            # ...otherwise just delete the frames, but leave the folder in place
            else:
                
                if options.force_rendered_frame_folder_deletion:
                    assert caller_provided_rendering_output_folder
                    print('Warning: force_rendered_frame_folder_deletion supplied with a ' + \
                          'user-provided folder, only removing frames')
                        
                for rendered_frame_fn in detected_frame_files:
                    os.remove(rendered_frame_fn)
                    
        except Exception as e:
            print('Warning: error deleting rendered frames from folder {}:\n{}'.format(
                rendering_output_folder,str(e)))
            pass

    elif options.force_rendered_frame_folder_deletion:
        
        print('Warning: keep_rendered_frames and force_rendered_frame_folder_deletion both ' + \
              'specified, not deleting')
            
# ...def _clean_up_rendered_frames(...)


def _clean_up_extracted_frames(options,frame_output_folder,frame_filenames):
    """
    If necessary, delete extracted frames and/or the entire temporary frame folder.
    """
    
    if frame_output_folder is None:
        return
    
    caller_provided_frame_output_folder = (options.frame_folder is not None)
                                           
    if not options.keep_extracted_frames:
        
        try:
            
            # If (a) we're supposed to delete the temporary frame folder no
            # matter where it is and (b) we created it in temp space, delete the 
            # whole tree.
            if options.force_extracted_frame_folder_deletion and \
               (not caller_provided_frame_output_folder):
                   
                if options.verbose:
                    print('Recursively deleting frame output folder {}'.format(frame_output_folder))
                    
                shutil.rmtree(frame_output_folder)
                
            # ...otherwise just delete the frames, but leave the folder in place
            else:
                
                if frame_filenames is None:
                    return
                
                if options.force_extracted_frame_folder_deletion:
                    assert caller_provided_frame_output_folder
                    print('Warning: force_extracted_frame_folder_deletion supplied with a ' + \
                          'user-provided folder, only removing frames')
                    
                for extracted_frame_fn in frame_filenames:
                    os.remove(extracted_frame_fn)
                    
        except Exception as e:
            print('Warning: error removing extracted frames from folder {}:\n{}'.format(
                frame_output_folder,str(e)))
            pass
    
    elif options.force_extracted_frame_folder_deletion:
        
        print('Warning: keep_extracted_frames and force_extracted_frame_folder_deletion both ' + \
              'specified, not deleting')

# ...def _clean_up_extracted_frames


def process_video(options):
    """
    Process a single video through MD, optionally writing a new video with boxes.
    Can also be used just to split a video into frames, without running a model.
    
    Args: 
        options (ProcessVideoOptions): all the parameters used to control this process,
            including filenames; see ProcessVideoOptions for details
            
    Returns:
        dict: frame-level MegaDetector results, identical to what's in the output .json file
    """

    if options.output_json_file is None:
        options.output_json_file = options.input_video_file + '.json'

    if options.render_output_video and (options.output_video_file is None):
        options.output_video_file = options.input_video_file + '.detections.mp4'

    if options.model_file == 'no_detection' and not options.keep_extracted_frames:
        print('Warning: you asked for no detection, but did not specify keep_extracted_frames, this is a no-op')
        return
    
    # Track whether frame and rendering folders were created by this script
    caller_provided_frame_output_folder = (options.frame_folder is not None)
    caller_provided_rendering_output_folder = (options.frame_rendering_folder is not None)
    
    frame_output_folder = None
        
    # If we should re-use existing results, and the output file exists, don't bother running MD
    if (options.reuse_results_if_available and os.path.isfile(options.output_json_file)):
            
            print('Loading results from {}'.format(options.output_json_file))
            with open(options.output_json_file,'r') as f:
                results = json.load(f)
    
    # Run MD in memory if we don't need to generate frames
    #
    # Currently if we're generating an output video, we need to generate frames on disk first.
    elif (not options.keep_extracted_frames and \
          not options.render_output_video and \
          not options.force_on_disk_frame_extraction):
        
        # Run MegaDetector in memory
        
        if options.verbose:
            print('Running MegaDetector in memory for {}'.format(options.input_video_file))
            
        if options.frame_folder is not None:
            print('Warning: frame_folder specified, but keep_extracted_frames is ' + \
                  'not; no raw frames will be written')
        
        detector = load_detector(options.model_file)
        
        def frame_callback(image_np,image_id):
            return detector.generate_detections_one_image(image_np,
                                                          image_id,
                                                          detection_threshold=options.json_confidence_threshold,
                                                          augment=options.augment)
        
        frame_results = run_callback_on_frames(options.input_video_file, 
                                               frame_callback,
                                               every_n_frames=options.frame_sample, 
                                               verbose=options.verbose, 
                                               frames_to_process=options.frames_to_extract)
        
        _add_frame_numbers_to_results(frame_results['results'])
        
        run_detector_batch.write_results_to_file(
            frame_results['results'], 
            options.output_json_file,
            relative_path_base=None,
            detector_file=options.model_file,
            custom_metadata={'video_frame_rate':frame_results['frame_rate']})

    # Extract frames and optionally run MegaDetector on those frames
    else:
                
        if options.verbose:
            print('Extracting frames for {}'.format(options.input_video_file))
            
        # This does not create any folders, just defines temporary folder names in 
        # case we need them.
        temporary_folder_info = _select_temporary_output_folders(options)
            
        if (caller_provided_frame_output_folder):
            frame_output_folder = options.frame_folder
        else:
            frame_output_folder = temporary_folder_info['frame_output_folder']
       
        os.makedirs(frame_output_folder, exist_ok=True)
    
    
        ## Extract frames
        
        frame_filenames, Fs = video_to_frames(
                                options.input_video_file, 
                                frame_output_folder, 
                                every_n_frames=options.frame_sample, 
                                overwrite=(not options.reuse_frames_if_available),
                                quality=options.quality, 
                                max_width=options.max_width, 
                                verbose=options.verbose,
                                frames_to_extract=options.frames_to_extract,
                                allow_empty_videos=options.allow_empty_videos)
    
        image_file_names = frame_filenames
        if options.debug_max_frames > 0:
            image_file_names = image_file_names[0:options.debug_max_frames]
                
        ## Run MegaDetector on those frames
        
        if options.model_file != 'no_detection':
        
            if options.verbose:
                print('Running MD for {}'.format(options.input_video_file))
                
            results = run_detector_batch.load_and_run_detector_batch(
                options.model_file, 
                image_file_names,
                confidence_threshold=options.json_confidence_threshold,
                n_cores=options.n_cores,
                class_mapping_filename=options.class_mapping_filename,
                quiet=True,
                augment=options.augment,
                image_size=options.image_size)
        
            _add_frame_numbers_to_results(results)
        
            run_detector_batch.write_results_to_file(
                results, 
                options.output_json_file,
                relative_path_base=frame_output_folder,
                detector_file=options.model_file,
                custom_metadata={'video_frame_rate':Fs})
            
    # ...if we are/aren't keeping raw frames on disk
        
    
    ## (Optionally) render output video
    
    if options.render_output_video:
        
        # Render detections to images
        if (caller_provided_rendering_output_folder):
            rendering_output_dir = options.frame_rendering_folder
        else:
            rendering_output_dir = temporary_folder_info['rendering_output_folder']
            
        os.makedirs(rendering_output_dir,exist_ok=True)
        
        detected_frame_files = visualize_detector_output.visualize_detector_output(
            detector_output_path=options.output_json_file,
            out_dir=rendering_output_dir,
            images_dir=frame_output_folder,
            confidence_threshold=options.rendering_confidence_threshold)

        # Combine into a video
        if options.rendering_fs is not None:
            rendering_fs = options.rendering_fs
        elif options.frame_sample is None:
            rendering_fs = Fs
        else:
            # If the original video was 30fps and we sampled every 10th frame, 
            # render at 3fps
            rendering_fs = Fs / options.frame_sample
            
        print('Rendering {} frames to {} at {} fps (original video {} fps)'.format(
            len(detected_frame_files), options.output_video_file,rendering_fs,Fs))
        frames_to_video(detected_frame_files, 
                        rendering_fs, 
                        options.output_video_file, 
                        codec_spec=options.fourcc)
        
        # Possibly clean up rendered frames
        _clean_up_rendered_frames(options,rendering_output_dir,detected_frame_files)
    
    # ...if we're rendering video
    
    
    ## (Optionally) delete the extracted frames
    _clean_up_extracted_frames(options, frame_output_folder, frame_filenames)
    
# ...process_video()


def process_video_folder(options):
    """
    Process a folder of videos through MD. Can also be used just to split a folder of
    videos into frames, without running a model.
    
    When this function is used to run MD, two .json files will get written, one with 
    an entry for each *frame* (identical to what's created by process_video()), and
    one with an entry for each *video* (which is more suitable for, e.g., reading into
    Timelapse).
    
    Args: 
        options (ProcessVideoOptions): all the parameters used to control this process,
            including filenames; see ProcessVideoOptions for details            
    """
    
    ## Validate options

    assert os.path.isdir(options.input_video_file), \
        '{} is not a folder'.format(options.input_video_file)
           
    if options.model_file == 'no_detection' and not options.keep_extracted_frames:
        print('Warning: you asked for no detection, but did not specify keep_extracted_frames, this is a no-op')
        return
    
    if options.model_file != 'no_detection':
        assert options.output_json_file is not None, \
            'When processing a folder, you must specify an output .json file'      
        assert options.output_json_file.endswith('.json')
        video_json = options.output_json_file
        frames_json = options.output_json_file.replace('.json','.frames.json')
        os.makedirs(os.path.dirname(video_json),exist_ok=True)
    
    # Track whether frame and rendering folders were created by this script
    caller_provided_frame_output_folder = (options.frame_folder is not None)
    caller_provided_rendering_output_folder = (options.frame_rendering_folder is not None)
    
    # This does not create any folders, just defines temporary folder names in 
    # case we need them.
    temporary_folder_info = _select_temporary_output_folders(options)
    
    frame_output_folder = None
    image_file_names = None
    
    # Run MD in memory if we don't need to generate frames
    #
    # Currently if we're generating an output video, we need to generate frames on disk first.
    if (not options.keep_extracted_frames and \
        not options.render_output_video and \
        not options.force_on_disk_frame_extraction):
                
        if options.verbose:
            print('Running MegaDetector in memory for folder {}'.format(options.input_video_file))
            
        if options.frame_folder is not None:
            print('Warning: frame_folder specified, but keep_extracted_frames is ' + \
                  'not; no raw frames will be written')
        
        detector = load_detector(options.model_file)
        
        def frame_callback(image_np,image_id):
            return detector.generate_detections_one_image(image_np,
                                                          image_id,
                                                          detection_threshold=options.json_confidence_threshold,
                                                          augment=options.augment)
        
        md_results = run_callback_on_frames_for_folder(input_video_folder=options.input_video_file, 
                                                       frame_callback=frame_callback,
                                                       every_n_frames=options.frame_sample, 
                                                       verbose=options.verbose)
        
        video_results = md_results['results']
        
        all_frame_results = []
        
        # r = video_results[0]
        for frame_results in video_results:
            _add_frame_numbers_to_results(frame_results)
            all_frame_results.extend(frame_results)
        
        run_detector_batch.write_results_to_file(
            all_frame_results, 
            frames_json,
            relative_path_base=None,
            detector_file=options.model_file,
            custom_metadata={'video_frame_rate':md_results['frame_rates']})
    
    else:
        
        ## Split every video into frames
        
        if caller_provided_frame_output_folder:
            frame_output_folder = options.frame_folder
        else:
            frame_output_folder = temporary_folder_info['frame_output_folder']
            
        os.makedirs(frame_output_folder, exist_ok=True)
    
        print('Extracting frames')
        
        frame_filenames, Fs, video_filenames = \
            video_folder_to_frames(input_folder=options.input_video_file,
                                   output_folder_base=frame_output_folder, 
                                   recursive=options.recursive, 
                                   overwrite=(not options.reuse_frames_if_available),
                                   n_threads=options.n_cores,
                                   every_n_frames=options.frame_sample,
                                   verbose=options.verbose,
                                   quality=options.quality,
                                   max_width=options.max_width,
                                   frames_to_extract=options.frames_to_extract,
                                   allow_empty_videos=options.allow_empty_videos)
        
        print('Extracted frames for {} videos'.format(len(set(video_filenames))))
        image_file_names = list(itertools.chain.from_iterable(frame_filenames))
        
        if len(image_file_names) == 0:
            if len(video_filenames) == 0:
                print('No videos found in folder {}'.format(options.input_video_file))
            else:
                print('No frames extracted from folder {}, this may be due to an '\
                      'unsupported video codec'.format(options.input_video_file))
            return
    
        if options.debug_max_frames is not None and options.debug_max_frames > 0:
            image_file_names = image_file_names[0:options.debug_max_frames]
            
        if options.model_file == 'no_detection':
            assert options.keep_extracted_frames, \
                'Internal error: keep_extracted_frames not set, but no model specified'
            return
        
        
        ## Run MegaDetector on the extracted frames
        
        if options.reuse_results_if_available and \
            os.path.isfile(frames_json):
                print('Bypassing inference, loading results from {}'.format(frames_json))
                with open(frames_json,'r') as f:
                    results = json.load(f)
        else:
            print('Running MegaDetector')
            results = run_detector_batch.load_and_run_detector_batch(
                options.model_file, 
                image_file_names,
                confidence_threshold=options.json_confidence_threshold,
                n_cores=options.n_cores,
                class_mapping_filename=options.class_mapping_filename,
                quiet=True,
                augment=options.augment,
                image_size=options.image_size)
        
            _add_frame_numbers_to_results(results)
            
            run_detector_batch.write_results_to_file(
                results, 
                frames_json,
                relative_path_base=frame_output_folder,
                detector_file=options.model_file,
                custom_metadata={'video_frame_rate':Fs})
        
    # ...if we're running MD on in-memory frames vs. extracting frames to disk
    
    ## Convert frame-level results to video-level results

    print('Converting frame-level results to video-level results')
    frame_results_to_video_results(frames_json,video_json)


    ## (Optionally) render output videos
    
    if options.render_output_video:
                
        # Render detections to images
        if (caller_provided_rendering_output_folder):
            rendering_output_dir = options.frame_rendering_folder
        else:
            rendering_output_dir = temporary_folder_info['rendering_output_folder']
            
        os.makedirs(rendering_output_dir,exist_ok=True)
        
        detected_frame_files = visualize_detector_output.visualize_detector_output(
            detector_output_path=frames_json,
            out_dir=rendering_output_dir,
            images_dir=frame_output_folder,
            confidence_threshold=options.rendering_confidence_threshold,
            preserve_path_structure=True,
            output_image_width=-1)
        detected_frame_files = [s.replace('\\','/') for s in detected_frame_files]
        
        # Choose an output folder
        output_folder_is_input_folder = False
        if options.output_video_file is not None:
            if os.path.isfile(options.output_video_file):
                raise ValueError('Rendering videos for a folder, but an existing file was specified as output')
            elif options.output_video_file == options.input_video_file:
                output_folder_is_input_folder = True
                output_video_folder = options.input_video_file
            else:
                os.makedirs(options.output_video_file,exist_ok=True)
                output_video_folder = options.output_video_file
        else:
            output_folder_is_input_folder = True
            output_video_folder = options.input_video_file
                                
        # For each video
        #
        # TODO: parallelize this loop
        #
        # i_video=0; input_video_file_abs = video_filenames[i_video]
        for i_video,input_video_file_abs in enumerate(video_filenames):
            
            video_fs = Fs[i_video]
            
            if options.rendering_fs is not None:
                rendering_fs = options.rendering_fs
            elif options.frame_sample is None:                
                rendering_fs = video_fs
            else:
                # If the original video was 30fps and we sampled every 10th frame, 
                # render at 3fps                
                rendering_fs = video_fs / options.frame_sample
            
            input_video_file_relative = os.path.relpath(input_video_file_abs,options.input_video_file)
            video_frame_output_folder = os.path.join(rendering_output_dir,input_video_file_relative)
            
            video_frame_output_folder = video_frame_output_folder.replace('\\','/')
            assert os.path.isdir(video_frame_output_folder), \
                'Could not find frame folder for video {}'.format(input_video_file_relative)
            
            # Find the corresponding rendered frame folder
            video_frame_files = [fn for fn in detected_frame_files if \
                                 fn.startswith(video_frame_output_folder)]
            assert len(video_frame_files) > 0, 'Could not find rendered frames for video {}'.format(
                input_video_file_relative)
            
            # Select the output filename for the rendered video
            if output_folder_is_input_folder:
                video_output_file = insert_before_extension(input_video_file_abs,'annotated','_')
            else:
                video_output_file = os.path.join(output_video_folder,input_video_file_relative)
            
            os.makedirs(os.path.dirname(video_output_file),exist_ok=True)
            
            # Create the output video            
            print('Rendering detections for video {} to {} at {} fps (original video {} fps)'.format(
                input_video_file_relative,video_output_file,rendering_fs,video_fs))
            frames_to_video(video_frame_files, 
                            rendering_fs, 
                            video_output_file, 
                            codec_spec=options.fourcc)
                
        # ...for each video
        
        # Possibly clean up rendered frames
        _clean_up_rendered_frames(options,rendering_output_dir,detected_frame_files)          
        
    # ...if we're rendering video
    
    
    ## (Optionally) delete the extracted frames
    _clean_up_extracted_frames(options, frame_output_folder, image_file_names)

# ...process_video_folder()


def options_to_command(options):
    """
    Convert a ProcessVideoOptions object to a corresponding command line.
    
    Args:
        options (ProcessVideoOptions): the options set to render as a command line
        
    Returns:
        str: the command line corresponding to [options]
        
    :meta private:
    """
    cmd = 'python process_video.py'
    cmd += ' "' + options.model_file + '"'
    cmd += ' "' + options.input_video_file + '"'
    
    if options.recursive:
        cmd += ' --recursive'
    if options.frame_folder is not None:
        cmd += ' --frame_folder' + ' "' + options.frame_folder + '"'
    if options.frame_rendering_folder is not None:
        cmd += ' --frame_rendering_folder' + ' "' + options.frame_rendering_folder + '"'    
    if options.output_json_file is not None:
        cmd += ' --output_json_file' + ' "' + options.output_json_file + '"'
    if options.output_video_file is not None:
        cmd += ' --output_video_file' + ' "' + options.output_video_file + '"'
    if options.keep_extracted_frames:
        cmd += ' --keep_extracted_frames'
    if options.reuse_results_if_available:
        cmd += ' --reuse_results_if_available'    
    if options.reuse_frames_if_available:
        cmd += ' --reuse_frames_if_available'
    if options.render_output_video:
        cmd += ' --render_output_video'
    if options.keep_rendered_frames:
        cmd += ' --keep_rendered_frames'    
    if options.rendering_confidence_threshold is not None:
        cmd += ' --rendering_confidence_threshold ' + str(options.rendering_confidence_threshold)
    if options.json_confidence_threshold is not None:
        cmd += ' --json_confidence_threshold ' + str(options.json_confidence_threshold)
    if options.n_cores is not None:
        cmd += ' --n_cores ' + str(options.n_cores)
    if options.frame_sample is not None:
        cmd += ' --frame_sample ' + str(options.frame_sample)
    if options.frames_to_extract is not None:
        cmd += ' --frames_to_extract '
        if isinstance(options.frames_to_extract,int):
            frames_to_extract = [options.frames_to_extract]
        else:
            frames_to_extract = options.frames_to_extract
        for frame_number in frames_to_extract:
            cmd += ' {}'.format(frame_number)
    if options.debug_max_frames is not None:
        cmd += ' --debug_max_frames ' + str(options.debug_max_frames)
    if options.class_mapping_filename is not None:
        cmd += ' --class_mapping_filename ' + str(options.class_mapping_filename)
    if options.fourcc is not None:
        cmd += ' --fourcc ' + options.fourcc
    if options.quality is not None:
        cmd += ' --quality ' + str(options.quality)
    if options.max_width is not None:
        cmd += ' --max_width ' + str(options.max_width)
    if options.verbose:
        cmd += ' --verbose'
    if options.force_extracted_frame_folder_deletion:
        cmd += ' --force_extracted_frame_folder_deletion'
    if options.force_rendered_frame_folder_deletion:
        cmd += ' --force_rendered_frame_folder_deletion'

    return cmd

    
#%% Interactive driver

if False:    
        
    pass

    #%% Process a folder of videos
    
    model_file = 'MDV5A'
    input_dir = r'g:\temp\test-videos'
    # input_dir = r'G:\temp\md-test-package\md-test-images\video-samples'
    output_base = r'g:\temp\video_test'
    frame_folder = os.path.join(output_base,'frames')
    rendering_folder = os.path.join(output_base,'rendered-frames')
    output_json_file = os.path.join(output_base,'video-test.json')
    output_video_folder = os.path.join(output_base,'output_videos')
    
    
    print('Processing folder {}'.format(input_dir))
    
    options = ProcessVideoOptions()    
    options.model_file = model_file
    options.input_video_file = input_dir
    options.output_video_file = output_video_folder
    options.output_json_file = output_json_file
    options.recursive = True
    options.reuse_frames_if_available = False
    options.reuse_results_if_available = False
    options.quality = None # 90
    options.frame_sample = 10
    options.max_width = None # 1280
    options.n_cores = 4
    options.verbose = True
    options.render_output_video = False  
    options.frame_folder = frame_folder
    options.frame_rendering_folder = rendering_folder    
    options.keep_extracted_frames = False
    options.keep_rendered_frames = False
    options.force_extracted_frame_folder_deletion = False
    options.force_rendered_frame_folder_deletion = False
    options.fourcc = 'mp4v'
    options.force_on_disk_frame_extraction = True
    # options.rendering_confidence_threshold = 0.15
    
    cmd = options_to_command(options); print(cmd)
        
    # import clipboard; clipboard.copy(cmd)
    process_video_folder(options)
        
    
    #%% Process a single video

    fn = r'g:\temp\test-videos\person_and_dog\DSCF0056.AVI'
    assert os.path.isfile(fn)
    model_file = 'MDV5A'
    input_video_file = fn
    
    output_base = r'g:\temp\video_test'
    frame_folder = os.path.join(output_base,'frames')
    rendering_folder = os.path.join(output_base,'rendered-frames')
    output_json_file = os.path.join(output_base,'video-test.json')
    output_video_file = os.path.join(output_base,'output_video.mp4')
    
    options = ProcessVideoOptions()
    options.model_file = model_file
    options.input_video_file = input_video_file
    options.render_output_video = True
    options.output_video_file = output_video_file
    options.output_json_file = output_json_file    
    options.verbose = True    
    options.quality = 75
    options.frame_sample = 10
    options.max_width = 1600    
    options.frame_folder = frame_folder
    options.frame_rendering_folder = rendering_folder    
    options.keep_extracted_frames = False
    options.keep_rendered_frames = False
    options.force_extracted_frame_folder_deletion = True
    options.force_rendered_frame_folder_deletion = True    
    options.fourcc = 'mp4v'
    # options.rendering_confidence_threshold = 0.15
    
    cmd = options_to_command(options); print(cmd)
    
    # import clipboard; clipboard.copy(cmd)    
    process_video(options)
            
    
    #%% Extract specific frames from a single video, no detection

    fn = r'g:\temp\test-videos\person_and_dog\DSCF0064.AVI'
    assert os.path.isfile(fn)
    model_file = 'no_detection'
    input_video_file = fn
    
    output_base = r'g:\temp\video_test'
    frame_folder = os.path.join(output_base,'frames')
    output_video_file = os.path.join(output_base,'output_videos.mp4')
    
    options = ProcessVideoOptions()
    options.model_file = model_file
    options.input_video_file = input_video_file    
    options.verbose = True    
    options.quality = 90
    options.frame_sample = None
    options.frames_to_extract = [0,100]
    options.max_width = None    
    options.frame_folder = frame_folder
    options.keep_extracted_frames = True
    
    cmd = options_to_command(options); print(cmd)
    
    # import clipboard; clipboard.copy(cmd)
    process_video(options)    
        
        
    #%% Extract specific frames from a folder, no detection

    fn = r'g:\temp\test-videos\person_and_dog'
    assert os.path.isdir(fn)
    model_file = 'no_detection'
    input_video_file = fn
    
    output_base = r'g:\temp\video_test'
    frame_folder = os.path.join(output_base,'frames')
    output_video_file = os.path.join(output_base,'output_videos.mp4')
    
    options = ProcessVideoOptions()
    options.model_file = model_file
    options.input_video_file = input_video_file    
    options.verbose = True    
    options.quality = 90
    options.frame_sample = None
    options.frames_to_extract = [0,100]
    options.max_width = None    
    options.frame_folder = frame_folder
    options.keep_extracted_frames = True
    
    cmd = options_to_command(options); print(cmd)
    
    # import clipboard; clipboard.copy(cmd)
    process_video(options)    

        
#%% Command-line driver

def main():

    default_options = ProcessVideoOptions()
    
    parser = argparse.ArgumentParser(description=(
        'Run MegaDetector on each frame (or every Nth frame) in a video (or folder of videos), optionally '\
        'producing a new video with detections annotated'))

    parser.add_argument('model_file', type=str,
                        help='MegaDetector model file (.pt or .pb) or model name (e.g. "MDV5A"), '\
                             'or the string "no_detection" to run just frame extraction')

    parser.add_argument('input_video_file', type=str,
                        help='video file (or folder) to process')

    parser.add_argument('--recursive', action='store_true',
                        help='recurse into [input_video_file]; only meaningful if a folder '\
                         'is specified as input')
    
    parser.add_argument('--frame_folder', type=str, default=None,
                        help='folder to use for intermediate frame storage, defaults to a folder '\
                        'in the system temporary folder')
        
    parser.add_argument('--frame_rendering_folder', type=str, default=None,
                        help='folder to use for rendered frame storage, defaults to a folder in '\
                        'the system temporary folder')
    
    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json')

    parser.add_argument('--output_video_file', type=str,
                        default=None, help='video output file (or folder), defaults to '\
                            '[video file].mp4 for files, or [video file]_annotated for folders')

    parser.add_argument('--keep_extracted_frames',
                       action='store_true', help='Disable the deletion of extracted frames')
    
    parser.add_argument('--reuse_frames_if_available',
                       action='store_true', help="Don't extract frames that are already available in the frame extraction folder")
    
    parser.add_argument('--reuse_results_if_available',
                       action='store_true', help='If the output .json files exists, and this flag is set,'\
                           'we\'ll skip running MegaDetector')
    
    parser.add_argument('--render_output_video', action='store_true',
                        help='enable video output rendering (not rendered by default)')

    parser.add_argument('--fourcc', default=default_fourcc,
                        help='fourcc code to use for video encoding (default {}), only used if render_output_video is True'.format(
                            default_fourcc))
    
    parser.add_argument('--keep_rendered_frames',
                       action='store_true', help='Disable the deletion of rendered (w/boxes) frames')

    parser.add_argument('--force_extracted_frame_folder_deletion',
                       action='store_true', help='By default, when keep_extracted_frames is False, we '\
                           'delete the frames, but leave the (probably-empty) folder in place.  This option '\
                           'forces deletion of the folder as well.  Use at your own risk; does not check '\
                           'whether other files were present in the folder.')
        
    parser.add_argument('--force_rendered_frame_folder_deletion',
                       action='store_true', help='By default, when keep_rendered_frames is False, we '\
                           'delete the frames, but leave the (probably-empty) folder in place.  This option '\
                           'forces deletion of the folder as well.  Use at your own risk; does not check '\
                           'whether other files were present in the folder.')
        
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=None, 
                        help="don't render boxes with confidence below this threshold (defaults to choosing based on the MD version)")

    parser.add_argument('--rendering_fs', type=float,
                        default=None, 
                        help='force a specific frame rate for output videos (only relevant when using '\
                             '--render_output_video) (defaults to the original frame rate)')

    parser.add_argument('--json_confidence_threshold', type=float,
                        default=default_options.json_confidence_threshold, 
                        help="don't include boxes in the .json file with confidence "\
                            'below this threshold (default {})'.format(
                                default_options.json_confidence_threshold))

    parser.add_argument('--n_cores', type=int,
                        default=default_options.n_cores,
                        help='Number of cores to use for frame separation and detection. '\
                            'If using a GPU, this option will be respected for frame separation but '\
                            'ignored for detection.  Only relevant to frame separation when processing '\
                            'a folder.  Default {}.'.format(default_options.n_cores))

    parser.add_argument('--frame_sample', type=int,
                        default=None, help='process every Nth frame (defaults to every frame)')

    parser.add_argument('--frames_to_extract', nargs='+', type=int,
                        default=None, help='extract specific frames (one or more ints)')

    parser.add_argument('--quality', type=int,
                        default=default_options.quality, 
                        help='JPEG quality for extracted frames (defaults to {}), use -1 to force no quality setting'.format(
                            default_options.quality))

    parser.add_argument('--max_width', type=int,
                        default=default_options.max_width, 
                        help='Resize frames larger than this before writing (defaults to {})'.format(
                            default_options.max_width))

    parser.add_argument('--debug_max_frames', type=int,
                        default=-1, help='Trim to N frames for debugging (impacts model execution, '\
                            'not frame rendering)')
    
    parser.add_argument('--class_mapping_filename',
                        type=str,
                        default=None, help='Use a non-default class mapping, supplied in a .json file '\
                            'with a dictionary mapping int-strings to strings.  This will also disable '\
                            'the addition of "1" to all category IDs, so your class mapping should start '\
                            'at zero.')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable additional debug output')
    
    parser.add_argument('--image_size',
                        type=int,
                        default=None,
                        help=('Force image resizing to a specific integer size on the long '\
                              'axis (not recommended to change this)'))    
    
    parser.add_argument('--augment',
                        action='store_true',
                        help='Enable image augmentation')
    
    parser.add_argument('--allow_empty_videos',
                        action='store_true',
                        help='By default, videos with no retrievable frames cause an error, this makes it a warning')
            
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()
    options = ProcessVideoOptions()
    args_to_object(args,options)

    if os.path.isdir(options.input_video_file):
        process_video_folder(options)
    else:
        assert os.path.isfile(options.input_video_file), \
            '{} is not a valid file or folder name'.format(options.input_video_file)
        assert not options.recursive, \
            '--recursive is only meaningful when processing a folder'
        process_video(options)

if __name__ == '__main__':
    main()
