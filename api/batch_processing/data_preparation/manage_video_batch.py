########
#
# manage_video_batch.py
#
# Notebook-esque script to manage the process of running a local batch of videos
# through MD.  Defers most of the heavy lifting to manage_local_batch.py .
#
########

#%% Imports and constants

import os

from md_utils import path_utils
from detection import video_utils

input_folder = '/datadrive/data'
output_folder_base = '/datadrive/frames'

assert os.path.isdir(input_folder)
os.makedirs(output_folder_base,exist_ok=True)


#%% Split videos into frames

assert os.path.isdir(input_folder)
os.makedirs(output_folder_base,exist_ok=True)

recursive = True
overwrite = True
n_threads = 4
every_n_frames = 10

frame_filenames_by_video,fs_by_video,video_filenames = \
    video_utils.video_folder_to_frames(input_folder=input_folder,
                                                              output_folder_base=output_folder_base,
                                                              recursive=recursive,
                                                              overwrite=overwrite,
                                                              n_threads=n_threads,
                                                              every_n_frames=every_n_frames,
                                                              parallelization_uses_threads=False)


#%% List frame files, break into folders

from collections import defaultdict

frame_files = path_utils.find_images(output_folder_base,True)
frame_files = [s.replace('\\','/') for s in frame_files]
print('Enumerated {} total frames'.format(len(frame_files)))

# Find unique (relative) folders
folder_to_frame_files = defaultdict(list)

# fn = frame_files[0]
for fn in frame_files:
    folder_name = os.path.dirname(fn)
    folder_name = os.path.relpath(folder_name,output_folder_base)
    folder_to_frame_files[folder_name].append(fn)

print('Found {} folders for {} files'.format(len(folder_to_frame_files),len(frame_files)))


#%% List videos

video_filenames = video_utils.find_videos(input_folder,recursive=True)
video_filenames = [os.path.relpath(fn,input_folder) for fn in video_filenames]
print('Input folder contains {} videos'.format(len(video_filenames)))


#%% Check for videos that are missing entirely

# list(folder_to_frame_files.keys())[0]
# video_filenames[0]

missing_videos = []

# fn = video_filenames[0]
for relative_fn in video_filenames:
    if relative_fn not in folder_to_frame_files:
        missing_videos.append(relative_fn)
        
print('{} of {} folders are missing frames entirely'.format(len(missing_videos),
                                                            len(video_filenames)))


#%% Check for videos with very few frames

min_frames_for_valid_video = 10

low_frame_videos = []

for folder_name in folder_to_frame_files.keys():
    frame_files = folder_to_frame_files[folder_name]
    if len(frame_files) < min_frames_for_valid_video:
        low_frame_videos.append(folder_name)

print('{} of {} folders have fewer than {} frames'.format(
    len(low_frame_videos),len(video_filenames),min_frames_for_valid_video))


#%% Print the list of videos that are problematic

print('Videos that could not be decoded:\n')

for fn in missing_videos:
    print(fn)
    
print('\nVideos with fewer than {} decoded frames:\n'.format(min_frames_for_valid_video))

for fn in low_frame_videos:
    print(fn)
    
    
#%% Process images like we would for any other camera trap job

# ...typically using manage_local_batch.py or manage_local_batch.ipynb, but do this however 
# you like, as long as you get a results file at the end.
#
# If you do RDE, remember to use the second folder from the bottom, rather than the
# bottom-most folder.


#%% Convert frame results to video results
    
from detection.video_utils import frame_results_to_video_results

filtered_output_filename = '/results/organization/stuff.json'
video_output_filename = filtered_output_filename.replace('.json','_aggregated.json')
frame_results_to_video_results(filtered_output_filename,video_output_filename)
    

#%% Confirm that the videos in the .json file are what we expect them to be

import json

with open(video_output_filename,'r') as f:
    video_results = json.load(f)

video_filenames_set = set(video_filenames)

filenames_in_video_results_set = set([im['file'] for im in video_results['images']])

for fn in filenames_in_video_results_set:
    assert fn in video_filenames_set
    

#%% Scrap

if False:
    
    pass

    #%% Render one or more sample videos...
        
    # ...while we still have the frames and detections around
    
    ## Imports
    
    from md_visualization import visualize_detector_output
    from detection.video_utils import frames_to_video    
    
    
    ## Constants and paths
    
    confidence_threshold = 0.2
    input_fs = 30
    
    filtered_output_filename = '/a/b/c/blah_detections.filtered_rde_0.150_0.850_10_1.000.json'
    video_fn_relative = '4.10cam6/IMG_0022.MP4'
    output_video_base = os.path.expanduser('~/tmp/video_preview')
    
    
    ## Filename handling
    
    video_fn_relative = video_fn_relative.replace('\\','/')
    video_fn_flat = video_fn_relative.replace('/','#')
    video_name = os.path.splitext(video_fn_flat)[0]
    output_video = os.path.join(output_video_base,'{}_detections.mp4'.format(video_name))        
    output_fs = input_fs / every_n_frames
    
    rendered_detections_folder = os.path.join(output_video_base,'rendered_detections_{}'.format(video_name))
    os.makedirs(rendered_detections_folder,exist_ok=True)
    
    
    ## Find frames corresponding to this video
    
    with open(filtered_output_filename,'r') as f:
        frame_results = json.load(f)
    
    frame_results_this_video = []
    
    # im = frame_results['images'][0]
    for im in frame_results['images']:
        if im['file'].replace('\\','/').startswith(video_fn_relative):
            frame_results_this_video.append(im)

    assert len(frame_results_this_video) > 0, \
        'No frame results matched {}'.format(video_fn_relative)
    print('Found {} matching frame results'.format(len(frame_results_this_video)))
    
    frame_results['images'] = frame_results_this_video
    
    frames_json = os.path.join(rendered_detections_folder,video_fn_flat + '.json')
    
    with open(frames_json,'w') as f:
        json.dump(frame_results,f,indent=1)        
        
        
    ## Render detections on those frames    
    
    detected_frame_files = visualize_detector_output.visualize_detector_output(
        detector_output_path=frames_json,
        out_dir=rendered_detections_folder,
        images_dir=output_folder_base,
        confidence_threshold=confidence_threshold,
        preserve_path_structure=True,
        output_image_width=-1)
        
    
    ## Render the output video
    
    frames_to_video(detected_frame_files, output_fs, output_video, codec_spec='h264')
    
    # from md_utils.path_utils import open_file; open_file(output_video)


    #%% Test a possibly-broken video
    
    fn = '/datadrive/tmp/video.AVI'
        
    fs = video_utils.get_video_fs(fn)
    print(fs)
    
    tmpfolder = '/home/user/tmp/frametmp'
    os.makedirs(tmpfolder,exist_ok=True)
    
    video_utils.video_to_frames(fn, tmpfolder, verbose=True, every_n_frames=10)
    
    
    #%% List videos in a folder
    
    input_folder = '/datadrive/tmp/organization/data'
    video_filenames = video_utils.find_videos(input_folder,recursive=True)


#%% End notebook: turn this script into a notebook (how meta!)

import os
import nbformat as nbf

input_py_file = os.path.expanduser(
    '~/git/MegaDetector/api/batch_processing/data_preparation/manage_video_batch.py')
assert os.path.isfile(input_py_file)
output_ipynb_file = input_py_file.replace('.py','.ipynb')

nb_header = '# Managing a local MegaDetector video batch'

nb_header += '\n'

nb_header += \
"""
This notebook represents an interactive process for running MegaDetector on large batches of videos, including typical and optional postprocessing steps.

This notebook is auto-generated from manage_video_batch.py (a cell-delimited .py file that is used the same way, typically in Spyder or VS Code).

"""

with open(input_py_file,'r') as f:
    lines = f.readlines()

nb = nbf.v4.new_notebook()
nb['cells'].append(nbf.v4.new_markdown_cell(nb_header))

i_line = 0

# Exclude everything before the first cell
while(not lines[i_line].startswith('#%%')):
    i_line += 1

current_cell = []

def write_code_cell(c):
    
    first_non_empty_line = None
    last_non_empty_line = None
    
    for i_code_line,code_line in enumerate(c):
        if len(code_line.strip()) > 0:
            if first_non_empty_line is None:
                first_non_empty_line = i_code_line
            last_non_empty_line = i_code_line
            
    # Remove the first [first_non_empty_lines] from the list
    c = c[first_non_empty_line:]
    last_non_empty_line -= first_non_empty_line
    c = c[:last_non_empty_line+1]
    
    nb['cells'].append(nbf.v4.new_code_cell('\n'.join(c)))
        
while(True):    
            
    line = lines[i_line].rstrip()
    
    if 'end notebook' in line.lower():
        break
    
    if lines[i_line].startswith('#%% '):
        if len(current_cell) > 0:
            write_code_cell(current_cell)
            current_cell = []
        markdown_content = line.replace('#%%','##')
        nb['cells'].append(nbf.v4.new_markdown_cell(markdown_content))
    else:
        current_cell.append(line)

    i_line += 1

# Add the last cell
write_code_cell(current_cell)

nbf.write(nb,output_ipynb_file)
