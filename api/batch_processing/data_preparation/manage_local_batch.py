########
#
# manage_local_batch.py
#    
# Semi-automated process for managing a local MegaDetector job, including
# standard postprocessing steps.
#
# This script is not intended to be run from top to bottom like a typical Python script,
# it's a notebook disguised with a .py extension.  It's the Bestest Most Awesome way to
# run MegaDetector, but it's also pretty subtle; if you want to play with this, you might
# want to check in with cameratraps@lila.science for some tips.  Otherwise... YMMV.
#
# Some general notes on using this script, which I do in Spyder, though everything will be
# the same if you are reading this in Jupyter Notebook (using the .ipynb version of the 
# script):
#
# * You can specify the MegaDetector location, but you may find it useful to use the same paths 
#   I use; on all the machines where I run MD, I keep all versions of MegaDetector handy at these 
#   paths:
#  
#   ~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt
#   ~/models/camera_traps/megadetector/md_v5.0.0/md_v5b.0.0.pt
#   ~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb
#
#   On Windows, this translates to, for example:
#
#   c:\users\dmorr\models\camera_traps\megadetector\md_v5.0.0\md_v5a.0.0.pt
#    
# * Typically when I have a MegaDetector job to run, I make a copy of this script.  Let's 
#   say I'm running a job for an organization called "bibblebop"; I have a big folder of
#   job-specific copies of this script, and I might save a new one called "bibblebop-2023-07-26.py" 
#   (the filename doesn't matter, it just helps me keep these organized).
#
# * There are three variables you need to set in this script before you start running code:
#   "input_path", "organization_name_short", and "job_date".  You will get a sensible error if you forget 
#   to set any of these.  In this case I might set those to "/data/bibblebobcamerastuff",
#   "bibblebop", and "2023-07-26", respectively.
#
# * The defaults assume you want to split the job into two tasks (this is the default because I have 
#   two GPUs).  Nothing bad will happen if you do this on a zero-GPU or single-GPU machine, but if you
#   want everything to run in one logical task, change "n_gpus" and "n_jobs" to 1 (instead of 2).
#
# * After setting the required variables, I run the first few cells - up to and including the one 
#   called "Generate commands" - which collectively take basically zero seconds.  After you run the
#   "Generate commands" cell, you will have a folder that looks something like:
#
#   ~/postprocessing/bibblebop/bibblebop-2023-07-06-mdv5a/
#  
#   On Windows, this means:
#
#   ~/postprocessing/bibblebop/bibblebop-2023-07-06-mdv5a/    
#
#   Everything related to this job - scripts, outputs, intermediate stuff - will be in this folder.
#   Specifically, after the "Generate commands" cell, you'll have scripts in that folder called something
#   like:
#
#   run_chunk_000_gpu_00.sh (or .bat on Windows)
#
#   Personally, I like to run that script directly in a command prompt (I just leave Spyder open, though 
#   it's OK if Spyder gets shut down while MD is running).  
#
#   At this point, once you get the hang of it, you've invested about zero seconds of human time,
#   but possibly several days of unattended compute time, depending on the size of your job.
#   
# * Then when the jobs are done, back to the interactive environment!  I run the next few cells,
#   which make sure the job finished OK, and the cell called "Post-processing (pre-RDE)", which 
#   generates an HTML preview of the results.  You are very plausibly done at this point, and can ignore
#   all the remaining cells.  If you want to do things like repeat detection elimination, or running 
#   a classifier, or splitting your results file up in specialized ways, there are cells for all of those
#   things, but now you're in power-user territory, so I'm going to leave this guide here.  Email
#   cameratraps@lila.science with questions about the fancy stuff.
#
########

#%% Imports and constants

import json
import os
import stat
import time

import humanfriendly

from tqdm import tqdm
from collections import defaultdict

from md_utils import path_utils
from md_utils.ct_utils import is_list_sorted
from md_utils.ct_utils import split_list_into_n_chunks

from detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from detection.run_detector import DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)
from detection.run_detector import get_detector_version_from_filename

max_task_name_length = 92

# To specify a non-default confidence threshold for including detections in the .json file
json_threshold = None

# Turn warnings into errors if more than this many images are missing
max_tolerable_failed_images = 100

use_image_queue = False

# Only relevant when we're using a single GPU
default_gpu_number = 0

quiet_mode = True

# Specify a target image size when running MD... strongly recommended to leave this at "None"
image_size = None

# Only relevant when running on CPU
ncores = 1

# OS-specific script line continuation character
slcc = '\\'

# OS-specific script comment character
scc = '#' 

script_extension = '.sh'

# Prefer threads on Windows, processes on Linux
parallelization_defaults_to_threads = False

# This is for things like image rendering, not for MegaDetector
default_workers_for_parallel_tasks = 30

# Should we use YOLOv5's val.py instead of run_detector_batch.py?
use_yolo_inference_scripts = False

# Directory in which to run val.py.  Only relevant if use_yolo_inference_scripts is True.
yolo_working_dir = os.path.expanduser('~/git/yolov5')

# Should we remove intermediate files used for running YOLOv5's val.py?
#
# Only relevant if use_yolo_inference_scripts is True.
remove_yolo_intermediate_results = False
remove_yolo_symlink_folder = False
use_symlinks_for_yolo_inference = True

overwrite_handling = 'skip' # 'skip', 'error', or 'overwrite'

# Set later if EK113/RCNX101-style overflow folders are being handled in this dataset
overflow_folder_handling_enabled = False

# Should we apply YOLOv5's augmentation?  Only allowed when use_yolo_inference_scripts
# is True.
augment = False

if os.name == 'nt':
    slcc = '^'
    scc = 'REM'
    script_extension = '.bat'
    parallelization_defaults_to_threads = True
    default_workers_for_parallel_tasks = 10


#%% Constants I set per script

input_path = '/drive/organization'

assert not (input_path.endswith('/') or input_path.endswith('\\'))

organization_name_short = 'organization'
job_date = None # '2023-05-08'
assert job_date is not None and organization_name_short != 'organization'

# Optional descriptor
job_tag = None

if job_tag is None:
    job_description_string = ''
else:
    job_description_string = '-' + job_tag

model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5b.0.0.pt')
# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb')

postprocessing_base = os.path.expanduser('~/postprocessing')

# Number of jobs to split data into, typically equal to the number of available GPUs, though
# when using augmentation or an image queue (and thus not using checkpoints), I typically
# use ~100 jobs per GPU; those serve as de facto checkpoints.
n_jobs = 2
n_gpus = 2

# Set to "None" when using augmentation or an image queue, which don't currently support
# checkpointing.  Don't worry, this will be assert()'d in the next cell.
checkpoint_frequency = 10000

# gpu_images_per_second is only used to print out a time estimate, and it's completely
# tied to the assumption of running on an RTX 3090.  YMMV.
if ('v5') in model_file:
    gpu_images_per_second = 10
else:
    gpu_images_per_second = 2.9
    
# Rough estimate for how much slower everything runs when using augmentation    
if augment:
    gpu_images_per_second = gpu_images_per_second * 0.7
    
base_task_name = organization_name_short + '-' + job_date + job_description_string + '-' + \
    get_detector_version_from_filename(model_file)
base_output_folder_name = os.path.join(postprocessing_base,organization_name_short)
os.makedirs(base_output_folder_name,exist_ok=True)


#%% Derived variables, constant validation, path setup

if use_image_queue:
    assert checkpoint_frequency is None,\
        'Checkpointing is not supported when using an image queue'        
    
if augment:
    assert checkpoint_frequency is None,\
        'Checkpointing is not supported when using augmentation'
    
    assert use_yolo_inference_scripts,\
        'Augmentation is only supported when running with the YOLO inference scripts'

filename_base = os.path.join(base_output_folder_name, base_task_name)
combined_api_output_folder = os.path.join(filename_base, 'combined_api_outputs')
postprocessing_output_folder = os.path.join(filename_base, 'preview')

os.makedirs(filename_base, exist_ok=True)
os.makedirs(combined_api_output_folder, exist_ok=True)
os.makedirs(postprocessing_output_folder, exist_ok=True)

if input_path.endswith('/'):
    input_path = input_path[0:-1]

print('Output folder:\n{}'.format(filename_base))


#%% Enumerate files

all_images = sorted(path_utils.find_images(input_path,recursive=True))

# It's common to run this notebook on an external drive with the main folders in the drive root
all_images = [fn for fn in all_images if not \
              (fn.startswith('$RECYCLE') or fn.startswith('System Volume Information'))]
    
print('Enumerated {} image files in {}'.format(len(all_images),input_path))

if False:

    pass 
    
    #%% Load files from prior enumeration
    
    import re    
    chunk_files = os.listdir(filename_base)
    pattern = re.compile('chunk\d+.json')
    chunk_files = [fn for fn in chunk_files if pattern.match(fn)]
    all_images = []
    for fn in chunk_files:
        with open(os.path.join(filename_base,fn),'r') as f:
            chunk = json.load(f)
            assert isinstance(chunk,list)
            all_images.extend(chunk)
    all_images = sorted(all_images)
    print('Loaded {} image files from chunks in {}'.format(len(all_images),filename_base))
    

#%% Divide images into chunks 

folder_chunks = split_list_into_n_chunks(all_images,n_jobs)


#%% Estimate total time

n_images = len(all_images)
execution_seconds = n_images / gpu_images_per_second
wallclock_seconds = execution_seconds / n_gpus
print('Expected time: {}'.format(humanfriendly.format_timespan(wallclock_seconds)))

seconds_per_chunk = len(folder_chunks[0]) / gpu_images_per_second
print('Expected time per chunk: {}'.format(humanfriendly.format_timespan(seconds_per_chunk)))


#%% Write file lists

task_info = []

for i_chunk,chunk_list in enumerate(folder_chunks):
    
    chunk_fn = os.path.join(filename_base,'chunk{}.json'.format(str(i_chunk).zfill(3)))
    task_info.append({'id':i_chunk,'input_file':chunk_fn})
    path_utils.write_list_to_file(chunk_fn, chunk_list)
    
    
#%% Generate commands

# A list of the scripts tied to each GPU, as absolute paths.  We'll write this out at
# the end so each GPU's list of commands can be run at once.  Generally only used when 
# running lots of small batches via YOLOv5's val.py, which doesn't support checkpointing.
gpu_to_scripts = defaultdict(list)

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):
    
    chunk_file = task['input_file']
    output_fn = chunk_file.replace('.json','_results.json')
    
    task['output_file'] = output_fn
    
    if n_jobs > 1:
        gpu_number = i_task % n_gpus        
    else:
        gpu_number = default_gpu_number
        
    image_size_string = ''
    if image_size is not None:
        image_size_string = '--image_size {}'.format(image_size)
        
    # Generate the script to run MD
    
    if use_yolo_inference_scripts:

        augment_string = ''
        if augment:
            augment_string = '--augment_enabled 1'
        
        symlink_folder = os.path.join(filename_base,'symlinks','symlinks_{}'.format(
            str(i_task).zfill(3)))
        yolo_results_folder = os.path.join(filename_base,'yolo_results','yolo_results_{}'.format(
            str(i_task).zfill(3)))
                
        symlink_folder_string = '--symlink_folder "{}"'.format(symlink_folder)
        yolo_results_folder_string = '--yolo_results_folder "{}"'.format(yolo_results_folder)
        
        remove_symlink_folder_string = ''
        if not remove_yolo_symlink_folder:
            remove_symlink_folder_string = '--no_remove_symlink_folder'
        
        remove_yolo_results_string = ''
        if not remove_yolo_intermediate_results:
            remove_yolo_results_string = '--no_remove_yolo_results_folder'
        
        confidence_threshold_string = ''
        if json_threshold is not None:
            confidence_threshold_string = '--conf_thres {}'.format(json_threshold)
        else:
            confidence_threshold_string = '--conf_thres {}'.format(DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD)
            
        cmd = ''
        
        device_string = '--device {}'.format(gpu_number)
        
        overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)        
        
        cmd += f'python run_inference_with_yolov5_val.py "{model_file}" "{chunk_file}" "{output_fn}" '
        cmd += f'--yolo_working_folder "{yolo_working_dir}" {image_size_string} {augment_string} '
        cmd += f'{symlink_folder_string} {yolo_results_folder_string} {remove_yolo_results_string} '
        cmd += f'{remove_symlink_folder_string} {confidence_threshold_string} {device_string} '
        cmd += f'{overwrite_handling_string}'
        
        if not use_symlinks_for_yolo_inference:
            cmd += ' --no_use_symlinks'
        
        cmd += '\n'
        
    else:
        
        if os.name == 'nt':
            cuda_string = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
        else:
            cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number} '
                
        checkpoint_frequency_string = ''
        checkpoint_path_string = ''
        checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')
        
        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
            checkpoint_path_string = '--checkpoint_path "{}"'.format(checkpoint_filename)
                
        use_image_queue_string = ''
        if (use_image_queue):
            use_image_queue_string = '--use_image_queue'

        ncores_string = ''
        if (ncores > 1):
            ncores_string = '--ncores {}'.format(ncores)
            
        quiet_string = ''
        if quiet_mode:
            quiet_string = '--quiet'
        
        confidence_threshold_string = ''
        if json_threshold is not None:
            confidence_threshold_string = '--threshold {}'.format(json_threshold)
        
        overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)        
        cmd = f'{cuda_string} python run_detector_batch.py "{model_file}" "{chunk_file}" "{output_fn}" {checkpoint_frequency_string} {checkpoint_path_string} {use_image_queue_string} {ncores_string} {quiet_string} {image_size_string} {confidence_threshold_string} {overwrite_handling_string}'
                
    cmd_file = os.path.join(filename_base,'run_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                            str(gpu_number).zfill(2),script_extension))
    
    with open(cmd_file,'w') as f:
        f.write(cmd + '\n')
    
    st = os.stat(cmd_file)
    os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)
        
    task['command'] = cmd
    task['command_file'] = cmd_file

    # Generate the script to resume from the checkpoint (only supported with MD inference code)
    
    gpu_to_scripts[gpu_number].append(cmd_file)
    
    if checkpoint_frequency is not None:
        
        resume_string = ' --resume_from_checkpoint "{}"'.format(checkpoint_filename)
        resume_cmd = cmd + resume_string
    
        resume_cmd_file = os.path.join(filename_base,
                                       'resume_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                                       str(gpu_number).zfill(2),script_extension))
        
        with open(resume_cmd_file,'w') as f:
            f.write(resume_cmd + '\n')
        
        st = os.stat(resume_cmd_file)
        os.chmod(resume_cmd_file, st.st_mode | stat.S_IEXEC)
        
        task['resume_command'] = resume_cmd
        task['resume_command_file'] = resume_cmd_file

# ...for each task

# Write out a script for each GPU that runs all of the commands associated with
# that GPU.  Typically only used when running lots of little scripts in lieu
# of checkpointing.
for gpu_number in gpu_to_scripts:
    
    gpu_script_file = os.path.join(filename_base,'run_all_for_gpu_{}{}'.format(
        str(gpu_number).zfill(2),script_extension))
    with open(gpu_script_file,'w') as f:
        for script_name in gpu_to_scripts[gpu_number]:
            s = script_name
            # When calling a series of batch files on Windows from within a batch file, you need to
            # use "call", or only the first will be executed.  No, it doesn't make sense.
            if os.name == 'nt':
                s = 'call ' + s
            f.write(s + '\n')
        f.write('echo "Finished all commands for GPU {}"'.format(gpu_number))
    st = os.stat(gpu_script_file)
    os.chmod(gpu_script_file, st.st_mode | stat.S_IEXEC)

# ...for each GPU


#%% Run the tasks

r"""
The cells we've run so far wrote out some shell scripts (.bat files on Windows, 
.sh files on Linx/Mac) that will run MegaDetector.  I like to leave the interactive
environment at this point and run those scripts at the command line.  So, for example,
if you're on Windows, and you've basically used the default values above, there will be
batch files called, e.g.:

c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_000_gpu_00.bat
c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_001_gpu_01.bat

Those batch files expect to be run from the "detection" folder of the MegaDetector repo,
typically:
    
c:\git\MegaDetector\detection

All of that said, you don't *have* to do this at the command line.  The following cell 
runs these scripts programmatically, so if you just run the "run the tasks (commented out)"
cell, you should be running MegaDetector.

One downside of the programmatic approach is that this cell doesn't yet parallelize over
multiple processes, so the tasks will run serially.  This only matters if you have multiple
GPUs.
"""

if False:
    
    pass

    #%%% Run the tasks (commented out)

    assert not use_yolo_inference_scripts, \
        'If you want to use the YOLOv5 inference scripts, you can\'t run the model interactively (yet)'
        
    # i_task = 0; task = task_info[i_task]
    for i_task,task in enumerate(task_info):
    
        chunk_file = task['input_file']
        output_fn = task['output_file']
        
        checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')
        
        if json_threshold is not None:
            confidence_threshold = json_threshold
        else:
            confidence_threshold = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
            
        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            cp_freq_arg = checkpoint_frequency
        else:
            cp_freq_arg = -1
            
        start_time = time.time()
        results = load_and_run_detector_batch(model_file=model_file, 
                                              image_file_names=chunk_file, 
                                              checkpoint_path=checkpoint_filename, 
                                              confidence_threshold=confidence_threshold,
                                              checkpoint_frequency=cp_freq_arg, 
                                              results=None,
                                              n_cores=ncores, 
                                              use_image_queue=use_image_queue,
                                              quiet=quiet_mode,
                                              image_size=image_size)        
        elapsed = time.time() - start_time
        
        print('Task {}: finished inference for {} images in {}'.format(
            i_task, len(results),humanfriendly.format_timespan(elapsed)))

        # This will write absolute paths to the file, we'll fix this later
        write_results_to_file(results, output_fn, detector_file=model_file)

        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            if os.path.isfile(checkpoint_filename):                
                os.remove(checkpoint_filename)
                print('Deleted checkpoint file {}'.format(checkpoint_filename))
                
    # ...for each chunk
    
# ...if False

    
#%% Load results, look for failed or missing images in each task

n_total_failures = 0

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):
    
    chunk_file = task['input_file']
    output_file = task['output_file']
    
    with open(chunk_file,'r') as f:
        task_images = json.load(f)
    with open(output_file,'r') as f:
        task_results = json.load(f)
    
    task_images_set = set(task_images)
    filename_to_results = {}
    
    n_task_failures = 0
    
    # im = task_results['images'][0]
    for im in task_results['images']:
        assert im['file'].startswith(input_path)
        assert im['file'] in task_images_set
        filename_to_results[im['file']] = im
        if 'failure' in im:
            assert im['failure'] is not None
            n_task_failures += 1
    
    task['n_failures'] = n_task_failures
    task['results'] = task_results
    
    for fn in task_images:
        assert fn in filename_to_results
    
    n_total_failures += n_task_failures

# ...for each task

assert n_total_failures < max_tolerable_failed_images,\
    '{} failures (max tolerable set to {})'.format(n_total_failures,
                                                   max_tolerable_failed_images)

print('Processed all {} images with {} failures'.format(
    len(all_images),n_total_failures))
        

##%% Merge results files and make filenames relative

combined_results = {}
combined_results['images'] = []
images_processed = set()

for i_task,task in enumerate(task_info):

    task_results = task['results']
    
    if i_task == 0:
        combined_results['info'] = task_results['info']
        combined_results['detection_categories'] = task_results['detection_categories']        
    else:
        assert task_results['info']['format_version'] == combined_results['info']['format_version']
        assert task_results['detection_categories'] == combined_results['detection_categories']
        
    # Make sure we didn't see this image in another chunk
    for im in task_results['images']:
        assert im['file'] not in images_processed
        images_processed.add(im['file'])

    combined_results['images'].extend(task_results['images'])
    
# Check that we ended up with the right number of images    
assert len(combined_results['images']) == len(all_images), \
    'Expected {} images in combined results, found {}'.format(
        len(all_images),len(combined_results['images']))

# Check uniqueness
result_filenames = [im['file'] for im in combined_results['images']]
assert len(combined_results['images']) == len(set(result_filenames))

# Check for valid path names
for im in combined_results['images']:
    if input_path.endswith(':'):
        assert im['file'].startswith(input_path)
        im['file'] = im['file'].replace(input_path,'',1)
    else:
        assert im['file'].startswith(input_path + os.path.sep)
        im['file'] = im['file'].replace(input_path + os.path.sep,'',1)
    
combined_api_output_file = os.path.join(
    combined_api_output_folder,
    '{}_detections.json'.format(base_task_name))

with open(combined_api_output_file,'w') as f:
    json.dump(combined_results,f,indent=1)

print('Wrote results to {}'.format(combined_api_output_file))


#%% Post-processing (pre-RDE)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_path
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.confidence_threshold = 0.2
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.sample_seed = 0
options.max_figures_per_html_file = 2500

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                     'detections_person_vehicle','non_detections']

output_base = os.path.join(postprocessing_output_folder,
    base_task_name + '_{:.3f}'.format(options.confidence_threshold))
if render_animals_only:
    output_base = output_base + '_animals_only'

os.makedirs(output_base, exist_ok=True)
print('Processing to {}'.format(output_base))

options.api_output_file = combined_api_output_file
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
path_utils.open_file(html_output_file)


#%% RDE (sample directory collapsing)

#
# The next few cells are about repeat detection elimination; if you want to skip this,
# and still do other stuff in this notebook (e.g. running classifiers), that's fine, but
# the rest of the notebook weakly assumes you've done this.  Specifically, it looks for
# the variable "filtered_api_output_file" (a file produced by the RDE process).  If you
# don't run the RDE cells, just change "filtered_api_output_file" to "combined_api_output_file"
# (the raw output from MegaDetector).  Then it will be like all this RDE stuff doesn't exist.
#
# Though FWIW, once you're sufficiently power-user-ish to use this notebook, RDE is almost
# always worth it.
#

def relative_path_to_location(relative_path):
    """
    This is a sample function that returns a camera name given an image path.  By 
    default in the RDE process, leaf-node folders are equivalent to cameras.  To map 
    something other than leaf-node folders to cameras, fill in this function, and un-comment the 
    line below containing "relative_path_to_location".
    
    Sample regular expressions are included here for common patterns, particularly the 
    overflow folders created by Reconyx and Bushnell camera traps.  So if one of those 
    fits your scenario, you don't have to modify this function, just un-comment the line
    below that enables this feature.
    
    Nothing bad happens if you have overflow folders like this and you don't
    enable this mapping, you are just taking a more conservative approach to RDE in that
    scenario.
    """
    
    import re
    
    # 100RECNX is the overflow folder style for Reconyx cameras
    # 100EK113 is (for some reason) the overflow folder style for Bushnell cameras
    # 100_BTCF is the overflow folder style for Browning cameras
    patterns = ['\/\d+RECNX\/','\/\d+EK\d+\/','\/\d+_BTCF\/']
    
    relative_path = relative_path.replace('\\','/')    
    for pat in patterns:
        relative_path = re.sub(pat,'/',relative_path)
    location_name = os.path.dirname(relative_path)
    
    return location_name


#%% Test cells for relative_path_to_location

if False:

    pass

    #%% Test the generic cases
    
    relative_path = 'a/b/c/d/100EK113/blah.jpg'
    print(relative_path_to_location(relative_path))
    
    relative_path = 'a/b/c/d/100RECNX/blah.jpg'
    print(relative_path_to_location(relative_path))
    
    
    #%% Test relative_path_to_location on the current dataset
    
    with open(combined_api_output_file,'r') as f:
        d = json.load(f)
    image_filenames = [im['file'] for im in d['images']]
    
    location_names = set()
    
    # relative_path = image_filenames[0]
    for relative_path in tqdm(image_filenames):
        location_name = relative_path_to_location(relative_path)
        location_names.add(location_name)
        
    location_names = list(location_names)
    location_names.sort()
    
    for s in location_names:
        print(s)


#%% Repeat detection elimination, phase 1

# Deliberately leaving these imports here, rather than at the top, because this
# cell is not typically executed
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core
task_index = 0

options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.15
options.confidenceMax = 1.01
options.iouThreshold = 0.85
options.occurrenceThreshold = 15
options.maxSuspiciousDetectionSize = 0.2
# options.minSuspiciousDetectionSize = 0.05

options.parallelizationUsesThreads = parallelization_defaults_to_threads
options.nWorkers = default_workers_for_parallel_tasks

# This will cause a very light gray box to get drawn around all the detections
# we're *not* considering as suspicious.
options.bRenderOtherDetections = True
options.otherDetectionsThreshold = options.confidenceMin

options.bRenderDetectionTiles = True
options.maxOutputImageWidth = 2000
options.detectionTilesMaxCrops = 500

# options.lineThickness = 5
# options.boxExpansion = 8

# To invoke custom collapsing of folders for a particular manufacturer's naming scheme
# options.customDirNameFunction = relative_path_to_location; overflow_folder_handling_enabled = True

options.bRenderHtml = False
options.imageBase = input_path
rde_string = 'rde_{:.3f}_{:.3f}_{}_{:.3f}'.format(
    options.confidenceMin, options.iouThreshold,
    options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
options.outputBase = os.path.join(filename_base, rde_string + '_task_{}'.format(task_index))
options.filenameReplacements = None # {'':''}

# Exclude people and vehicles from RDE
# options.excludeClasses = [2,3]

# options.maxImagesPerFolder = 50000
# options.includeFolders = ['a/b/c']
# options.excludeFolder = ['a/b/c']

options.debugMaxDir = -1
options.debugMaxRenderDir = -1
options.debugMaxRenderDetection = -1
options.debugMaxRenderInstance = -1

# Can be None, 'xsort', or 'clustersort'
options.smartSort = 'xsort'

suspiciousDetectionResults = repeat_detections_core.find_repeat_detections(combined_api_output_file,
                                                                           None,
                                                                           options)


#%% Manual RDE step

## DELETE THE VALID DETECTIONS ##

# If you run this line, it will open the folder up in your file browser
path_utils.open_file(os.path.dirname(suspiciousDetectionResults.filterFile))

#
# If you ran the previous cell, but then you change your mind and you don't want to do 
# the RDE step, that's fine, but don't just blast through this cell once you've run the 
# previous cell.  If you do that, you're implicitly telling the notebook that you looked 
# at everything in that folder, and confirmed there were no red boxes on animals.
#
# Instead, either change "filtered_api_output_file" below to "combined_api_output_file", 
# or delete *all* the images in the filtering folder.
#


#%% Re-filtering

from api.batch_processing.postprocessing.repeat_detection_elimination import remove_repeat_detections

filtered_output_filename = path_utils.insert_before_extension(combined_api_output_file, 'filtered_{}'.format(rde_string))

remove_repeat_detections.remove_repeat_detections(
    inputFile=combined_api_output_file,
    outputFile=filtered_output_filename,
    filteringDir=os.path.dirname(suspiciousDetectionResults.filterFile)
    )


#%% Post-processing (post-RDE)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_path
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.confidence_threshold = 0.2
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.sample_seed = 0
options.max_figures_per_html_file = 5000

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                      'detections_person_vehicle','non_detections']    

output_base = os.path.join(postprocessing_output_folder, 
    base_task_name + '_{}_{:.3f}'.format(rde_string, options.confidence_threshold))    

if render_animals_only:
    output_base = output_base + '_render_animals_only'
os.makedirs(output_base, exist_ok=True)

print('Processing post-RDE to {}'.format(output_base))

options.api_output_file = filtered_output_filename
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file

path_utils.open_file(html_output_file)


#%% Run MegaClassifier (actually, write out a script that runs MegaClassifier)

# Variables that will indicate which classifiers we ran
final_output_path_mc = None
final_output_path_ic = None

classifier_name_short = 'megaclassifier'
threshold_str = '0.15' # 0.6
classifier_name = 'megaclassifier_v0.1_efficientnet-b3'

organization_name = organization_name_short
job_name = base_task_name
input_filename = filtered_output_filename # combined_api_output_file
input_files = [input_filename]
image_base = input_path
crop_path = os.path.join(os.path.expanduser('~/crops'),job_name + '_crops')
output_base = combined_api_output_folder
device_id = 0

output_file = os.path.join(filename_base,'run_{}_'.format(classifier_name_short) + job_name + script_extension)

classifier_base = os.path.expanduser('~/models/camera_traps/megaclassifier/v0.1/')
assert os.path.isdir(classifier_base)

checkpoint_path = os.path.join(classifier_base,'v0.1_efficientnet-b3_compiled.pt')
assert os.path.isfile(checkpoint_path)

classifier_categories_path = os.path.join(classifier_base,'v0.1_index_to_name.json')
assert os.path.isfile(classifier_categories_path)

target_mapping_path = os.path.join(classifier_base,'idfg_to_megaclassifier_labels.json')
assert os.path.isfile(target_mapping_path)

classifier_output_suffix = '_megaclassifier_output.csv.gz'
final_output_suffix = '_megaclassifier.json'

n_threads_str = str(default_workers_for_parallel_tasks)
image_size_str = '300'
batch_size_str = '64'
num_workers_str = str(default_workers_for_parallel_tasks)
classification_threshold_str = '0.05'

logdir = filename_base

# This is just passed along to the metadata in the output file, it has no impact
# on how the classification scripts run.
typical_classification_threshold_str = '0.75'

##%% Set up environment

commands = []
# commands.append('cd MegaDetector/classification\n')
# commands.append('conda activate cameratraps-classifier\n')

##%% Crop images

commands.append('\n' + scc + ' Cropping ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    crop_cmd = ''
    
    crop_comment = '\n' + scc + ' Cropping {}\n'.format(fn)
    crop_cmd += crop_comment
    
    crop_cmd += "python crop_detections.py " + slcc + "\n" + \
    	 ' "' + input_file_path + '" ' + slcc + '\n' + \
         ' "' + crop_path + '" ' + slcc + '\n' + \
         ' ' + '--images-dir "' + image_base + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--square-crops ' + ' ' + slcc + '\n' + \
         ' ' + '--threads "' + n_threads_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--logdir "' + logdir + '"' + '\n' + \
         ' ' + '\n'
    crop_cmd = '{}'.format(crop_cmd)
    commands.append(crop_cmd)


##%% Run classifier

commands.append('\n' + scc + ' Classifying ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classify_cmd = ''
    
    classify_comment = '\n' + scc + ' Classifying {}\n'.format(fn)
    classify_cmd += classify_comment
    
    classify_cmd += "python run_classifier.py " + slcc + "\n" + \
    	 ' "' + checkpoint_path + '" ' + slcc + '\n' + \
         ' "' + crop_path + '" ' + slcc + '\n' + \
         ' "' + classifier_output_path + '" ' + slcc + '\n' + \
         ' ' + '--detections-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-categories "' + classifier_categories_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--image-size "' + image_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--batch-size "' + batch_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--num-workers "' + num_workers_str + '"' + ' ' + slcc + '\n'
    
    if device_id is not None:
        classify_cmd += ' ' + '--device {}'.format(device_id)
        
    classify_cmd += '\n\n'        
    classify_cmd = '{}'.format(classify_cmd)
    commands.append(classify_cmd)
		

##%% Remap classifier outputs

commands.append('\n' + scc + ' Remapping ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    classifier_output_path_remapped = \
        classifier_output_path.replace(".csv.gz","_remapped.csv.gz")
    assert not (classifier_output_path == classifier_output_path_remapped)
    
    output_label_index = classifier_output_path_remapped.replace(
        "_remapped.csv.gz","_label_index_remapped.json")
                                       
    remap_cmd = ''
    
    remap_comment = '\n' + scc + ' Remapping {}\n'.format(fn)
    remap_cmd += remap_comment
    
    remap_cmd += "python aggregate_classifier_probs.py " + slcc + "\n" + \
        ' "' + classifier_output_path + '" ' + slcc + '\n' + \
        ' ' + '--target-mapping "' + target_mapping_path + '"' + ' ' + slcc + '\n' + \
        ' ' + '--output-csv "' + classifier_output_path_remapped + '"' + ' ' + slcc + '\n' + \
        ' ' + '--output-label-index "' + output_label_index + '"' \
        '\n'
     
    remap_cmd = '{}'.format(remap_cmd)
    commands.append(remap_cmd)
    

##%% Merge classification and detection outputs

commands.append('\n' + scc + ' Merging ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classifier_output_path_remapped = \
        classifier_output_path.replace(".csv.gz","_remapped.csv.gz")
    
    output_label_index = classifier_output_path_remapped.replace(
        "_remapped.csv.gz","_label_index_remapped.json")
    
    final_output_path = os.path.join(output_base,
                                     os.path.basename(classifier_output_path)).\
        replace(classifier_output_suffix,
        final_output_suffix)
    final_output_path = final_output_path.replace('_detections','')
    final_output_path = final_output_path.replace('_crops','')
    final_output_path_mc = final_output_path
    
    merge_cmd = ''
    
    merge_comment = '\n' + scc + ' Merging {}\n'.format(fn)
    merge_cmd += merge_comment
    
    merge_cmd += "python merge_classification_detection_output.py " + slcc + "\n" + \
    	 ' "' + classifier_output_path_remapped + '" ' + slcc + '\n' + \
         ' "' + output_label_index + '" ' + slcc + '\n' + \
         ' ' + '--output-json "' + final_output_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--detection-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-name "' + classifier_name + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + classification_threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--typical-confidence-threshold "' + typical_classification_threshold_str + '"' + '\n' + \
         '\n'
    merge_cmd = '{}'.format(merge_cmd)
    commands.append(merge_cmd)


##%% Write  out classification script

with open(output_file,'w') as f:
    for s in commands:
        f.write('{}'.format(s))

import stat
st = os.stat(output_file)
os.chmod(output_file, st.st_mode | stat.S_IEXEC)


#%% Run a non-MegaClassifier classifier (i.e., a classifier with no output mapping)

classifier_name_short = 'idfgclassifier'
threshold_str = '0.15' # 0.6
classifier_name = 'idfg_classifier_ckpt_14_compiled'

organization_name = organization_name_short
job_name = base_task_name
input_filename = filtered_output_filename # combined_api_output_file
input_files = [input_filename]
image_base = input_path
crop_path = os.path.join(os.path.expanduser('~/crops'),job_name + '_crops')
output_base = combined_api_output_folder
device_id = 1

output_file = os.path.join(filename_base,'run_{}_'.format(classifier_name_short) + job_name +  script_extension)

classifier_base = os.path.expanduser('~/models/camera_traps/idfg_classifier/idfg_classifier_20200905_042558')
assert os.path.isdir(classifier_base)

checkpoint_path = os.path.join(classifier_base,'idfg_classifier_ckpt_14_compiled.pt')
assert os.path.isfile(checkpoint_path)

classifier_categories_path = os.path.join(classifier_base,'label_index.json')
assert os.path.isfile(classifier_categories_path)

classifier_output_suffix = '_{}_output.csv.gz'.format(classifier_name_short)
final_output_suffix = '_{}.json'.format(classifier_name_short)

threshold_str = '0.65'
n_threads_str = str(default_workers_for_parallel_tasks)
image_size_str = '300'
batch_size_str = '64'
num_workers_str = str(default_workers_for_parallel_tasks)
logdir = filename_base

classification_threshold_str = '0.05'

# This is just passed along to the metadata in the output file, it has no impact
# on how the classification scripts run.
typical_classification_threshold_str = '0.75'


##%% Set up environment

commands = []


##%% Crop images
    
commands.append('\n' + scc + ' Cropping ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    crop_cmd = ''
    
    crop_comment = '\n' + scc + ' Cropping {}\n'.format(fn)
    crop_cmd += crop_comment
    
    crop_cmd += "python crop_detections.py " + slcc + "\n" + \
    	 ' "' + input_file_path + '" ' + slcc + '\n' + \
         ' "' + crop_path + '" ' + slcc + '\n' + \
         ' ' + '--images-dir "' + image_base + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--square-crops ' + ' ' + slcc + '\n' + \
         ' ' + '--threads "' + n_threads_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--logdir "' + logdir + '"' + '\n' + \
         '\n'
    crop_cmd = '{}'.format(crop_cmd)
    commands.append(crop_cmd)


##%% Run classifier

commands.append('\n' + scc + ' Classifying ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classify_cmd = ''
    
    classify_comment = '\n' + scc + ' Classifying {}\n'.format(fn)
    classify_cmd += classify_comment
    
    classify_cmd += "python run_classifier.py " + slcc + "\n" + \
    	 ' "' + checkpoint_path + '" ' + slcc + '\n' + \
         ' "' + crop_path + '" ' + slcc + '\n' + \
         ' "' + classifier_output_path + '" ' + slcc + '\n' + \
         ' ' + '--detections-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-categories "' + classifier_categories_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--image-size "' + image_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--batch-size "' + batch_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--num-workers "' + num_workers_str + '"' + ' ' + slcc + '\n'
    
    if device_id is not None:
        classify_cmd += ' ' + '--device {}'.format(device_id)
        
    classify_cmd += '\n\n'    
    classify_cmd = '{}'.format(classify_cmd)
    commands.append(classify_cmd)
		

##%% Merge classification and detection outputs

commands.append('\n' + scc + ' Merging ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    final_output_path = os.path.join(output_base,
                                     os.path.basename(classifier_output_path)).\
                                     replace(classifier_output_suffix,
                                     final_output_suffix)
    final_output_path = final_output_path.replace('_detections','')
    final_output_path = final_output_path.replace('_crops','')
    final_output_path_ic = final_output_path
    
    merge_cmd = ''
    
    merge_comment = '\n' + scc + ' Merging {}\n'.format(fn)
    merge_cmd += merge_comment
    
    merge_cmd += "python merge_classification_detection_output.py " + slcc + "\n" + \
    	 ' "' + classifier_output_path + '" ' + slcc + '\n' + \
         ' "' + classifier_categories_path + '" ' + slcc + '\n' + \
         ' ' + '--output-json "' + final_output_path_ic + '"' + ' ' + slcc + '\n' + \
         ' ' + '--detection-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-name "' + classifier_name + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + classification_threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--typical-confidence-threshold "' + typical_classification_threshold_str + '"' + '\n' + \
         '\n'
    merge_cmd = '{}'.format(merge_cmd)
    commands.append(merge_cmd)


##%% Write everything out

with open(output_file,'w') as f:
    for s in commands:
        f.write('{}'.format(s))

import stat
st = os.stat(output_file)
os.chmod(output_file, st.st_mode | stat.S_IEXEC)


#%% Run the classifier(s) via the .sh script(s) or batch file(s) we just wrote

# I do this manually, primarily because this requires a different mamba environment
# (cameratraps-classifier) from MegaDetector's environment (cameratraps-detector).
#
# The next few pseudo-cells (#%) in this script are basically always run all at once, getting us
# all the way from running the classifier to classification previews and zipped .json files that
# are ready to upload.


#%% Within-image classification smoothing

from collections import defaultdict

#
# Only count detections with a classification confidence threshold above
# *classification_confidence_threshold*, which in practice means we're only
# looking at one category per detection.
#
# If an image has at least *min_detections_above_threshold* such detections
# in the most common category, and no more than *max_detections_secondary_class*
# in the second-most-common category, flip all detections to the most common
# category.
#
# Optionally treat some classes as particularly unreliable, typically used to overwrite an 
# "other" class.
#
# This cell also removes everything but the non-dominant classification for each detection.
#

# How many detections do we need above the classification threshold to determine a dominant category
# for an image?
min_detections_above_threshold = 4

# Even if we have a dominant class, if a non-dominant class has at least this many classifications
# in an image, leave them alone.
max_detections_secondary_class = 3

# If the dominant class has at least this many classifications, overwrite "other" classifications
min_detections_to_overwrite_other = 2
other_category_names = ['other']

# What confidence threshold should we use for assessing the dominant category in an image?
classification_confidence_threshold = 0.6

# Which classifications should we even bother over-writing?
classification_overwrite_threshold = 0.3

# Detection confidence threshold for things we count when determining a dominant class
detection_confidence_threshold = 0.2

# Which detections should we even bother over-writing?
detection_overwrite_threshold = 0.05

classification_detection_files = []

# Did we run MegaClassifier
if final_output_path_mc is not None:
    classification_detection_files.append(final_output_path_mc)
    
# Did we run the IDFG classifier?
if final_output_path_ic is not None:
    classification_detection_files.append(final_output_path_ic)

assert all([os.path.isfile(fn) for fn in classification_detection_files])

smoothed_classification_files = []

for final_output_path in classification_detection_files:

    classifier_output_path = final_output_path
    classifier_output_path_within_image_smoothing = classifier_output_path.replace(
        '.json','_within_image_smoothing.json')
    
    with open(classifier_output_path,'r') as f:
        d = json.load(f)
        
    category_name_to_id = {d['classification_categories'][k]:k for k in d['classification_categories']}
    other_category_ids = []
    for s in other_category_names:
        if s in category_name_to_id:
            other_category_ids.append(category_name_to_id[s])
        else:
            print('Warning: "other" category {} not present in file {}'.format(
                s,classifier_output_path))
    
    n_other_classifications_changed = 0
    n_other_images_changed = 0
    
    n_detections_flipped = 0
    n_images_changed = 0
    
    # Before we do anything else, get rid of everything but the top classification
    # for each detection.
    for im in tqdm(d['images']):
        
        if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
            continue
        
        detections = im['detections']
        
        for det in detections:
            
            if 'classifications' not in det or len(det['classifications']) == 0:
                continue
            
            classification_confidence_values = [c[1] for c in det['classifications']]
            assert is_list_sorted(classification_confidence_values,reverse=True)
            det['classifications'] = [det['classifications'][0]]
    
        # ...for each detection in this image
        
    # ...for each image
    
    # im = d['images'][0]    
    for im in tqdm(d['images']):
        
        if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
            continue
        
        detections = im['detections']
    
        category_to_count = defaultdict(int)
        for det in detections:
            if ('classifications' in det) and (det['conf'] >= detection_confidence_threshold):
                for c in det['classifications']:
                    if c[1] >= classification_confidence_threshold:
                        category_to_count[c[0]] += 1
                # ...for each classification
            # ...if there are classifications for this detection
        # ...for each detection
                        
        if len(category_to_count) <= 1:
            continue
        
        category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                     key=lambda item: item[1], 
                                                     reverse=True)}
        
        keys = list(category_to_count.keys())
        
        # Handle a quirky special case: if the most common category is "other" and 
        # it's "tied" with the second-most-common category, swap them
        if (len(keys) > 1) and \
            (keys[0] in other_category_ids) and \
            (keys[1] not in other_category_ids) and \
            (category_to_count[keys[0]] == category_to_count[keys[1]]):
                keys[1], keys[0] = keys[0], keys[1]
        
        max_count = category_to_count[keys[0]]
        # secondary_count = category_to_count[keys[1]]
        # The 'secondary count' is the most common non-other class
        secondary_count = 0
        for i_key in range(1,len(keys)):
            if keys[i_key] not in other_category_ids:
                secondary_count = category_to_count[keys[i_key]]
                break

        most_common_category = keys[0]
        
        assert max_count >= secondary_count
        
        # If we have at least *min_detections_to_overwrite_other* in a category that isn't
        # "other", change all "other" classifications to that category
        if max_count >= min_detections_to_overwrite_other and \
            most_common_category not in other_category_ids:
            
            other_change_made = False
            
            for det in detections:
                
                if ('classifications' in det) and (det['conf'] >= detection_overwrite_threshold): 
                    
                    for c in det['classifications']:                
                        
                        if c[1] >= classification_overwrite_threshold and \
                            c[0] in other_category_ids:
                                
                            n_other_classifications_changed += 1
                            other_change_made = True
                            c[0] = most_common_category
                            
                    # ...for each classification
                    
                # ...if there are classifications for this detection
                
            # ...for each detection
            
            if other_change_made:
                n_other_images_changed += 1
            
        # ...if we should overwrite all "other" classifications
    
        if max_count < min_detections_above_threshold:
            continue
        
        if secondary_count >= max_detections_secondary_class:
            continue
        
        # At this point, we know we have a dominant category; change all other above-threshold
        # classifications to that category.  That category may have been "other", in which
        # case we may have already made the relevant changes.
        
        n_detections_flipped_this_image = 0
        
        # det = detections[0]
        for det in detections:
            
            if ('classifications' in det) and (det['conf'] >= detection_overwrite_threshold):
                
                for c in det['classifications']:
                    if c[1] >= classification_overwrite_threshold and \
                        c[0] != most_common_category:
                            
                        c[0] = most_common_category
                        n_detections_flipped += 1
                        n_detections_flipped_this_image += 1
                
                # ...for each classification
                
            # ...if there are classifications for this detection
            
        # ...for each detection
        
        if n_detections_flipped_this_image > 0:
            n_images_changed += 1
    
    # ...for each image    
    
    print('Classification smoothing: changed {} detections on {} images'.format(
        n_detections_flipped,n_images_changed))
    
    print('"Other" smoothing: changed {} detections on {} images'.format(
          n_other_classifications_changed,n_other_images_changed))
    
    with open(classifier_output_path_within_image_smoothing,'w') as f:
        json.dump(d,f,indent=1)
        
    print('Wrote results to:\n{}'.format(classifier_output_path_within_image_smoothing))
    smoothed_classification_files.append(classifier_output_path_within_image_smoothing)

# ...for each file we want to smooth


#% Read EXIF data from all images

from data_management import read_exif
exif_options = read_exif.ReadExifOptions()

exif_options.verbose = False
exif_options.n_workers = default_workers_for_parallel_tasks
exif_options.use_threads = parallelization_defaults_to_threads
exif_options.processing_library = 'pil'
exif_options.byte_handling = 'delete'

exif_results_file = os.path.join(filename_base,'exif_data.json')

if os.path.isfile(exif_results_file):
    print('Reading EXIF results from {}'.format(exif_results_file))
    with open(exif_results_file,'r') as f:
        exif_results = json.load(f)
else:        
    exif_results = read_exif.read_exif_from_folder(input_path,
                                                   output_file=exif_results_file,
                                                   options=exif_options)


#% Prepare COCO-camera-traps-compatible image objects for EXIF results

import datetime    
from data_management.read_exif import parse_exif_datetime_string

min_valid_timestamp_year = 2015

now = datetime.datetime.now()

image_info = []

images_without_datetime = []
images_with_invalid_datetime = []

exif_datetime_tag = 'DateTimeOriginal'

# exif_result = exif_results[0]
for exif_result in tqdm(exif_results):
    
    im = {}
    
    # By default we assume that each leaf-node folder is a location
    if overflow_folder_handling_enabled:
        im['location'] = relative_path_to_location(os.path.dirname(exif_result['file_name']))
    else:
        im['location'] = os.path.dirname(exif_result['file_name'])
        
    im['file_name'] = exif_result['file_name']
    im['id'] = im['file_name']
    if ('exif_tags' not in exif_result) or (exif_result['exif_tags'] is None) or \
        (exif_datetime_tag not in exif_result['exif_tags']): 
        exif_dt = None
    else:
        exif_dt = exif_result['exif_tags'][exif_datetime_tag]
        exif_dt = parse_exif_datetime_string(exif_dt)
    if exif_dt is None:
        im['datetime'] = None
        images_without_datetime.append(im['file_name'])
    else:
        dt = exif_dt
        
        # An image from the future (or within the last 24 hours) is invalid
        if (now - dt).total_seconds() <= 1*24*60*60:
            print('Warning: datetime for {} is {}'.format(
                im['file_name'],dt))
            im['datetime'] = None            
            images_with_invalid_datetime.append(im['file_name'])
        
        # An image from before the dawn of time is also invalid
        elif dt.year < min_valid_timestamp_year:
            print('Warning: datetime for {} is {}'.format(
                im['file_name'],dt))
            im['datetime'] = None
            images_with_invalid_datetime.append(im['file_name'])
        
        else:
            im['datetime'] = dt

    image_info.append(im)
    
# ...for each exif image result

print('Parsed EXIF datetime information, unable to parse EXIF data from {} of {} images'.format(
    len(images_without_datetime),len(exif_results)))


#% Assemble into sequences

from collections import defaultdict
from data_management import cct_json_utils

print('Assembling images into sequences')

cct_json_utils.create_sequences(image_info)

# Make a list of images appearing at each location
sequence_to_images = defaultdict(list)

# im = image_info[0]
for im in tqdm(image_info):
    sequence_to_images[im['seq_id']].append(im)

all_sequences = list(sorted(sequence_to_images.keys()))


#% Load classification results

sequence_level_smoothing_input_file = smoothed_classification_files[0]

with open(sequence_level_smoothing_input_file,'r') as f:
    d = json.load(f)

# Map each filename to classification results for that file
filename_to_results = {}

for im in tqdm(d['images']):
    filename_to_results[im['file'].replace('\\','/')] = im


#% Smooth classification results over sequences (prep)

from md_utils.ct_utils import is_list_sorted

classification_category_id_to_name = d['classification_categories']
classification_category_name_to_id = {v: k for k, v in classification_category_id_to_name.items()}

class_names = list(classification_category_id_to_name.values())

animal_detection_category = '1'
assert(d['detection_categories'][animal_detection_category] == 'animal')

other_category_names = set(['other'])
other_category_ids = set([classification_category_name_to_id[s] for s in other_category_names])

# These are the only classes to which we're going to switch other classifications
category_names_to_smooth_to = set(['deer','elk','cow','canid','cat','bird','bear'])
category_ids_to_smooth_to = set([classification_category_name_to_id[s] for s in category_names_to_smooth_to])
assert all([s in class_names for s in category_names_to_smooth_to])    

# Only switch classifications to the dominant class if we see the dominant class at least
# this many times
min_dominant_class_classifications_above_threshold_for_class_smoothing = 5 # 2

# If we see more than this many of a class that are above threshold, don't switch those
# classifications to the dominant class.
max_secondary_class_classifications_above_threshold_for_class_smoothing = 5

# If the ratio between a dominant class and a secondary class count is greater than this, 
# regardless of the secondary class count, switch those classificaitons (i.e., ignore
# max_secondary_class_classifications_above_threshold_for_class_smoothing).
#
# This may be different for different dominant classes, e.g. if we see lots of cows, they really
# tend to be cows.  Less so for canids, so we set a higher "override ratio" for canids.
min_dominant_class_ratio_for_secondary_override_table = {classification_category_name_to_id['cow']:2,None:3}

# If there are at least this many classifications for the dominant class in a sequence,
# regardless of what that class is, convert all 'other' classifications (regardless of 
# confidence) to that class.
min_dominant_class_classifications_above_threshold_for_other_smoothing = 3 # 2

# If there are at least this many classifications for the dominant class in a sequence,
# regardless of what that class is, classify all previously-unclassified detections
# as that class.
min_dominant_class_classifications_above_threshold_for_unclassified_smoothing = 3 # 2

# Only count classifications above this confidence level when determining the dominant
# class, and when deciding whether to switch other classifications.
classification_confidence_threshold = 0.6

# Confidence values to use when we change a detection's classification (the
# original confidence value is irrelevant at that point)
flipped_other_confidence_value = 0.6
flipped_class_confidence_value = 0.6
flipped_unclassified_confidence_value = 0.6

min_detection_confidence_for_unclassified_flipping = 0.15


#% Smooth classification results over sequences (supporting functions)
    
def results_for_sequence(images_this_sequence):
    """
    Fetch MD results for every image in this sequence, based on the 'file_name' field
    """
    
    results_this_sequence = []
    for im in images_this_sequence:
        fn = im['file_name']
        results_this_image = filename_to_results[fn]
        assert isinstance(results_this_image,dict)
        results_this_sequence.append(results_this_image)
        
    return results_this_sequence
            
    
def top_classifications_for_sequence(images_this_sequence):
    """
    Return all top-1 animal classifications for every detection in this 
    sequence, regardless of  confidence

    May modify [images_this_sequence] (removing non-top-1 classifications)
    """
    
    classifications_this_sequence = []

    # im = images_this_sequence[0]
    for im in images_this_sequence:
        
        fn = im['file_name']
        results_this_image = filename_to_results[fn]
        
        if results_this_image['detections'] is None:
            continue
        
        # det = results_this_image['detections'][0]
        for det in results_this_image['detections']:
            
            # Only process animal detections
            if det['category'] != animal_detection_category:
                continue
            
            # Only process detections with classification information
            if 'classifications' not in det:
                continue
            
            # We only care about top-1 classifications, remove everything else
            if len(det['classifications']) > 1:
                
                # Make sure the list of classifications is already sorted by confidence
                classification_confidence_values = [c[1] for c in det['classifications']]
                assert is_list_sorted(classification_confidence_values,reverse=True)
                
                # ...and just keep the first one
                det['classifications'] = [det['classifications'][0]]
                
            # Confidence values should be sorted within a detection; verify this, and ignore 
            top_classification = det['classifications'][0]
            
            classifications_this_sequence.append(top_classification)
    
        # ...for each detection in this image
        
    # ...for each image in this sequence

    return classifications_this_sequence

# ...top_classifications_for_sequence()


def count_above_threshold_classifications(classifications_this_sequence):    
    """
    Given a list of classification objects (tuples), return a dict mapping
    category IDs to the count of above-threshold classifications.
    
    This dict's keys will be sorted in descending order by frequency.
    """
    
    # Count above-threshold classifications in this sequence
    category_to_count = defaultdict(int)
    for c in classifications_this_sequence:
        if c[1] >= classification_confidence_threshold:
            category_to_count[c[0]] += 1
    
    # Sort the dictionary in descending order by count
    category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    keys_sorted_by_frequency = list(category_to_count.keys())
        
    # Handle a quirky special case: if the most common category is "other" and 
    # it's "tied" with the second-most-common category, swap them.
    if len(other_category_names) > 0:
        if (len(keys_sorted_by_frequency) > 1) and \
            (keys_sorted_by_frequency[0] in other_category_names) and \
            (keys_sorted_by_frequency[1] not in other_category_names) and \
            (category_to_count[keys_sorted_by_frequency[0]] == \
             category_to_count[keys_sorted_by_frequency[1]]):
                keys_sorted_by_frequency[1], keys_sorted_by_frequency[0] = \
                    keys_sorted_by_frequency[0], keys_sorted_by_frequency[1]

    sorted_category_to_count = {}    
    for k in keys_sorted_by_frequency:
        sorted_category_to_count[k] = category_to_count[k]
        
    return sorted_category_to_count

# ...def count_above_threshold_classifications()
    
def sort_images_by_time(images):
    """
    Returns a copy of [images], sorted by the 'datetime' field (ascending).
    """
    return sorted(images, key = lambda im: im['datetime'])        
    

def get_first_key_from_sorted_dictionary(di):
    if len(di) == 0:
        return None
    return next(iter(di.items()))[0]


def get_first_value_from_sorted_dictionary(di):
    if len(di) == 0:
        return None
    return next(iter(di.items()))[1]


#% Smooth classifications at the sequence level (main loop)

n_other_flips = 0
n_classification_flips = 0
n_unclassified_flips = 0

# Break if this token is contained in a filename (set to None for normal operation)
debug_fn = None

# i_sequence = 0; seq_id = all_sequences[i_sequence]
for i_sequence,seq_id in tqdm(enumerate(all_sequences),total=len(all_sequences)):
    
    images_this_sequence = sequence_to_images[seq_id]
    
    # Count top-1 classifications in this sequence (regardless of confidence)
    classifications_this_sequence = top_classifications_for_sequence(images_this_sequence)
    
    # Handy debugging code for looking at the numbers for a particular sequence
    for im in images_this_sequence:
        if debug_fn is not None and debug_fn in im['file_name']:
            raise ValueError('')
             
    if len(classifications_this_sequence) == 0:
        continue
    
    # Count above-threshold classifications for each category
    sorted_category_to_count = count_above_threshold_classifications(classifications_this_sequence)
    
    if len(sorted_category_to_count) == 0:
        continue
    
    max_count = get_first_value_from_sorted_dictionary(sorted_category_to_count)    
    dominant_category_id = get_first_key_from_sorted_dictionary(sorted_category_to_count)
    
    # If our dominant category ID isn't something we want to smooth to, don't mess around with this sequence
    if dominant_category_id not in category_ids_to_smooth_to:
        continue
        
    
    ## Smooth "other" classifications ##
    
    if max_count >= min_dominant_class_classifications_above_threshold_for_other_smoothing:        
        for c in classifications_this_sequence:           
            if c[0] in other_category_ids:
                n_other_flips += 1
                c[0] = dominant_category_id
                c[1] = flipped_other_confidence_value


    # By not re-computing "max_count" here, we are making a decision that the count used
    # to decide whether a class should overwrite another class does not include any "other"
    # classifications we changed to be the dominant class.  If we wanted to include those...
    # 
    # sorted_category_to_count = count_above_threshold_classifications(classifications_this_sequence)
    # max_count = get_first_value_from_sorted_dictionary(sorted_category_to_count)    
    # assert dominant_category_id == get_first_key_from_sorted_dictionary(sorted_category_to_count)
    
    
    ## Smooth non-dominant classes ##
    
    if max_count >= min_dominant_class_classifications_above_threshold_for_class_smoothing:
        
        # Don't flip classes to the dominant class if they have a large number of classifications
        category_ids_not_to_flip = set()
        
        for category_id in sorted_category_to_count.keys():
            secondary_class_count = sorted_category_to_count[category_id]
            dominant_to_secondary_ratio = max_count / secondary_class_count
            
            # Don't smooth over this class if there are a bunch of them, and the ratio
            # if primary to secondary class count isn't too large
            
            # Default ratio
            ratio_for_override = min_dominant_class_ratio_for_secondary_override_table[None]
            
            # Does this dominant class have a custom ratio?
            if dominant_category_id in min_dominant_class_ratio_for_secondary_override_table:
                ratio_for_override = \
                    min_dominant_class_ratio_for_secondary_override_table[dominant_category_id]
                    
            if (dominant_to_secondary_ratio < ratio_for_override) and \
                (secondary_class_count > \
                 max_secondary_class_classifications_above_threshold_for_class_smoothing):
                category_ids_not_to_flip.add(category_id)
                
        for c in classifications_this_sequence:
            if c[0] not in category_ids_not_to_flip and c[0] != dominant_category_id:
                c[0] = dominant_category_id
                c[1] = flipped_class_confidence_value
                n_classification_flips += 1
        
        
    ## Smooth unclassified detections ##
        
    if max_count >= min_dominant_class_classifications_above_threshold_for_unclassified_smoothing:
        
        results_this_sequence = results_for_sequence(images_this_sequence)
        detections_this_sequence = []
        for r in results_this_sequence:
            if r['detections'] is not None:
                detections_this_sequence.extend(r['detections'])
        for det in detections_this_sequence:
            if 'classifications' in det and len(det['classifications']) > 0:
                continue
            if det['category'] != animal_detection_category:
                continue
            if det['conf'] < min_detection_confidence_for_unclassified_flipping:
                continue
            det['classifications'] = [[dominant_category_id,flipped_unclassified_confidence_value]]
            n_unclassified_flips += 1
                            
# ...for each sequence    
    
print('\Finished sequence smoothing\n')
print('Flipped {} "other" classifications'.format(n_other_flips))
print('Flipped {} species classifications'.format(n_classification_flips))
print('Flipped {} unclassified detections'.format(n_unclassified_flips))
    

#% Write smoothed classification results

sequence_smoothed_classification_file = sequence_level_smoothing_input_file.replace(
    '.json','_seqsmoothing.json')

print('Writing sequence-smoothed classification results to {}'.format(
    sequence_smoothed_classification_file))

with open(sequence_smoothed_classification_file,'w') as f:
    json.dump(d,f,indent=1)


#% Post-processing (post-classification, post-within-image-and-within-sequence-smoothing)

options = PostProcessingOptions()
options.image_base_dir = input_path
options.include_almost_detections = True
options.num_images_to_sample = 10000
options.confidence_threshold = 0.2
options.classification_confidence_threshold = 0.7
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.max_figures_per_html_file = 2500

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

folder_token = sequence_smoothed_classification_file.split(os.path.sep)[-1].replace(
    '_within_image_smoothing_seqsmoothing','')
folder_token = folder_token.replace('.json','_seqsmoothing')

output_base = os.path.join(postprocessing_output_folder, folder_token + \
    base_task_name + '_{:.3f}'.format(options.confidence_threshold))
os.makedirs(output_base, exist_ok=True)
print('Processing {} to {}'.format(base_task_name, output_base))

options.api_output_file = sequence_smoothed_classification_file
options.output_dir = output_base
ppresults = process_batch_results(options)
path_utils.open_file(ppresults.output_html_file)


#% Zip .json files

from md_utils.path_utils import parallel_zip_files

json_files = os.listdir(combined_api_output_folder)
json_files = [fn for fn in json_files if fn.endswith('.json')]
json_files = [os.path.join(combined_api_output_folder,fn) for fn in json_files]

parallel_zip_files(json_files)


#%% 99.9% of jobs end here

# Everything after this is run ad hoc and/or requires some manual editing.


#%% Compare results files for different model versions (or before/after RDE)

import itertools

from api.batch_processing.postprocessing.compare_batch_results import (
    BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)

options = BatchComparisonOptions()

options.job_name = organization_name_short
options.output_folder = os.path.join(postprocessing_output_folder,'model_comparison')
options.image_folder = input_path

options.pairwise_options = []

filenames = [
    '/postprocessing/organization/mdv4_results.json',
    '/postprocessing/organization/mdv5a_results.json',
    '/postprocessing/organization/mdv5b_results.json'    
    ]

detection_thresholds = [0.7,0.15,0.15]

assert len(detection_thresholds) == len(filenames)

rendering_thresholds = [(x*0.6666) for x in detection_thresholds]

# Choose all pairwise combinations of the files in [filenames]
for i, j in itertools.combinations(list(range(0,len(filenames))),2):
        
    pairwise_options = PairwiseBatchComparisonOptions()
    
    pairwise_options.results_filename_a = filenames[i]
    pairwise_options.results_filename_b = filenames[j]
    
    pairwise_options.rendering_confidence_threshold_a = rendering_thresholds[i]
    pairwise_options.rendering_confidence_threshold_b = rendering_thresholds[j]
    
    pairwise_options.detection_thresholds_a = {'animal':detection_thresholds[i],
                                               'person':detection_thresholds[i],
                                               'vehicle':detection_thresholds[i]}
    pairwise_options.detection_thresholds_b = {'animal':detection_thresholds[j],
                                               'person':detection_thresholds[j],
                                               'vehicle':detection_thresholds[j]}
    options.pairwise_options.append(pairwise_options)

results = compare_batch_results(options)

from md_utils.path_utils import open_file
open_file(results.html_output_file)


#%% Merge in high-confidence detections from another results file

from api.batch_processing.postprocessing.merge_detections import MergeDetectionsOptions,merge_detections

source_files = ['']
target_file = ''
output_file = target_file.replace('.json','_merged.json')

options = MergeDetectionsOptions()
options.max_detection_size = 1.0
options.target_confidence_threshold = 0.25
options.categories_to_include = [1]
options.source_confidence_thresholds = [0.2]
merge_detections(source_files, target_file, output_file, options)

merged_detections_file = output_file


#%% Create a new category for large boxes

from api.batch_processing.postprocessing import categorize_detections_by_size

size_options = categorize_detections_by_size.SizeCategorizationOptions()

size_options.size_thresholds = [0.9]
size_options.size_category_names = ['large_detections']

size_options.categories_to_separate = [1]
size_options.measurement = 'size' # 'width'

threshold_string = '-'.join([str(x) for x in size_options.size_thresholds])

input_file = filtered_output_filename
size_separated_file = input_file.replace('.json','-size-separated-{}.json'.format(
    threshold_string))
d = categorize_detections_by_size.categorize_detections_by_size(input_file,size_separated_file,
                                                                size_options)


#%% Preview large boxes

output_base_large_boxes = os.path.join(postprocessing_output_folder, 
    base_task_name + '_{}_{:.3f}_size_separated_boxes'.format(rde_string, options.confidence_threshold))    
os.makedirs(output_base_large_boxes, exist_ok=True)
print('Processing post-RDE, post-size-separation to {}'.format(output_base_large_boxes))

options.api_output_file = size_separated_file
options.output_dir = output_base_large_boxes

ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
path_utils.open_file(html_output_file)


#%% .json splitting

data = None

from api.batch_processing.postprocessing.subset_json_detector_output import (
    subset_json_detector_output, SubsetJsonDetectorOutputOptions)

input_filename = filtered_output_filename
output_base = os.path.join(combined_api_output_folder,base_task_name + '_json_subsets')

if False:
    if data is None:
        with open(input_filename) as f:
            data = json.load(f)
    print('Data set contains {} images'.format(len(data['images'])))

print('Processing file {} to {}'.format(input_filename,output_base))          

options = SubsetJsonDetectorOutputOptions()
# options.query = None
# options.replacement = None

options.split_folders = True
options.make_folder_relative = True

# Reminder: 'n_from_bottom' with a parameter of zero is the same as 'bottom'
options.split_folder_mode = 'bottom'  # 'top', 'n_from_top', 'n_from_bottom'
options.split_folder_param = 0
options.overwrite_json_files = False
options.confidence_threshold = 0.01

subset_data = subset_json_detector_output(input_filename, output_base, options, data)

# Zip the subsets folder
from md_utils.path_utils import zip_folder
zip_folder(output_base,verbose=True)


#%% Custom splitting/subsetting

data = None

from api.batch_processing.postprocessing.subset_json_detector_output import (
    subset_json_detector_output, SubsetJsonDetectorOutputOptions)

input_filename = filtered_output_filename
output_base = os.path.join(filename_base,'json_subsets')

folders = os.listdir(input_path)

if data is None:
    with open(input_filename) as f:
        data = json.load(f)

print('Data set contains {} images'.format(len(data['images'])))

# i_folder = 0; folder_name = folders[i_folder]
for i_folder, folder_name in enumerate(folders):

    output_filename = os.path.join(output_base, folder_name + '.json')
    print('Processing folder {} of {} ({}) to {}'.format(i_folder, len(folders), folder_name,
          output_filename))

    options = SubsetJsonDetectorOutputOptions()
    options.confidence_threshold = 0.01
    options.overwrite_json_files = True
    options.query = folder_name + '/'

    # This doesn't do anything in this case, since we're not splitting folders
    # options.make_folder_relative = True        
    
    subset_data = subset_json_detector_output(input_filename, output_filename, options, data)


#%% String replacement
    
data = None

from api.batch_processing.postprocessing.subset_json_detector_output import (
    subset_json_detector_output, SubsetJsonDetectorOutputOptions)

input_filename = filtered_output_filename
output_filename = input_filename.replace('.json','_replaced.json')

options = SubsetJsonDetectorOutputOptions()
options.query = folder_name + '/'
options.replacement = ''
subset_json_detector_output(input_filename,output_filename,options)


#%% Splitting images into folders

from api.batch_processing.postprocessing.separate_detections_into_folders import (
    separate_detections_into_folders, SeparateDetectionsIntoFoldersOptions)

default_threshold = 0.2
base_output_folder = os.path.expanduser('~/data/{}-{}-separated'.format(base_task_name,default_threshold))

options = SeparateDetectionsIntoFoldersOptions(default_threshold)

options.results_file = filtered_output_filename
options.base_input_folder = input_path
options.base_output_folder = os.path.join(base_output_folder,folder_name)
options.n_threads = default_workers_for_parallel_tasks
options.allow_existing_directory = False

separate_detections_into_folders(options)


#%% Convert frame-level results to video-level results

# This cell is only useful if the files submitted to this job were generated via
# video_folder_to_frames().

from detection.video_utils import frame_results_to_video_results

video_output_filename = filtered_output_filename.replace('.json','_aggregated.json')
frame_results_to_video_results(filtered_output_filename,video_output_filename)


#%% End notebook: turn this script into a notebook (how meta!)

import os
import nbformat as nbf

input_py_file = os.path.expanduser(
    '~/git/MegaDetector/api/batch_processing/data_preparation/manage_local_batch.py')
assert os.path.isfile(input_py_file)
output_ipynb_file = input_py_file.replace('.py','.ipynb')

nb_header = '# Managing a local MegaDetector batch'

nb_header += '\n'

nb_header += \
"""
This notebook represents an interactive process for running MegaDetector on large batches of images, including typical and optional postprocessing steps.  Everything after "Merge results..." is basically optional, and we typically do a mix of these optional steps, depending on the job.

This notebook is auto-generated from manage_local_batch.py (a cell-delimited .py file that is used the same way, typically in Spyder or VS Code).

"""

with open(input_py_file,'r') as f:
    lines = f.readlines()

i_line = 0

header_comment = ''

lines_to_ignore = 7

# Everything before the first cell is the header comment
while(not lines[i_line].startswith('#%%')):
    if i_line < lines_to_ignore:
        i_line += 1
        continue
    
    s = lines[i_line].replace('#','').strip()
    if len(s) == 0:
        header_comment += '\n\n'
    else:
        header_comment += ' ' + s
    i_line += 1

nb_header += header_comment
nb = nbf.v4.new_notebook()
nb['cells'].append(nbf.v4.new_markdown_cell(nb_header))

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
