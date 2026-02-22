#%% Header

"""

manage_local_batch.py

Semi-automated process for managing a local MegaDetector (and, optionally, SpeciesNet) job,
including standard postprocessing steps.

This script is not intended to be run from top to bottom like a typical Python script,
it's a notebook disguised with a .py extension.  It's the Bestest Most Awesome way to
run MegaDetector, but it's also pretty complex; if you want to play with this, you might
want to check in with cameratraps@lila.science for some tips.  Otherwise... YMMV.

Some general notes on using this script, which I run in VS Code or Spyder, though everything
will be the same if you are reading this in Jupyter Notebook (using the .ipynb version of the
script):

* This script assumes you have set up a Python environment with the MegaDetector python package,
  and that you're running in that environment.  The MegaDetector User Guide has a lot more detail,
  but the gist of the setup is:

  mamba create -n megadetector python=3.11 pip -y
  mamba activate megadetector
  pip install --upgrade megadetector

  ...and if you'll be running SpeciesNet also:

  pip install --upgrade speciesnet

* Typically when I have a MegaDetector job to run, I make a copy of this script.  Let's
  say I'm running a job for an organization called "bibblebop"; I have a big folder of
  job-specific copies of this script, and I might save a new one called "bibblebop-2023-07-26.py"
  (the filename doesn't matter, it just helps me keep these organized).

* There are three variables you need to set in this script before you start running code:
  "input_path", "organization_name_short", and "job_date".  You will get a sensible error if you
  forget to set any of these.  In this case I might set those to "c:/data/camera-trap-stuff",
  "dancorp", and "2023-07-26", respectively.

* After setting the required variables, I run the first few cells - up to and including the one
  called "Generate commands" - which collectively take basically zero seconds.  After you run the
  "Generate commands" cell, you will have a folder that looks something like:

   c:/users/dan/postprocessing/dancorp/dancorp-2023-07-06-mdv5a/

  Everything related to this job - scripts, outputs, intermediate stuff - will be in this folder.
  Specifically, after the "Generate commands" cell, you'll have scripts in that folder called
  something like:

  run_chunk_000_gpu_00.bat (or .sh on Linux)

  Personally, I like to run that script directly in a command prompt (I just leave VS Code or
  Spyder open, though it's OK if that window gets shut down while MD is running).

* Then when the jobs are done, back to the interactive environment!  I run the next few cells,
  which make sure the job finished.  You are very plausibly done at this point, and can ignore
  all the remaining cells.  If you want to do things like repeat detection elimination, or running
  a classifier, or making fancy preview pages, or splitting your results file up in specialized ways,
  there are cells for all of those things, but now you're in power-user territory, so I'm going to
  end this guide here. Email cameratraps@lila.science with questions about the fancy stuff.

"""

#%% Imports and constants

import json
import os
import stat
import time
import re

import humanfriendly
import clipboard # type: ignore #noqa

from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.utils.ct_utils import image_file_to_camera_folder
from megadetector.utils.ct_utils import split_list_into_fixed_size_chunks

from megadetector.detection.run_detector_batch import load_and_run_detector_batch
from megadetector.detection.run_detector_batch import write_results_to_file
from megadetector.detection.run_detector import estimate_md_images_per_second
from megadetector.detection.run_detector import get_detector_version_from_model_file

from megadetector.postprocessing.postprocess_batch_results import PostProcessingOptions
from megadetector.postprocessing.postprocess_batch_results import process_batch_results

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.path_utils import find_images
from megadetector.utils.path_utils import path_join
from megadetector.utils.path_utils import write_list_to_file
from megadetector.utils.path_utils import open_file

from megadetector.utils.wi_taxonomy_utils import generate_md_results_from_predictions_json
from megadetector.utils.wi_taxonomy_utils import generate_instances_json_from_folder

from megadetector.postprocessing.classification_postprocessing import restrict_to_taxa_list

from megadetector.utils.url_utils import SingletonHTTPServer
from megadetector.utils.url_utils import download_url

## Inference options

# Should we allow the cell in this notebook that runs the MD tasks to execute?
#
# Set to False by default, i.e. I normally run the MD tasks at the command prompt.
run_tasks_in_notebook = False

# Turn warnings into errors if more than this many images are missing
max_tolerable_failed_images = 100

# Should we supply the --image_queue_option to run_detector_batch?
use_image_queue = True

# If we are using an image queue (worker pool), should that include image preprocessing
# (as opposed to just image loading)?  Only relevant if use_image_queue is True.
preprocess_on_image_queue = True

# Number of image queue loader workers.  Only relevant if use_image_queue is True.
image_queue_loader_workers = 4

# Only relevant when we're using a single GPU
default_gpu_number = 0

# Should we supply --quiet to run_detector_batch.py?
quiet_mode = True

# Specify a target image size when running MD... strongly recommended to leave this at "None"
# unless you want to live at the cutting edge.
image_size = None

# Should we include image size, timestamp, and/or EXIF data in MD output?
include_image_size = False
include_image_timestamp = False
include_exif_data = False

# String to pass as the "detector_options" parameter to run_detector_batch (or None)
# detector_options = 'compatibility_mode=classic'
# detector_options = 'compatibility_mode=modern'
detector_options = None

# Only relevant when running on CPU
ncores = 1

# If False, we'll load chunk list files if they exist.  Only relevant when
# re-starting this script after an earlier attempt.
force_enumeration = False

# If this is None, we'll prefer threads on Windows, processes on Linux.
parallelization_defaults_to_threads = None

# This is for things like image rendering, not for MegaDetector
default_workers_for_parallel_tasks = 20

# Whether to over-write existing results files; only relevant if re-running this
# script after an earlier attempt.
overwrite_handling = 'skip' # 'skip', 'error', or 'overwrite'

# The function used to get camera names from image paths, used only for repeat
# detection elimination.  This defaults to a standard function (image_file_to_camera_folder)
# that replaces typical strings like "BTCF", "RECNYX001", or "DCIM".  There's an example near
# the end of this notebook of using a custom function instead.
relative_path_to_location = image_file_to_camera_folder

# OS-specific script line continuation character (modified later if we're running on Windows)

if os.name == 'nt':

    script_header = ''
    slcc = '^'
    scc = 'REM'
    script_extension = '.bat'

    # Include this after each command in a .sh/.bat file
    command_suffix = 'if %errorlevel% neq 0 exit /b %errorlevel%\n'

    # My experience has been that Python multiprocessing is flaky on Windows, so
    # default to threads on Windows
    if parallelization_defaults_to_threads is None:
        parallelization_defaults_to_threads = True

else:

    # Force scripts to exit immediately if there's an error
    script_header = '#!/bin/bash\n\nset -e\n'
    slcc = '\\'
    scc = '#'
    script_extension = '.sh'
    command_suffix = ''

    if parallelization_defaults_to_threads is None:
        parallelization_defaults_to_threads = False


## Constants related to using YOLOv5's val.py

# You can almost definitely ignore this section.
#
# If you set use_yolo_inference_scripts to True, we'll use "yolo val" to run
# MD.  This is almost never a good idea, I only use this for esoteric
# reference data jobs.

# Should we use YOLOv5's val.py instead of run_detector_batch.py?
use_yolo_inference_scripts = False

# Directory in which to run val.py (relevant for YOLOv5, not for YOLOv8)
yolo_working_dir = os.path.expanduser('~/git/yolov5')

# Only used for loading the mapping from class indices to names
yolo_dataset_file = None

# 'yolov5' or 'yolov8'; assumes YOLOv5 if this is None
yolo_model_type = None

# Inference batch size
yolo_batch_size = 1

# Should we remove intermediate files used for running YOLOv5's val.py?
#
# Only relevant if use_yolo_inference_scripts is True.
remove_yolo_intermediate_results = True
remove_yolo_symlink_folder = True
use_symlinks_for_yolo_inference = True
write_yolo_debug_output = False

# Should we apply YOLOv5's test-time augmentation?
augment = False


## Constants related to tiled inference

# You can almost definitely ignore this section.
#
# Setting use_tiled_inference to True enables a highly experimental mode where
# we chop each image up into smaller images before running MD.
use_tiled_inference = False

# Should we delete tiles after each job?  Only set this to False for debugging;
# large jobs will take up a lot of space if you keep tiles around after each task.
remove_tiles = True
tile_size = (1280,1280)
tile_overlap = 0.2

# Specify folder for temporary tile storage, default is to use the postprocessing folder
tiling_folder_base = None

## Constants related to preview generation

# Optionally omit non-animal images from the output, useful when animals are rare and
# we want to dial up the total number of images used in the preview
render_animals_only = False

preview_options_base = PostProcessingOptions()
preview_options_base.image_base_dir = None
preview_options_base.include_almost_detections = True
preview_options_base.num_images_to_sample = 7500
preview_options_base.confidence_threshold = 0.2
preview_options_base.almost_detection_confidence_threshold = \
    preview_options_base.confidence_threshold - 0.05
preview_options_base.ground_truth_json_file = None
preview_options_base.separate_detections_by_category = True
preview_options_base.sample_seed = 0
preview_options_base.max_figures_per_html_file = 2500
preview_options_base.sort_classification_results_by_count = True
preview_options_base.parallelize_rendering = True
preview_options_base.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
preview_options_base.parallelize_rendering_with_threads = parallelization_defaults_to_threads
preview_options_base.additional_image_fields_to_display = \
    {'pre_smoothing_description':'pre-smoothing labels',
     'pre_filtering_description':'pre-filtering labels',
     'post_filtering_description':'post-filtering labels',
     'top_classification_common_name':'top class'}
preview_options_base.category_name_to_sort_weight = \
    {'animal':1,'blank':1,'unknown':1,'unreliable':1,'mammal':1}

if render_animals_only:
    preview_options_base.rendering_bypass_sets = \
        ['detections_person','detections_vehicle',
         'detections_person_vehicle','non_detections']
    # preview_options_base.rendering_bypass_sets.append('almost_detections')
    # preview_options_base.rendering_bypass_sets.append('detections_animal_person')


#%% Variables I set for each job

input_path = '/drive/organization'
organization_name_short = 'organization'
job_date = None # '2025-01-01'
model_file = 'MDV5A' # 'MDV5A', 'MDV5B', 'MDV4', 'MDv1000-redwood'

# Number of jobs to split data into, typically equal to the number of available GPUs, though
# when using an image loading queue, I typically use ~10 jobs per GPU;  those serve as de
# facto checkpoints.
n_jobs = 10
n_gpus = 1

# Set to "None" when using an image loading queue, which doesn't currently support
# checkpointing.  If you are using multiple jobs, which is the default, there's almost
# no reason to enable checkpointing.
checkpoint_frequency = None

# Local root folder where we do all our MegaDetector work; results and
# temporary files will be stored in a subfolder for this job
postprocessing_base = os.path.expanduser('~/postprocessing')

# Optional job descriptor (separated by "-", so you don't have to include the delimiter here)
job_tag = None

# SpeciesNet-related variables

# Set to None to use the default SpeciesNet model file
speciesnet_model_file = None # os.path.expanduser('~/models/speciesnet/crop')

country_code = None
state_code = None

# Can be None to run the classifier in a single chunk
max_images_per_chunk = None
classifier_batch_size = 128

# Text file containing binomial names and common names of allowed taxa
custom_taxa_list = None

# If custom_taxa_list is not None, when should we apply the custom taxonomy?  Can be
# 'before_smoothing' or 'after_smoothing'.
custom_taxa_stage = 'before_smoothing'

# If custom_taxa_list is not None, should we propagate labels *down* the taxonomy tree?
# E.g. if your custom list says there's only one carnivore in your ecosystem, should we
# turn *every* carnivore prediction (including "order carnivora") into that taxon?
custom_taxa_allow_walk_down = False

# Only necessary when using a custom taxonomy list
#
# If this is None, the notebook will try to download this file from a standard
# location.
taxonomy_file = None # path_join(postprocessing_base,'taxonomy_release.txt')

# Setting this to True says that if I have two predicted species in the same family
# in a sequence, I will force them all to be the more common species.  Don't set this
# if you have images where multiple species from the same family can occur in the same
# sequence.
allow_same_family_smoothing = False

# Only relevant if you have a .json file you want to use to provide sequence information
# to the sequence-level smoothing process.  Typically only used for LILA-related jobs.
cct_formatted_json = None

# Should we remove SpeciesNet classifications from non-animal detections?
remove_classifications_from_non_animals = True


#%% Derived variables, constant validation, path setup

input_path = input_path.replace('\\','/')

assert not (input_path.endswith('/') or input_path.endswith('\\'))
assert os.path.isdir(input_path), 'Could not find input folder {}'.format(input_path)
assert job_date is not None and organization_name_short != 'organization'

preview_options_base.image_base_dir = input_path

if job_tag is None:
    job_description_string = ''
else:
    job_description_string = '-' + job_tag

# Estimate inference speed for the current GPU
approx_images_per_second = estimate_md_images_per_second(model_file)

# Rough estimate for the inference time cost of augmentation
if augment and (approx_images_per_second is not None):
    approx_images_per_second = approx_images_per_second * 0.7

base_task_name = organization_name_short + '-' + job_date + job_description_string + '-' + \
    get_detector_version_from_model_file(model_file)
base_output_folder_name = \
    path_join(postprocessing_base,organization_name_short)
os.makedirs(base_output_folder_name,exist_ok=True)

if use_image_queue:
    assert checkpoint_frequency is None,\
        'Checkpointing is not supported when using an image queue'
    if preprocess_on_image_queue and (detector_options is not None) and \
        'compatibility_mode=modern' in detector_options:
        raise NotImplementedError('Standalone preprocessing is not yet supported for "modern" preprocessing')
if use_tiled_inference:
    assert not use_yolo_inference_scripts, \
        'Using the YOLO inference script is not supported when using tiled inference'
    assert checkpoint_frequency is None, \
        'Checkpointing is not supported when using tiled inference'

filename_base = path_join(base_output_folder_name, base_task_name)
combined_api_output_folder = path_join(filename_base, 'combined_api_outputs')
postprocessing_output_folder = path_join(filename_base, 'preview')

combined_api_output_file = path_join(
    combined_api_output_folder,
    '{}_detections.json'.format(base_task_name))

# This will be the .json results file after RDE; if this doesn't exist when
# we get to classification stuff, that will indicate that we didn't do RDE.
filtered_output_filename = insert_before_extension(combined_api_output_file,'filtered')

# If we do sequence-level smoothing, we'll read EXIF data and put it here
exif_results_file = path_join(filename_base,'exif_data.json')

os.makedirs(filename_base, exist_ok=True)
os.makedirs(combined_api_output_folder, exist_ok=True)
os.makedirs(postprocessing_output_folder, exist_ok=True)

if input_path.endswith('/'):
    input_path = input_path[0:-1]

print('Output folder:\n{}'.format(filename_base))

if custom_taxa_list is not None:

    assert os.path.isfile(custom_taxa_list), \
        'Could not find custom taxa file {}'.format(custom_taxa_list)
    assert custom_taxa_stage in ('before_smoothing','after_smoothing')

    if taxonomy_file is None:

        local_taxonomy_file = os.path.join(filename_base,'taxonomy_release.txt')
        if os.path.isfile(local_taxonomy_file):
            print('Found previously-downloaded taxonomy file at {}'.format(local_taxonomy_file))
            taxonomy_file = local_taxonomy_file
        else:
            taxonomy_file_url = \
                'https://lila.science/speciesnet-taxonomy-file'
            print('Attempting to download taxonomy file from {}'.format(taxonomy_file_url))
            download_url(taxonomy_file_url,
                         destination_filename=local_taxonomy_file,
                         progress_updater=True,
                         force_download=False,
                         verbose=True,
                         escape_spaces=True)
            taxonomy_file = local_taxonomy_file

    # Validate the species list
    restrict_to_taxa_list(taxa_list=custom_taxa_list,
                          speciesnet_taxonomy_file=taxonomy_file,
                          input_file=None,
                          output_file=None,
                          use_original_common_names_if_available=True)


#%% Enumerate files

# Have we already listed files for this job?
chunk_file_base = path_join(filename_base,'file_chunks')
os.makedirs(chunk_file_base,exist_ok=True)

chunk_files = os.listdir(chunk_file_base)
pattern = re.compile(r'chunk\d+.json')
chunk_files = [fn for fn in chunk_files if pattern.match(fn)]

if (not force_enumeration) and (len(chunk_files) > 0):

    print('Found {} chunk files in folder {}, bypassing enumeration'.format(
        len(chunk_files),
        filename_base))

    all_images = []
    for fn in chunk_files:
        with open(path_join(chunk_file_base,fn),'r') as f:
            chunk = json.load(f)
            assert isinstance(chunk,list)
            all_images.extend(chunk)
    all_images = sorted(all_images)

    print('Loaded {} image files from {} chunks in {}'.format(
        len(all_images),len(chunk_files),chunk_file_base))

else:

    print('Enumerating image files in {}'.format(input_path))

    all_images = sorted(find_images(input_path,recursive=True,convert_slashes=True))

    # It's common to run this notebook on an external drive with the main folders in the drive root
    all_images = [fn for fn in all_images if not \
                  (fn.startswith('$RECYCLE') or fn.startswith('System Volume Information'))]

    print('\nEnumerated {} image files in {}'.format(len(all_images),input_path))


#%% Divide images into chunks

folder_chunks = split_list_into_n_chunks(all_images,n_jobs)


#%% Estimate total time

if approx_images_per_second is None:

    print("Can't estimate inference time for the current environment")

else:

    n_images = len(all_images)
    execution_seconds = n_images / approx_images_per_second
    wallclock_seconds = execution_seconds / n_gpus
    print('Expected time: {}'.format(humanfriendly.format_timespan(wallclock_seconds)))

    seconds_per_chunk = len(folder_chunks[0]) / approx_images_per_second
    print('Expected time per chunk: {}'.format(humanfriendly.format_timespan(seconds_per_chunk)))


#%% Write file lists

task_info = []

for i_chunk,chunk_list in enumerate(folder_chunks):

    chunk_fn = path_join(chunk_file_base,'chunk{}.json'.format(str(i_chunk).zfill(3)))
    task_info.append({'id':i_chunk,'input_file':chunk_fn})
    write_list_to_file(chunk_fn, chunk_list)


#%% Generate commands

# A list of the scripts tied to each GPU, as absolute paths.  We'll write this out at
# the end so each GPU's list of commands can be run at once
gpu_to_scripts = defaultdict(list)

detector_chunk_base = path_join(filename_base,'detector_commands')
os.makedirs(detector_chunk_base,exist_ok=True)

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):

    chunk_file = task['input_file']
    checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')

    output_fn = chunk_file.replace('.json','_results.json')

    task['output_file'] = output_fn

    if n_gpus > 1:
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
        else:
            augment_string = '--augment_enabled 0'

        batch_string = '--batch_size {}'.format(yolo_batch_size)

        symlink_folder = path_join(filename_base,'symlinks','symlinks_{}'.format(
            str(i_task).zfill(3)))
        yolo_results_folder = path_join(filename_base,'yolo_results','yolo_results_{}'.format(
            str(i_task).zfill(3)))

        symlink_folder_string = '--symlink_folder "{}"'.format(symlink_folder)
        yolo_results_folder_string = '--yolo_results_folder "{}"'.format(yolo_results_folder)

        remove_symlink_folder_string = ''
        if not remove_yolo_symlink_folder:
            remove_symlink_folder_string = '--no_remove_symlink_folder'

        write_yolo_debug_output_string = ''
        if write_yolo_debug_output:
            write_yolo_debug_output = '--write_yolo_debug_output'

        remove_yolo_results_string = ''
        if not remove_yolo_intermediate_results:
            remove_yolo_results_string = '--no_remove_yolo_results_folder'

        cmd = ''

        device_string = '--device {}'.format(gpu_number)

        overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)

        cmd += f'python -m megadetector.detection.run_inference_with_yolov5_val "{model_file}" "{chunk_file}" "{output_fn}" '
        cmd += f'{image_size_string} {augment_string} '
        cmd += f'{symlink_folder_string} {yolo_results_folder_string} {remove_yolo_results_string} '
        cmd += f'{remove_symlink_folder_string} {device_string} '
        cmd += f'{overwrite_handling_string} {batch_string} {write_yolo_debug_output_string}'

        if yolo_working_dir is not None:
            cmd += f' --yolo_working_folder "{yolo_working_dir}"'
        if yolo_dataset_file is not None:
            cmd += ' --yolo_dataset_file "{}"'.format(yolo_dataset_file)
        if yolo_model_type is not None:
            cmd += ' --model_type {}'.format(yolo_model_type)

        if not use_symlinks_for_yolo_inference:
            cmd += ' --no_use_symlinks'

        cmd += '\n'

    elif use_tiled_inference:

        if tiling_folder_base is None:
            tiling_folder_base = filename_base
        tiling_folder = path_join(tiling_folder_base,'tile_cache','tile_cache_{}'.format(
            str(i_task).zfill(3)))

        if os.name == 'nt':
            cuda_string = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
        else:
            cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number} '

        cmd = f'{cuda_string} python -m megadetector.detection.run_tiled_inference '
        cmd += f'"{model_file}" "{input_path}" "{tiling_folder}" "{output_fn}"'

        cmd += f' --image_list "{chunk_file}"'
        cmd += f' --overwrite_handling {overwrite_handling}'

        if augment:
            cmd += ' --augment'

        if not remove_tiles:
            cmd += ' --no_remove_tiles'

        if image_size is not None:
            cmd += ' --inference_size {}'.format(image_size)

        # If we're using non-default tile sizes
        if tile_size is not None and (tile_size[0] > 0 or tile_size[1] > 0):
            cmd += ' --tile_size_x {} --tile_size_y {}'.format(tile_size[0],tile_size[1])

        if tile_overlap is not None:
            cmd += f' --tile_overlap {tile_overlap}'

        if image_queue_loader_workers is not None:
            cmd += ' --loader_workers {}'.format(image_queue_loader_workers)

        cmd += ' --n_patch_extraction_workers 4'

    else:

        if os.name == 'nt':
            cuda_string = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
        else:
            cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number} '

        checkpoint_frequency_string = ''
        checkpoint_path_string = ''

        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
            checkpoint_path_string = '--checkpoint_path "{}"'.format(checkpoint_filename)

        use_image_queue_string = ''
        if (use_image_queue):
            use_image_queue_string = '--use_image_queue'
            if preprocess_on_image_queue:
                use_image_queue_string += ' --preprocess_on_image_queue'
            if image_queue_loader_workers is not None:
                use_image_queue_string += ' --loader_workers {}'.format(image_queue_loader_workers)

        ncores_string = ''
        if (ncores > 1):
            ncores_string = '--ncores {}'.format(ncores)

        quiet_string = ''
        if quiet_mode:
            quiet_string = '--quiet'

        overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)
        cmd = f'{cuda_string} python -m megadetector.detection.run_detector_batch '
        cmd += f'"{model_file}" "{chunk_file}" "{output_fn}"'
        cmd += f'{checkpoint_frequency_string} {checkpoint_path_string} {use_image_queue_string}'
        cmd += f'{ncores_string} {quiet_string} {image_size_string} {overwrite_handling_string}'

        if include_image_size:
            cmd += ' --include_image_size'
        if include_image_timestamp:
            cmd += ' --include_image_timestamp'
        if include_exif_data:
            cmd += ' --include_exif_data'
        if augment:
            if image_size is None:
                print('\n** Warning: you are using --augment with the default image size, '
                      'you may want to use a larger image size **\n')
            cmd += ' --augment'

        if detector_options is not None:
            cmd += ' --detector_options "{}"'.format(detector_options)

    cmd_file = path_join(filename_base,'detector_commands',
                            'run_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                            str(gpu_number).zfill(2),script_extension))

    with open(cmd_file,'w') as f:

        # This writes, e.g. "set -e"
        if script_header is not None and len(script_header) > 0:
            f.write(script_header + '\n')

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

        resume_cmd_file = path_join(filename_base,'detector_commands',
                                       'resume_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                                       str(gpu_number).zfill(2),script_extension))

        with open(resume_cmd_file,'w') as f:

            # This writes, e.g. "set -e"
            if script_header is not None and len(script_header) > 0:
                f.write(script_header + '\n')

            f.write(resume_cmd + '\n')

        st = os.stat(resume_cmd_file)
        os.chmod(resume_cmd_file, st.st_mode | stat.S_IEXEC)

        task['resume_command'] = resume_cmd
        task['resume_command_file'] = resume_cmd_file

# ...for each task

# Write out a script for each GPU that runs all of the commands associated with
# that GPU.
scripts_to_run = []
for gpu_number in gpu_to_scripts:

    gpu_script_file = path_join(filename_base,'run_all_for_gpu_{}{}'.format(
        str(gpu_number).zfill(2),script_extension))
    scripts_to_run.append(gpu_script_file)

    with open(gpu_script_file,'w') as f:

        # This writes, e.g. "set -e"
        if script_header is not None and len(script_header) > 0:
            f.write(script_header + '\n')

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

print('Scripts you probably want to run now:\n')
for s in scripts_to_run:
    print(s)

# import clipboard; clipboard.copy(scripts_to_run[0])


#%% Run the tasks

run_tasks_in_notebook = True

r"""
tl;dr: I almost never run this cell, i.e. "run_tasks_in_notebook" is almost always set to False.

Long version...

The cells we've run so far wrote out some shell scripts (.bat files on Windows,
.sh files on Linx/Mac) that will run MegaDetector.  I like to leave the interactive
environment at this point and run those scripts at the command line.  So, for example,
if you're on Windows, and you've basically used the default values above, there will be
batch files called, e.g.:

c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_000_gpu_00.bat
c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_001_gpu_01.bat

All of that said, you don't *have* to do this at the command line.  The following cell
runs these scripts programmatically, so if you set "run_tasks_in_notebook" to "True"
and run this cell, you can run MegaDetector without leaving this notebook.

One downside of the programmatic approach is that this cell doesn't yet parallelize over
multiple processes, so the tasks will run serially.  This only matters if you have
multiple GPUs.
"""

if run_tasks_in_notebook:

    assert not use_yolo_inference_scripts, \
        'If you want to use the YOLOv5 inference scripts, you can\'t run the model interactively (yet)'
    assert not use_tiled_inference, \
        'If you want to use tiled inference, you can\'t run the model interactively (yet)'

    # i_task = 0; task = task_info[i_task]
    for i_task,task in enumerate(task_info):

        chunk_file = task['input_file']
        output_fn = task['output_file']

        checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')

        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            cp_freq_arg = checkpoint_frequency
        else:
            cp_freq_arg = -1

        start_time = time.time()
        results = load_and_run_detector_batch(model_file=model_file,
                                              image_file_names=chunk_file,
                                              checkpoint_path=checkpoint_filename,
                                              checkpoint_frequency=cp_freq_arg,
                                              results=None,
                                              n_cores=ncores,
                                              # Minimize the risk of IPython process issues
                                              use_image_queue=False,
                                              quiet=quiet_mode,
                                              image_size=image_size,
                                              augment=augment,
                                              detector_options=detector_options)
        elapsed = time.time() - start_time

        print('Task {}: finished inference for {} images in {}'.format(
            i_task, len(results),humanfriendly.format_timespan(elapsed)))

        # This will write absolute paths to the file, we'll fix this later
        write_results_to_file(results, output_fn, detector_file=model_file)

        if (checkpoint_frequency is not None) and (checkpoint_frequency > 0):
            if os.path.isfile(checkpoint_filename):
                os.remove(checkpoint_filename)
                print('Deleted checkpoint file {}'.format(checkpoint_filename))

    # ...for each chunk

# ...if we're running tasks in this notebook


#%% Load results, look for failed or missing images in each task

# Check that all task output files exist

missing_output_files = []

# i_task = 0; task = task_info[i_task]
for i_task,task in tqdm(enumerate(task_info),total=len(task_info)):
    output_file = task['output_file']
    if not os.path.isfile(output_file):
        missing_output_files.append(output_file)

if len(missing_output_files) > 0:
    print('Missing {} output files:'.format(len(missing_output_files)))
    for s in missing_output_files:
        print(s)
    raise Exception('Missing output files')

n_total_failures = 0

# i_task = 0; task = task_info[i_task]
for i_task,task in tqdm(enumerate(task_info),total=len(task_info)):

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

        # For paths to be relative.  The only semi-common scenario when this
        # won't already be the case is when we're using tiled inference.
        if not os.path.isabs(im['file']):
            fn = path_join(input_path,im['file'])
            im['file'] = fn
        assert im['file'].startswith(input_path)
        assert im['file'] in task_images_set
        filename_to_results[im['file']] = im
        if 'failure' in im:
            assert im['failure'] is not None
            n_task_failures += 1

    task['n_failures'] = n_task_failures
    task['results'] = task_results

    for fn in task_images:
        assert fn in filename_to_results, \
            'File {} not found in results for task {}'.format(fn,i_task)

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

for i_task,task in tqdm(enumerate(task_info),total=len(task_info)):

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

# Convert to relative paths, preserving '/' as the path separator, regardless of OS
for im in combined_results['images']:
    assert '\\' not in im['file']
    assert im['file'].startswith(input_path)
    if input_path.endswith(':'):
        im['file'] = im['file'].replace(input_path,'',1)
    else:
        im['file'] = im['file'].replace(input_path + '/','',1)

with open(combined_api_output_file,'w') as f:
    json.dump(combined_results,f,indent=1)

print('\nWrote results to {}'.format(combined_api_output_file))


#%% Preview (pre-RDE)

"""
NB: I almost never run this cell.  This previews the results *before* repeat detection
elimination (RDE), but since I'm essentially always doing RDE, I'm basically never
interested in this preview.  There is a similar cell below for previewing results
*after* RDE, which I almost always run.
"""

preview_options = deepcopy(preview_options_base)

preview_folder = path_join(postprocessing_output_folder,
    base_task_name + '_{:.3f}'.format(preview_options.confidence_threshold))
preview_options.md_results_file = combined_api_output_file
preview_options.output_dir = preview_folder

print('Generating pre-RDE preview in {}'.format(preview_folder))
ppresults = process_batch_results(preview_options)
open_file(ppresults.output_html_file, attempt_to_open_in_wsl_host=True, browser_name='chrome')
# SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% Repeat detection elimination, phase 1

from megadetector.postprocessing.repeat_detection_elimination import repeat_detections_core

task_index = 0

options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.1
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
options.detectionTilesMaxCrops = 80

# options.lineThickness = 5
# options.boxExpansion = 8

options.customDirNameFunction = relative_path_to_location

# To invoke custom collapsing of folders for a particular naming scheme
# options.customDirNameFunction = custom_relative_path_to_location

# To treat a specific folder level as a camera, frequently used when the leaf
# folders each contain frames extracted from a single video
#
# Setting this value to 0 is the same as treating each leaf folder as a camera.
#
# options.nDirLevelsFromLeaf = 1

options.imageBase = input_path
rde_string = 'rde_{:.3f}_{:.3f}_{}_{:.3f}'.format(
    options.confidenceMin, options.iouThreshold,
    options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
options.outputBase = path_join(filename_base, rde_string + '_task_{}'.format(task_index))
options.filenameReplacements = None # {'':''}

# Exclude people and vehicles from RDE
# options.excludeClasses = [2,3]

# options.maxImagesPerFolder = 50000
# options.includeFolders = ['a/b/c','d/e/f']
# options.excludeFolders = ['a/b/c','d/e/f']

options.debugMaxDir = -1
options.debugMaxRenderDir = -1
options.debugMaxRenderDetection = -1
options.debugMaxRenderInstance = -1

# Can be None, 'xsort', or 'clustersort'
options.smartSort = 'xsort'

suspicious_detection_results = repeat_detections_core.find_repeat_detections(combined_api_output_file,
                                                                             output_file_name=None,
                                                                             options=options)


#%% Manual RDE step

## DELETE THE VALID DETECTIONS ##

# Running this line will open the folder up in your file browser
open_file(os.path.dirname(suspicious_detection_results.filterFile),
          attempt_to_open_in_wsl_host=True)

#
# If you ran the previous cell, but then you change your mind and you don't want to do
# the RDE step, that's fine, but don't just blast through this cell once you've run the
# previous cell.  If you do that, you're implicitly telling the notebook that you looked
# at everything in that folder, and confirmed there were no red boxes on animals.
#
# Instead, either change "filtered_output_filename" below to "combined_api_output_file",
# or delete *all* the images in the filtering folder.
#


#%% Re-filtering

from megadetector.postprocessing.repeat_detection_elimination import remove_repeat_detections

remove_repeat_detections.remove_repeat_detections(
    input_file=combined_api_output_file,
    output_file=filtered_output_filename,
    filtering_dir=os.path.dirname(suspicious_detection_results.filterFile)
    )


#%% Preview (post-RDE)

preview_options = deepcopy(preview_options_base)

preview_folder = path_join(postprocessing_output_folder,
    base_task_name + '_{}_{:.3f}'.format(rde_string, preview_options.confidence_threshold))
preview_options.md_results_file = filtered_output_filename
preview_options.output_dir = preview_folder

print('Generating post-RDE preview in {}'.format(preview_folder))
ppresults = process_batch_results(preview_options)
open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% SpeciesNet derived constants

## Detector/cropping constants

# A results file in MD format, referring to the original images
detection_results_file_with_crop_ids = path_join(combined_api_output_folder,
                                                    base_task_name + '-detection_results_with_crop_ids.json')

# A results file in MD format, referring to the crops, so every detection
# has bbox [0,0,1,1]
detection_results_file_for_crop_folder = insert_before_extension(
        detection_results_file_with_crop_ids,'unity_boxes')

# The folder where crops will be placed after running the detector
crop_folder = path_join(postprocessing_base,'crops',base_task_name)

# A detection results file in SpeciesNet format, referring to the crops, so every detection
# has bbox [0,0,1,1]
crop_detections_predictions_file = \
    insert_before_extension(detection_results_file_for_crop_folder,'speciesnet_format')

# The instances.json file that refers just to the crops folder
crop_instances_json = path_join(combined_api_output_folder,
                                   base_task_name + '-crop_instances.json')


## Classification constants

# The instances.json file we use to pass path names and the country code to the
# classifier and ensemble
instances_json = \
    path_join(combined_api_output_folder,
                 base_task_name + '-instances.json')

# The results of the classifier (in SpeciesNet format), after running it on the crops
classifier_output_file_modular_crops = \
    path_join(combined_api_output_folder,
                 base_task_name + '-classifier_output_modular_crops.json')

# The folder where we'll store classifier results for each chunk
#
# (...if we're breaking classification into chunks).
chunk_folder = path_join(filename_base,'classifier_chunks')


## Ensemble constants

# The results of the ensemble, after running it on the crops (in SpeciesNet format)
ensemble_output_file_modular_crops = \
    path_join(combined_api_output_folder,
                 base_task_name + '-ensemble_output_modular_crops.json')

# The results of the ensemble after running it on the crops (in MD format)
ensemble_output_file_crops_md_format = insert_before_extension(
    ensemble_output_file_modular_crops,
    'md-format')

# The results of the ensemble, mapped back to image level (in MD format)
ensemble_output_file_image_level_md_format = \
    ensemble_output_file_crops_md_format.replace('_crops','_image-level')


## Smoothing constants

# The ensemble results (in MD format) after image-level smoothing
classifier_output_path_within_image_smoothing = insert_before_extension(
    ensemble_output_file_image_level_md_format,'within_image_smoothing')

sequence_smoothed_classification_file = \
    insert_before_extension(classifier_output_path_within_image_smoothing,
                            'seqsmoothing')

custom_taxa_output_file = insert_before_extension(
    ensemble_output_file_image_level_md_format,'custom-species-{}'.format(custom_taxa_stage))


## Miscellaneous

geofence_footer = None

if filtered_output_filename is not None and os.path.isfile(filtered_output_filename):
    print('Using filtered MD output file {} for classification'.format(filtered_output_filename))
    detector_output_file_md_format = filtered_output_filename
else:
    print('It looks like you didn\'t do RDE, using raw MD output for classification')
    detector_output_file_md_format = combined_api_output_file

if os.path.isdir(crop_folder):
    print(f'*** Warning: crop folder {crop_folder} exists, if you create new '
           'crops in an existing folder, odd things can happen ***')
os.makedirs(crop_folder,exist_ok=True)

for fn in [classifier_output_file_modular_crops,
           ensemble_output_file_modular_crops]:
    if os.path.exists(fn):
        print('**\nWarning, file {} exists, this is OK if you are resuming\n**\n'.format(fn))

assert (custom_taxa_list is not None) or (country_code is not None), \
    'Did you mean to specify a country code?'

if country_code == 'USA' and state_code is None:
    print('*** Did you mean to specify a state code? ***')


#%% Major fork here, depending on whether we are running in the notebook

if run_tasks_in_notebook:

    pass


    #%% Run SpeciesNet here in the notebook

    from megadetector.detection.run_md_and_speciesnet import \
        RunMDSpeciesNetOptions, run_md_and_speciesnet

    md_speciesnet_options = RunMDSpeciesNetOptions()

    md_speciesnet_options.source = input_path
    md_speciesnet_options.output_file = ensemble_output_file_image_level_md_format
    md_speciesnet_options.classifier_batch_size = classifier_batch_size
    md_speciesnet_options.skip_video = True
    md_speciesnet_options.verbose = True

    # This is not necessary in VS Code, but it's necessary in Spyder
    md_speciesnet_options.worker_type = 'thread'

    md_speciesnet_options.detections_file = detector_output_file_md_format

    if speciesnet_model_file is not None:
        md_speciesnet_options.classification_model = speciesnet_model_file

    # Enable geofencing if (a) we have a country code and (b) we're not immediately
    # applying a custom taxa list
    enable_geofence = True

    if country_code is None:
        enable_geofence = False
    if (custom_taxa_list is not None) and (custom_taxa_stage == 'before_smoothing'):
        enable_geofence = False

    if enable_geofence:
        md_speciesnet_options.country = country_code
        if state_code is not None:
            md_speciesnet_options.admin1_region = state_code

    run_md_and_speciesnet(md_speciesnet_options)

    assert os.path.isfile(ensemble_output_file_image_level_md_format)


#%% The next few cells set up the command line tools to run SpeciesNet

if not run_tasks_in_notebook:

    pass

    #%% Generate instances.json

    # ...for the original images.

    instances = generate_instances_json_from_folder(folder=input_path,
                                                    country=country_code,
                                                    admin1_region=state_code,
                                                    output_file=instances_json,
                                                    filename_replacements=None)

    print('Generated {} instances'.format(len(instances['instances'])))


    #%% Generate crop dataset

    from megadetector.postprocessing.create_crop_folder import \
        CreateCropFolderOptions, create_crop_folder

    create_crop_folder_options = CreateCropFolderOptions()
    create_crop_folder_options.n_workers = 8
    create_crop_folder_options.pool_type = 'process'
    if parallelization_defaults_to_threads:
        create_crop_folder_options.pool_type = 'thread'

    create_crop_folder(input_file=detector_output_file_md_format,
                       input_folder=input_path,
                       output_folder=crop_folder,
                       output_file=detection_results_file_with_crop_ids,
                       crops_output_file=detection_results_file_for_crop_folder,
                       options=create_crop_folder_options)

    assert os.path.isfile(detection_results_file_with_crop_ids)
    assert os.path.isfile(detection_results_file_for_crop_folder)
    assert os.path.isdir(crop_folder)


    #%% Convert crop metadata to SpeciesNet input format

    # Convert detection results for the crops to predictions.json format
    #
    # This will be the input to the ensemble when we run it on the crops.

    from megadetector.utils.wi_taxonomy_utils import generate_predictions_json_from_md_results

    _ = generate_predictions_json_from_md_results(md_results_file=detection_results_file_for_crop_folder,
                                                  predictions_json_file=crop_detections_predictions_file,
                                                  base_folder=crop_folder)


    # Generate a new instances.json file for the crops

    crop_instances = generate_instances_json_from_folder(folder=crop_folder,
                                                         country=country_code,
                                                         admin1_region=state_code,
                                                         output_file=crop_instances_json,
                                                         filename_replacements=None)

    print('Generated {} instances for the crop folder (in file {})'.format(
        len(crop_instances['instances']),crop_instances_json))


    #%% Run SpeciesNet on crops

    os.makedirs(chunk_folder,exist_ok=True)

    print('Reading crop instances json...')

    with open(crop_instances_json,'r') as f:
        crop_instances_dict = json.load(f)

    crop_instances = crop_instances_dict['instances']

    if max_images_per_chunk is None:
        chunks = split_list_into_n_chunks(crop_instances,n_gpus)
    else:
        chunks = split_list_into_fixed_size_chunks(crop_instances,max_images_per_chunk)
    print('Split {} crop instances into {} chunks'.format(len(crop_instances),len(chunks)))

    chunk_scripts = []

    print('Reading detection results...')

    with open(crop_detections_predictions_file,'r') as f:
        detections = json.load(f)

    detection_filepath_to_instance = {p['filepath']:p for p in detections['predictions']}

    chunk_prediction_files = []

    gpu_to_classifier_scripts = defaultdict(list)

    # i_chunk = 0; chunk = chunks[i_chunk]
    for i_chunk,chunk in enumerate(chunks):

        if n_gpus > 1:
            gpu_number = i_chunk % n_gpus
        else:
            gpu_number = default_gpu_number

        if default_gpu_number is not None:
            if os.name == 'nt':
                cuda_prefix = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
            else:
                cuda_prefix = f'CUDA_VISIBLE_DEVICES={gpu_number} '
        else:
            cuda_prefix = ''

        chunk_str = str(i_chunk).zfill(3)

        chunk_instances_json = path_join(chunk_folder,'crop_instances_chunk_{}.json'.format(
            chunk_str))
        chunk_instances_dict = {'instances':chunk}
        with open(chunk_instances_json,'w') as f:
            json.dump(chunk_instances_dict,f,indent=1)

        chunk_detections_json = path_join(chunk_folder,'detections_chunk_{}.json'.format(
            chunk_str))

        detection_predictions_this_chunk = []

        images_this_chunk = [instance['filepath'] for instance in chunk]

        for image_fn in images_this_chunk:
            assert image_fn in detection_filepath_to_instance
            detection_predictions_this_chunk.append(detection_filepath_to_instance[image_fn])

        detection_predictions_dict = {'predictions':detection_predictions_this_chunk}

        with open(chunk_detections_json,'w') as f:
            json.dump(detection_predictions_dict,f,indent=1)

        chunk_files = [instance['filepath'] for instance in chunk]

        chunk_predictions_json = path_join(chunk_folder,'predictions_chunk_{}.json'.format(
            chunk_str))

        if os.path.isfile(chunk_predictions_json):
            print('Warning: chunk output file {} exists'.format(chunk_predictions_json))

        chunk_prediction_files.append(chunk_predictions_json)

        cmd = 'python -m speciesnet.scripts.run_model --classifier_only'
        if speciesnet_model_file is not None:
            cmd += ' --model "{}"'.format(speciesnet_model_file)
        cmd += ' --instances_json "{}"'.format(chunk_instances_json)
        cmd += ' --predictions_json "{}"'.format(chunk_predictions_json)
        cmd += ' --detections_json "{}"'.format(chunk_detections_json)
        cmd += ' --ignore_existing_predictions'

        if classifier_batch_size is not None:
            cmd += ' --batch_size {}'.format(classifier_batch_size)

        chunk_script_file = path_join(chunk_folder,'run_chunk_{}{}'.format(chunk_str,script_extension))

        with open(chunk_script_file,'w') as f:
            # This writes, e.g. "set -e"
            if (script_header is not None) and (len(script_header) > 0):
                f.write(script_header + '\n')
            f.write(cuda_prefix + cmd)

        st = os.stat(chunk_script_file)
        os.chmod(chunk_script_file, st.st_mode | stat.S_IEXEC)

        gpu_to_classifier_scripts[gpu_number].append(chunk_script_file)

    # ...for each chunk

    per_gpu_scripts = []

    # Write out a script for each GPU that runs all of the commands associated with
    # that GPU.
    for gpu_number in gpu_to_classifier_scripts:

        gpu_script_file = path_join(filename_base,'run_classifier_for_gpu_{}{}'.format(
            str(gpu_number).zfill(2),script_extension))
        per_gpu_scripts.append(gpu_script_file)

        with open(gpu_script_file,'w') as f:

            # This writes, e.g. "set -e"
            if (script_header is not None) and (len(script_header) > 0):
                f.write(script_header)

            for script_name in gpu_to_classifier_scripts[gpu_number]:

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

    print('\nClassification scripts you should run now:')
    for s in per_gpu_scripts:
        print(s)

    # import clipboard; clipboard.copy(per_gpu_scripts[0])


    #%% Prepare rollup/geofencing script

    ##%%# Merge crop classification result batches

    from megadetector.utils.wi_taxonomy_utils import merge_prediction_json_files

    merge_prediction_json_files(input_prediction_files=chunk_prediction_files,
                                output_prediction_file=classifier_output_file_modular_crops)


    ##%% Validate crop classification results

    from megadetector.utils.wi_taxonomy_utils import validate_predictions_file
    _ = validate_predictions_file(classifier_output_file_modular_crops,crop_instances_json)


    ##%% Run rollup (and possibly geofencing) (still crops)

    # It doesn't matter here which environment we use, and there's no need to add the CUDA prefix
    ensemble_commands = []

    cmd = 'python -m speciesnet.scripts.run_model --ensemble_only'
    if speciesnet_model_file is not None:
            cmd += ' --model "{}"'.format(speciesnet_model_file)
    cmd += ' --instances_json "{}"'.format(crop_instances_json)
    cmd += ' --classifications_json "{}"'.format(classifier_output_file_modular_crops)
    cmd += ' --detections_json "{}"'.format(crop_detections_predictions_file)
    cmd += ' --predictions_json "{}"'.format(ensemble_output_file_modular_crops)
    cmd += ' --ignore_existing_predictions'

    # Currently we only skip the geofence if we're imminently going to apply a custom taxa
    # list, otherwise the smoothing is quite messy.
    if (custom_taxa_list is not None) and (custom_taxa_stage == 'before_smoothing'):
        cmd += ' --nogeofence'

    ensemble_commands.append(cmd)

    ensemble_cmd = '\n\n'.join(ensemble_commands)
    # print(ensemble_cmd); clipboard.copy(ensemble_cmd)

    print('Ensemble command you should run now:\n\n{}'.format(ensemble_cmd))


    #%% Validate ensemble results and bring crop results back to image level

    ##%% Validate ensemble results (still crops)

    from megadetector.utils.wi_taxonomy_utils import validate_predictions_file
    _ = validate_predictions_file(ensemble_output_file_modular_crops,crop_instances_json)


    ##%% Convert output file to MD format (still crops)

    assert os.path.isfile(ensemble_output_file_modular_crops)

    generate_md_results_from_predictions_json(predictions_json_file=ensemble_output_file_modular_crops,
                                              md_results_file=ensemble_output_file_crops_md_format,
                                              base_folder=crop_folder+'/')


    ##%% Bring those crop-level results back to image level

    from megadetector.postprocessing.create_crop_folder import crop_results_to_image_results

    assert '_crops' in ensemble_output_file_crops_md_format

    crop_results_to_image_results(
        image_results_file_with_crop_ids=detection_results_file_with_crop_ids,
        crop_results_file=ensemble_output_file_crops_md_format,
        output_file=ensemble_output_file_image_level_md_format)

    assert os.path.isfile(ensemble_output_file_image_level_md_format)

    #%%

# ...are we running SpeciesNet in the notebook?


#%% Confirm that all the right images are in the classification results

import json
from megadetector.utils.path_utils import find_images

with open(ensemble_output_file_image_level_md_format,'r') as f:
    d = json.load(f)

filenames_in_results = set([im['file'] for im in d['images']])
images_in_folder = set(find_images(input_path,recursive=True,return_relative_paths=True))

for fn in filenames_in_results:
    assert fn in images_in_folder, \
        'Image {} present in results but not in folder'.format(fn)

for fn in images_in_folder:
    assert fn in filenames_in_results, \
        'Image {} present in folder but not in results'.format(fn)

n_failures = 0

# im = d['images'][0]
for im in d['images']:
    if 'failure' in im:
        n_failures += 1

print('Loaded results for {} images with {} failures'.format(
    len(images_in_folder),n_failures))


#%% Generate a list of corrections made by geofencing, and counts (still crops)

from megadetector.utils.wi_taxonomy_utils import find_geofence_adjustments, \
    generate_geofence_adjustment_html_summary

rollup_pair_to_count = find_geofence_adjustments(ensemble_output_file_modular_crops,
                                                    use_latin_names=False)

geofence_footer = generate_geofence_adjustment_html_summary(rollup_pair_to_count)

# If we didn't run geofencing, there should have been no geofence adjustments
if (custom_taxa_list is not None) and (custom_taxa_stage == 'before_smoothing'):
    assert len(rollup_pair_to_count) == 0
    assert len(geofence_footer) == 0


#%% Preview (post-classification, pre-smoothing/pre-custom-taxa)

preview_options = deepcopy(preview_options_base)

preview_folder = path_join(postprocessing_output_folder,
    base_task_name + '_{}_classification'.format(preview_options.confidence_threshold))
preview_options.md_results_file = ensemble_output_file_image_level_md_format
preview_options.output_dir = preview_folder
preview_options.footer_text = geofence_footer

print('Generating post-classification preview in {}'.format(preview_folder))
ppresults = process_batch_results(preview_options)
open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% Possibly apply a custom taxa list (before smoothing)

if (custom_taxa_list is not None) and (custom_taxa_stage == 'before_smoothing'):

    print('Restricting to custom taxonomy list: {}'.format(custom_taxa_list))

    taxa_list = custom_taxa_list
    restrict_to_taxa_list(taxa_list=taxa_list,
                          speciesnet_taxonomy_file=taxonomy_file,
                          input_file=ensemble_output_file_image_level_md_format,
                          output_file=custom_taxa_output_file,
                          allow_walk_down=custom_taxa_allow_walk_down,
                          use_original_common_names_if_available=True)

else:

    print('No custom taxonomy list supplied, skipping taxonomic restriction step')

pre_smoothing_file = ensemble_output_file_image_level_md_format
if os.path.isfile(custom_taxa_output_file):
    pre_smoothing_file = custom_taxa_output_file


#%% Possibly remove SpeciesNet results from non-animal detections

from megadetector.postprocessing.classification_postprocessing import \
    remove_classifications_from_non_animal_detections

if remove_classifications_from_non_animals:

    output_file = insert_before_extension(pre_smoothing_file,'animal_classifications_only')

    remove_classifications_from_non_animal_detections(pre_smoothing_file,output_file)
    pre_smoothing_file = output_file


#%% Preview (post-custom-taxa, pre-smoothing)

if (custom_taxa_list is not None) and (custom_taxa_stage == 'before_smoothing'):

    preview_options = deepcopy(preview_options_base)

    preview_folder = path_join(postprocessing_output_folder,
        base_task_name + '_{}_customtaxa'.format(preview_options.confidence_threshold))
    preview_options.md_results_file = pre_smoothing_file
    preview_options.output_dir = preview_folder
    preview_options.footer_text = geofence_footer

    print('Generating post-classification preview in {}'.format(preview_folder))
    ppresults = process_batch_results(preview_options)
    open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
    # SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% Within-image classification smoothing

from megadetector.postprocessing.classification_postprocessing import \
    smooth_classification_results_image_level, \
    ClassificationSmoothingOptions

within_image_smoothing_options = ClassificationSmoothingOptions()

if allow_same_family_smoothing:
    within_image_smoothing_options.max_detections_nondominant_class_same_family = 10000

_ = smooth_classification_results_image_level(input_file=pre_smoothing_file,
                                              output_file=classifier_output_path_within_image_smoothing,
                                              options=within_image_smoothing_options)


#%% Preview (post-within-image smoothing)

preview_options = deepcopy(preview_options_base)

preview_folder = path_join(postprocessing_output_folder,
    base_task_name + '_{}_within-image-smoothing'.format(preview_options.confidence_threshold))
preview_options.md_results_file = classifier_output_path_within_image_smoothing
preview_options.output_dir = preview_folder

print('Generating post-within-image-smoothing preview in {}'.format(preview_folder))
ppresults = process_batch_results(preview_options)
open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% Build sequences

# ...from EXIF timestamps, folder structure, or an existing .json file

# How should we determine sequence information?

# Use 'exif' for most image (non-video) cases
sequence_method = 'exif'

# Use 'folder' when leaf node folders are sequences, typically when each folder really
# represents frames from a single video.
# sequence_method = 'folder'

# Use 'json' when you already have a CCT-formatted .json file that has the fields
# "seq_id", "seq_num_frames", and "frame_num".  In this case, cct_formatted_json
# should be set to a valid .json file above.
# sequence_method = 'json'

##%% If we're building sequence information based on EXIF data

if sequence_method == 'exif':

    pass

    ##%% Read EXIF date and time from all images

    from megadetector.data_management import read_exif
    exif_options = read_exif.ReadExifOptions()

    exif_options.verbose = False
    exif_options.n_workers = default_workers_for_parallel_tasks
    exif_options.use_threads = parallelization_defaults_to_threads
    exif_options.processing_library = 'pil'
    exif_options.byte_handling = 'delete'
    exif_options.tags_to_include = ['DateTime','DateTimeOriginal']

    if os.path.isfile(exif_results_file):
        print('Reading EXIF data from {}'.format(exif_results_file))
        with open(exif_results_file,'r') as f:
            exif_results = json.load(f)
    else:
        exif_results = read_exif.read_exif_from_folder(input_path,
                                                       output_file=exif_results_file,
                                                       options=exif_options)


    ##%% Prepare COCO-camera-traps-compatible image objects for EXIF results

    # ...and add location/datetime info based on filenames and EXIF information.

    from megadetector.data_management.read_exif import \
        exif_results_to_cct, ExifResultsToCCTOptions
    from megadetector.utils.ct_utils import is_function_name

    exif_results_to_cct_options = ExifResultsToCCTOptions()

    exif_data_in_cct_format_file = path_join(filename_base,'exif_data_in_cct_format.json')

    if os.path.isfile(exif_data_in_cct_format_file):

        print('Reading CCT-formatted EXIF data from {}'.format(exif_data_in_cct_format_file))

        with open(exif_data_in_cct_format_file,'r') as f:
            cct_dict = json.load(f)

    else:

        # If we've defined a "custom_relative_path_to_location" location, which by convention
        # is what we use in this notebook for a non-standard location mapping function, use it
        # to parse locations when creating the CCT data.
        if is_function_name('custom_relative_path_to_location',locals()):
            print('Using custom location mapping function in EXIF conversion')
            exif_results_to_cct_options.filename_to_location_function = \
                custom_relative_path_to_location # type: ignore # noqa

        cct_dict = exif_results_to_cct(exif_results=exif_results,
                                       cct_output_file=exif_data_in_cct_format_file,
                                       options=exif_results_to_cct_options)


    ##%% Assemble images into sequences

    from megadetector.data_management import cct_json_utils
    from megadetector.data_management.cct_json_utils import SequenceOptions

    sequence_options = SequenceOptions()

    print('Assembling images into sequences')
    _ = cct_json_utils.create_sequences(cct_dict, options=sequence_options)


##%% If we're building sequence information based on folder structure

elif sequence_method == 'folder':

    pass

    ##%% Read the list of filenames

    input_file_for_sequence_aggregation = classifier_output_path_within_image_smoothing
    with open(input_file_for_sequence_aggregation,'r') as f:
        d = json.load(f)


    ##%% Synthesize sequences

    cct_dict = {'info':{},'annotations':[],'categories':[],'images':[]}

    folder_name_to_images = defaultdict(list) # noqa
    images_out = []

    # im_in = d['images'][0]
    for im_in in tqdm(d['images']):

        folder_name = os.path.dirname(im_in['file']).replace('\\','/')
        folder_name_to_images[folder_name].append(im_in['file'])

        im_out = {}
        images_out.append(im_out)

        im_out['file_name'] = im_in['file']
        im_out['seq_id'] = folder_name

        # Not required for smoothing
        # im_out['frame_num'] = len(folder_name_to_images[folder_name]) - 1
        # location_name = os.path.dirname(folder_name).replace('\\','/')
        # im_out['location'] = location_name

    cct_dict['images'] = images_out

    print('Extracted {} sequences from {} images'.format(
        len(folder_name_to_images),len(d['images'])))


##%% If we're loading sequence information from an existing json file

else:

    assert sequence_method == 'json'
    assert cct_formatted_json is not None
    print('Loading sequence information from {}'.format(cct_formatted_json))

    with open(cct_formatted_json,'r') as f:
        cct_dict = json.load(f)
        for im in cct_dict['images']:
            for field_name in ('seq_id','seq_num_frames','frame_num'):
                assert field_name in im, 'Image {} is missing field {}'.format(
                    im['file_name'],field_name)


#%% Sequence-level smoothing

from megadetector.postprocessing.classification_postprocessing import \
    smooth_classification_results_sequence_level, \
    ClassificationSmoothingOptions

input_file_for_sequence_level_smoothing = None
if os.path.isfile(classifier_output_path_within_image_smoothing):
    print('Using within-image smoothing results for sequence-level smoothing')
    input_file_for_sequence_level_smoothing = \
        classifier_output_path_within_image_smoothing
else:
    assert os.path.isfile(ensemble_output_file_image_level_md_format)
    print('Using ensemble output file for sequence-level smoothing (no image-level smoothing file found)')
    input_file_for_sequence_level_smoothing = \
        ensemble_output_file_image_level_md_format

sequence_level_smoothing_options = ClassificationSmoothingOptions()

if allow_same_family_smoothing:
    sequence_level_smoothing_options.max_detections_nondominant_class_same_family = 10000

_ = smooth_classification_results_sequence_level(input_file=input_file_for_sequence_level_smoothing,
                                                 cct_sequence_information=cct_dict,
                                                 output_file=sequence_smoothed_classification_file,
                                                 options=sequence_level_smoothing_options)


#%% Preview (post-sequence-smoothing)

preview_options = deepcopy(preview_options_base)

preview_folder = path_join(postprocessing_output_folder,
    base_task_name + '_{}_sequence-smoothing'.format(preview_options.confidence_threshold))
preview_options.md_results_file = sequence_smoothed_classification_file
preview_options.output_dir = preview_folder
preview_options.footer_text = geofence_footer
preview_options.category_name_to_sort_weight = \
    {'animal':1,'blank':1,'unknown':1,'unreliable':1,'mammal':1,'no cv result':1}

print('Generating post-sequence-smoothing preview in {}'.format(preview_folder))
ppresults = process_batch_results(preview_options)
open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% Possibly apply a custom taxa list (after smoothing)

if (custom_taxa_list is not None) and (custom_taxa_stage == 'after_smoothing'):

    taxa_list = custom_taxa_list
    custom_taxa_output_file = insert_before_extension(
        sequence_smoothed_classification_file,'custom-species')

    restrict_to_taxa_list(taxa_list=taxa_list,
                          speciesnet_taxonomy_file=taxonomy_file,
                          input_file=sequence_smoothed_classification_file,
                          output_file=custom_taxa_output_file,
                          allow_walk_down=custom_taxa_allow_walk_down,
                          use_original_common_names_if_available=True)


#%% Preview (post-custom_taxa-smoothing)

if (custom_taxa_list is not None) and (custom_taxa_stage == 'after_smoothing'):

    preview_options = deepcopy(preview_options_base)

    preview_folder = path_join(postprocessing_output_folder,
        base_task_name + '_{}_custom_taxa'.format(preview_options.confidence_threshold))
    preview_options.md_results_file = custom_taxa_output_file
    preview_options.output_dir = preview_folder
    preview_options.footer_text = geofence_footer

    print('Generating post-sequence-smoothing preview in {}'.format(preview_folder))
    ppresults = process_batch_results(preview_options)
    open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
    # SingletonHTTPServer.start_server(preview_folder,port=8000); open_file('http://localhost:8000')


#%% Remove unused categories

from megadetector.postprocessing.subset_json_detector_output import \
    SubsetJsonDetectorOutputOptions, subset_json_detector_output

from megadetector.postprocessing.validate_batch_results import \
    ValidateBatchResultsOptions, validate_batch_results

input_fn_abs = sequence_smoothed_classification_file
output_fn_abs = insert_before_extension(input_fn_abs,'trimmed')

options = SubsetJsonDetectorOutputOptions()
options.remove_classification_categories_below_count = 1
options.overwrite_json_files = True
_ = subset_json_detector_output(input_fn_abs, output_fn_abs, options)

validation_options = ValidateBatchResultsOptions()
validation_options.raise_errors = True
_ = validate_batch_results(output_fn_abs, validation_options)


#%% Zip .json files

from megadetector.utils.path_utils import parallel_zip_files

json_files = os.listdir(combined_api_output_folder)
json_files = [fn for fn in json_files if fn.endswith('.json')]
json_files = [path_join(combined_api_output_folder,fn) for fn in json_files]

parallel_zip_files(json_files,overwrite=True)


#%% 99.9% of jobs end here

# The remaining cells are run often, but not all the time.
#
# See manage_local_batch_scrap.py for additional cells I sometimes run at this point.


#%% .json splitting

data = None

from megadetector.postprocessing.subset_json_detector_output import \
    subset_json_detector_output, SubsetJsonDetectorOutputOptions

# input_filename = output_fn_abs
input_filename = '/a/b/c/file.json'
output_base = path_join(combined_api_output_folder,base_task_name + '_json_subsets')

print('Processing file {} to {}'.format(input_filename,output_base))

options = SubsetJsonDetectorOutputOptions()
# options.query = None
# options.replacement = None

options.split_folders = True
options.make_folder_relative = True

# Reminder: 'n_from_bottom' with a parameter of zero is the same as 'bottom'
options.split_folder_mode = 'bottom'  # 'bottom', 'n_from_top', 'n_from_bottom'
options.split_folder_param = 0
options.overwrite_json_files = False
options.confidence_threshold = 0.01

subset_data = subset_json_detector_output(input_filename, output_base, options, data)

# Zip the subsets folder
from megadetector.utils.path_utils import zip_folder
zip_folder(output_base,verbose=True)


#%% Custom splitting/subsetting

data = None

from megadetector.postprocessing.subset_json_detector_output import \
    subset_json_detector_output, SubsetJsonDetectorOutputOptions

input_filename = filtered_output_filename
output_base = path_join(filename_base,'json_subsets')

folders = os.listdir(input_path)

if data is None:
    with open(input_filename) as f:
        data = json.load(f)

print('Data set contains {} images'.format(len(data['images'])))

# i_folder = 0; folder_name = folders[i_folder]
for i_folder, folder_name in enumerate(folders):

    output_filename = path_join(output_base, folder_name + '.json')
    print('Processing folder {} of {} ({}) to {}'.format(i_folder, len(folders), folder_name,
          output_filename))

    options = SubsetJsonDetectorOutputOptions()
    options.confidence_threshold = 0.01
    options.overwrite_json_files = True
    options.query = folder_name + '/'

    # This doesn't do anything in this case, since we're not splitting folders
    # options.make_folder_relative = True

    subset_data = subset_json_detector_output(input_filename, output_filename, options, data)


#%% Sample custom path replacement function

def custom_relative_path_to_location(relative_path):

    relative_path = relative_path.replace('\\','/')
    tokens = relative_path.split('/')

    # This example uses a hypothetical (but relatively common) scheme
    # where the first two slash-separated tokens define a site, e.g.
    # where filenames might look like:
    #
    # north_fork/site001/recnyx001/image001.jpg
    location_name = '/'.join(tokens[0:2])
    return location_name


#%% Test relative_path_to_location on the current dataset

with open(combined_api_output_file,'r') as f:
    d = json.load(f)
image_filenames = [im['file'] for im in d['images']]

location_names = set()

# relative_path = image_filenames[0]
for relative_path in tqdm(image_filenames):

    # Use the standard replacement function
    # location_name = relative_path_to_location(relative_path)

    # Use a custom replacement function
    location_name = custom_relative_path_to_location(relative_path)

    location_names.add(location_name)

location_names = list(location_names)
location_names.sort()

for s in location_names:
    print(s)


#%% End notebook: turn this script into a notebook (how meta!)

import os # type: ignore
import nbformat as nbf # type: ignore

if os.name == 'nt':
    git_base = r'c:\git'
else:
    git_base = os.path.expanduser('~/git')

input_py_file = git_base + '/MegaDetector/notebooks/manage_local_batch.py'
assert os.path.isfile(input_py_file)
output_ipynb_file = input_py_file.replace('.py','.ipynb')

nb_header = '# Managing a local MegaDetector batch'

nb_header += '\n'

nb_header += \
"""
This notebook represents an interactive process for running MegaDetector and SpeciesNet on large batches of images, including typical and optional postprocessing steps.  Everything after "Merge results..." is basically optional, and we typically do a mix of these optional steps, depending on the job.

This notebook is auto-generated from manage_local_batch.py (a cell-delimited .py file that is used the same way, typically in Spyder or VS Code).

"""

with open(input_py_file,'r') as f:
    lines = f.readlines()

header_comment = ''

assert lines[0].strip() == '#%% Header'
assert lines[1].strip() == ''
assert lines[2].strip() == '"""'
assert lines[3].strip() == ''
assert lines[4].strip() == 'manage_local_batch.py'
assert lines[5].strip() == ''

i_line = 6

# Everything before the first non-header cell is the header comment
while(not lines[i_line].startswith('#%%')):

    s_raw = lines[i_line]
    s_trimmed = s_raw.strip()

    # Ignore the closing quotes at the end of the header
    if (s_trimmed == '"""'):
        i_line += 1
        continue

    if len(s_trimmed) == 0:
        header_comment += '\n\n'
    else:
        header_comment += ' ' + s_raw
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
