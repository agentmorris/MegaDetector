"""

speciesnet_multispecies_driver.py
   
Semi-automated process for managing a local SpeciesNet job, including
standard postprocessing steps.  This version handles multi-species images, at the 
expense of leveraging the complete ensemble logic.  If multi-species images
are rare in your data, consider using speciesnet_driver.py instead.

"""

#%% Imports

import os
import json
import stat

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.wi_utils import generate_md_results_from_predictions_json
from megadetector.utils.wi_utils import generate_instances_json_from_folder
from megadetector.utils.ct_utils import split_list_into_fixed_size_chunks

import clipboard # noqa


#%% Constants I set for each job

organization_name = 'organization_name'
job_name = 'job_name'

input_folder = '/stuff/input_folder'
assert not input_folder.endswith('/')
model_file = os.path.expanduser('~/models/speciesnet/crop')

# If None, will create a folder in ~/crops
crop_folder = None

# If None, will put a file inside [crop_folder]
crop_json_filename = None

country_code = None
state_code = None

speciesnet_folder = os.path.expanduser('~/git/cameratrapai')
speciesnet_pt_environment_name = 'speciesnet-package-pytorch'
speciesnet_tf_environment_name = 'speciesnet-package-tf'

# Can be None to omit the CUDA prefix
gpu_number = 0
    
# This is not related to running the model, only to postprocessing steps
# in this notebook.  Threads work better on Windows, processes on Linux.
use_threads_for_parallelization = (os.name == 'nt')
max_images_per_chunk = 5000
classifier_batch_size = 128


#%% Temp

print('**** fix me ****')

crop_folder = 'g:/temp/water-hole/crops'
model_file = 'g:/temp/speciesnet/crop'
input_folder = 'g:/temp/water-hole/Sample data/ShareXScreenShot_NambiaWaterhole/Oct_2024'

speciesnet_pt_environment_name = 'speciesnet'
speciesnet_tf_environment_name = 'speciesnet'

if os.name != 'nt':
    from megadetector.utils.path_utils import windows_path_to_wsl_path
    crop_folder = windows_path_to_wsl_path(crop_folder)
    model_file = windows_path_to_wsl_path(model_file)
    input_folder = windows_path_to_wsl_path(input_folder)
    speciesnet_pt_environment_name = 'speciesnet-package-pytorch'
    speciesnet_tf_environment_name = 'speciesnet-package-tf'
    
crop_json_filename = os.path.join(crop_folder,'crop_dataset.json')
gpu_number = None
organization_name = 'multispecies-test'
job_name = 'multispecies-test'
country_code = 'NAM'
classifier_batch_size = 16


#%% Validate constants, prepare folders and dependent constants

if crop_folder is None:
    crop_folder = os.path.join(os.path.expanduser('~/crops'),job_name)

if crop_json_filename is None:    
    crop_json_filename = os.path.join(crop_folder,'crop_dataset' + job_name + '.json')
    
if gpu_number is not None:
    cuda_prefix = 'export CUDA_VISIBLE_DEVICES={} && '.format(gpu_number)
else:
    cuda_prefix = ''

assert organization_name != 'organization_name'
assert job_name != 'job_name'

output_base = os.path.join(os.path.expanduser('~/postprocessing'),organization_name,job_name)
os.makedirs(output_base,exist_ok=True)
preview_folder_base = os.path.join(output_base,'preview')
instances_json = os.path.join(output_base,'instances.json')

assert os.path.isdir(speciesnet_folder)
assert os.path.isdir(input_folder)


#%% Generate instances.json

instances = generate_instances_json_from_folder(folder=input_folder,
                                                country=country_code,
                                                admin1_region=state_code,
                                                output_file=instances_json,
                                                filename_replacements=None)

print('Generated {} instances'.format(len(instances['instances'])))


#%% Prep

detector_output_file_modular = \
    os.path.join(output_base,job_name + '-detector_output_modular.json')
classifier_output_file_modular = \
    os.path.join(output_base,job_name + '-classifier_output_modular.json')
ensemble_output_file_modular = \
    os.path.join(output_base,job_name + '-ensemble_output_modular.json')

for fn in [detector_output_file_modular,classifier_output_file_modular,ensemble_output_file_modular]:
    if os.path.exists(fn):
        print('** Warning, file {} exists, this is OK if you are resuming **\n'.format(fn))


#%% Run detector

detector_commands = []
detector_commands.append(f'{cuda_prefix} cd {speciesnet_folder} && mamba activate {speciesnet_pt_environment_name}')

cmd = 'python speciesnet/scripts/run_model.py --detector_only --model "{}"'.format(model_file)
cmd += ' --instances_json "{}"'.format(instances_json)
cmd += ' --predictions_json "{}"'.format(detector_output_file_modular)
detector_commands.append(cmd)

detector_cmd = '\n\n'.join(detector_commands)
# print(detector_cmd); clipboard.copy(detector_cmd)


#%% Validate detector results

from megadetector.utils.wi_utils import validate_predictions_file
_ = validate_predictions_file(detector_output_file_modular,instances_json)


#%% Convert detection results to MD format

detector_output_file_md_format = insert_before_extension(detector_output_file_modular,
                                                         'md-format')

generate_md_results_from_predictions_json(predictions_json_file=detector_output_file_modular,
                                          md_results_file=detector_output_file_md_format,
                                          base_folder=input_folder+'/')


#%% Generate crop dataset

from megadetector.postprocessing.create_crop_folder import \
    CreateCropFolderOptions, create_crop_folder
    
create_crop_folder_options = CreateCropFolderOptions()

create_crop_folder(input_file=detector_output_file_md_format,
                   input_folder=input_folder,
                   output_folder=crop_folder,
                   output_file=crop_json_filename,
                   options=create_crop_folder_options)

assert os.path.isfile(crop_json_filename)
assert os.path.isdir(crop_folder)


#%% Generate new instances.json file for crops

crop_instances_json = insert_before_extension(crop_json_filename,'instances')

crop_instances = generate_instances_json_from_folder(folder=crop_folder,
                                                     country=country_code,
                                                     admin1_region=state_code,
                                                     output_file=crop_instances_json,
                                                     filename_replacements=None)

print('Generated {} instances for the crop folder'.format(len(crop_instances['instances'])))


#%% Run classifier on crops
   
chunk_folder = os.path.join(output_base,'chunks')
os.makedirs(chunk_folder,exist_ok=True)

print('Reading crop instances json...')

with open(crop_instances_json,'r') as f:
    crop_instances_dict = json.load(f)

crop_instances = crop_instances_dict['instances']
       
chunks = split_list_into_fixed_size_chunks(crop_instances,max_images_per_chunk)
print('Split {} crop instances into {} chunks'.format(len(crop_instances),len(chunks)))

chunk_scripts = []

chunk_prediction_files = []
# i_chunk = 0; chunk = chunks[i_chunk]
for i_chunk,chunk in enumerate(chunks):
    
    chunk_str = str(i_chunk).zfill(3)
    
    chunk_instances_json = os.path.join(chunk_folder,'crop_instances_chunk_{}.json'.format(
        chunk_str))
    chunk_instances_dict = {'instances':chunk}
    with open(chunk_instances_json,'w') as f:
        json.dump(chunk_instances_dict,f,indent=1)
    
    chunk_files = [instance['filepath'] for instance in chunk]
    # image_fn = chunk_files[0]
    
    chunk_predictions_json = os.path.join(chunk_folder,'predictions_chunk_{}.json'.format(
        chunk_str))
    
    if os.path.isfile(chunk_predictions_json):
        print('Warning: chunk output file {} exists'.format(chunk_predictions_json))
        
    chunk_prediction_files.append(chunk_predictions_json)
    
    chunk_script = os.path.join(chunk_folder,'run_chunk_{}.sh'.format(i_chunk))
    cmd = 'python speciesnet/scripts/run_model.py --classifier_only --model "{}"'.format(
        model_file)
    cmd += ' --instances_json "{}"'.format(chunk_instances_json)
    cmd += ' --predictions_json "{}"'.format(chunk_predictions_json)
    cmd += ' --bypass_prompts'
    
    if classifier_batch_size is not None:
       cmd += ' --batch_size {}'.format(classifier_batch_size)
       
    chunk_script_file = os.path.join(chunk_folder,'run_chunk_{}.sh'.format(chunk_str))
    with open(chunk_script_file,'w') as f:
        f.write(cmd)
    st = os.stat(chunk_script_file)
    os.chmod(chunk_script_file, st.st_mode | stat.S_IEXEC)
    
    chunk_scripts.append(chunk_script_file)
    
# ...for each chunk

classifier_script_file = os.path.join(output_base,'run_all_classifier_chunks.sh')            
   
classifier_init_cmd = f'{cuda_prefix} cd {speciesnet_folder} && mamba activate {speciesnet_tf_environment_name}'
with open(classifier_script_file,'w') as f:
    f.write('set -e\n')
    # f.write(classifier_init_cmd + '\n')
    for s in chunk_scripts:
        f.write(s + '\n')

st = os.stat(classifier_script_file)
os.chmod(classifier_script_file, st.st_mode | stat.S_IEXEC)
   
classifier_cmd = '\n\n'.join([classifier_init_cmd,classifier_script_file])
# print(classifier_cmd); clipboard.copy(classifier_cmd)
    

#%% Merge crop classification results

from megadetector.utils.wi_utils import merge_prediction_json_files

merge_prediction_json_files(input_prediction_files=chunk_prediction_files,
                            output_prediction_file=classifier_output_file_modular)
    

#%% Validate crop classification results

from megadetector.utils.wi_utils import validate_predictions_file
_ = validate_predictions_file(classifier_output_file_modular,crop_instances_json)


#%% Run geofencing

# It doesn't matter here which environment we use
ensemble_commands = []
ensemble_commands.append(f'{cuda_prefix} cd {speciesnet_folder} && mamba activate {speciesnet_pt_environment_name}')

cmd = 'python speciesnet/scripts/run_model.py --ensemble_only --model "{}"'.format(model_file)
cmd += ' --instances_json "{}"'.format(crop_instances_json)
cmd += ' --predictions_json "{}"'.format(ensemble_output_file_modular)
cmd += ' --classifications_json "{}"'.format(classifier_output_file_modular)
cmd += ' --bypass_prompts'
ensemble_commands.append(cmd)

ensemble_cmd = '\n\n'.join(ensemble_commands)
# print(ensemble_cmd); clipboard.copy(ensemble_cmd)


#%% Validate ensemble results

from megadetector.utils.wi_utils import validate_predictions_file
validate_predictions_file(ensemble_output_file_modular,instances_json)


#%% Generate a list of corrections made by geofencing, and counts

from megadetector.utils.wi_utils import find_geofence_adjustments
from megadetector.utils.ct_utils import is_list_sorted

rollup_pair_to_count = find_geofence_adjustments(ensemble_output_file_modular,
                                                 use_latin_names = False)

min_count = 50

footer_text = ''

rollup_pair_to_count = \
    {key: value for key, value in rollup_pair_to_count.items() if value >= min_count}

# rollup_pair_to_count is sorted in descending order by count
assert is_list_sorted(list(rollup_pair_to_count.values()),reverse=True)

if len(rollup_pair_to_count) > 0:
    
    footer_text = \
        '<h3>Geofence changes that occurred more than {} times</h3>\n'.format(min_count)
    footer_text += '<p>These numbers refer to the whole dataset, not just the sample used for this page.</p>\n'
    footer_text += '<div class="contentdiv">\n'
    
    print('Rollup changes with count > {}:'.format(min_count))
    for rollup_pair in rollup_pair_to_count.keys():
        count = rollup_pair_to_count[rollup_pair]
        rollup_pair_s = rollup_pair.replace(',',' --> ')
        print('{}: {}'.format(rollup_pair_s,count))
        rollup_pair_html = rollup_pair.replace(',',' &rarr; ')
        footer_text += '{} ({})<br>\n'.format(rollup_pair_html,count)

    footer_text += '</div>\n'


#%% Convert output file to MD format 

ensemble_file_to_convert = ensemble_output_file_modular   
assert os.path.isfile(ensemble_file_to_convert)
ensemble_output_file_md_format = insert_before_extension(ensemble_file_to_convert,
                                                         'md-format')

generate_md_results_from_predictions_json(predictions_json_file=ensemble_file_to_convert,
                                          md_results_file=ensemble_output_file_md_format,
                                          base_folder=input_folder+'/')

# from megadetector.utils.path_utils import open_file; open_file(ensemble_output_file_md_format)


#%% Confirm that all the right files are in the results

import json
from megadetector.utils.path_utils import find_images

with open(ensemble_output_file_md_format,'r') as f:
    d = json.load(f)

filenames_in_results = set([im['file'] for im in d['images']])
images_in_folder = set(find_images(input_folder,recursive=True,return_relative_paths=True))

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


#%% Optional RDE prep: define custom camera folder function

if False:
    
    #%% Sample custom camera folder function
    
    def custom_relative_path_to_location(relative_path):
        
        relative_path = relative_path.replace('\\','/')    
        tokens = relative_path.split('/')        
        location_name = '/'.join(tokens[0:2])
        return location_name
    
    #%% Test custom function
    
    from tqdm import tqdm
    
    with open(ensemble_output_file_md_format,'r') as f:
        d = json.load(f)
    image_filenames = [im['file'] for im in d['images']]

    location_names = set()

    # relative_path = image_filenames[0]
    for relative_path in tqdm(image_filenames):
        
        location_name = custom_relative_path_to_location(relative_path)
        # location_name = image_file_to_camera_folder(relative_path)
        
        location_names.add(location_name)
        
    location_names = list(location_names)
    location_names.sort()

    for s in location_names:
        print(s)


#%% Repeat detection elimination, phase 1

from megadetector.postprocessing.repeat_detection_elimination import repeat_detections_core
from megadetector.utils.ct_utils import image_file_to_camera_folder

rde_base = os.path.join(output_base,'rde')
options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.1
options.confidenceMax = 1.01
options.iouThreshold = 0.85
options.occurrenceThreshold = 15
options.maxSuspiciousDetectionSize = 0.2
# options.minSuspiciousDetectionSize = 0.05

options.parallelizationUsesThreads = use_threads_for_parallelization
options.nWorkers = 10

# This will cause a very light gray box to get drawn around all the detections
# we're *not* considering as suspicious.
options.bRenderOtherDetections = True
options.otherDetectionsThreshold = options.confidenceMin

options.bRenderDetectionTiles = True
options.maxOutputImageWidth = 2000
options.detectionTilesMaxCrops = 100

# options.lineThickness = 5
# options.boxExpansion = 8

options.customDirNameFunction = image_file_to_camera_folder
# options.customDirNameFunction = custom_relative_path_to_location

options.bRenderHtml = False
options.imageBase = input_folder
rde_string = 'rde_{:.3f}_{:.3f}_{}_{:.3f}'.format(
    options.confidenceMin, options.iouThreshold,
    options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
options.outputBase = os.path.join(rde_base, rde_string)
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

suspicious_detection_results = \
    repeat_detections_core.find_repeat_detections(ensemble_output_file_md_format,
                                                  outputFilename=None,
                                                  options=options)


#%% Manual RDE step

from megadetector.utils.path_utils import open_file

## DELETE THE VALID DETECTIONS ##

# If you run this line, it will open the folder up in your file browser
open_file(os.path.dirname(suspicious_detection_results.filterFile),
          attempt_to_open_in_wsl_host=True)


#%% Re-filtering

from megadetector.postprocessing.repeat_detection_elimination import \
    remove_repeat_detections

filtered_output_filename = insert_before_extension(ensemble_output_file_md_format, 
                                                              'filtered_{}'.format(rde_string))

remove_repeat_detections.remove_repeat_detections(
    inputFile=ensemble_output_file_md_format,
    outputFile=filtered_output_filename,
    filteringDir=os.path.dirname(suspicious_detection_results.filterFile)
    )


#%% Preview

from megadetector.utils.path_utils import open_file
from megadetector.postprocessing.postprocess_batch_results import \
    PostProcessingOptions, process_batch_results

assert os.path.isfile(ensemble_output_file_md_format)

try:
    preview_file = filtered_output_filename
    print('Using RDE results for preview')
except:
    preview_file = ensemble_output_file_md_format
    print('RDE results not found, using raw results for preview')

preview_folder = preview_folder_base

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_folder
options.include_almost_detections = True
options.num_images_to_sample = 10000
options.confidence_threshold = 0.2
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.sample_seed = 0
options.max_figures_per_html_file = 2500
options.sort_classification_results_by_count = True
options.footer_text = footer_text

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = 10
options.parallelize_rendering_with_threads = use_threads_for_parallelization

if render_animals_only:
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                     'detections_person_vehicle','non_detections']

preview_output_base = os.path.join(preview_folder,
    job_name + '_{:.3f}'.format(options.confidence_threshold))
if render_animals_only:
    preview_output_base = preview_output_base + '_animals_only'

os.makedirs(preview_output_base, exist_ok=True)
print('Processing to {}'.format(preview_output_base))

options.md_results_file = preview_file
options.output_dir = preview_output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# import clipboard; clipboard.copy(html_output_file)


#%% Zip results files

from megadetector.utils.path_utils import parallel_zip_files

json_files = os.listdir(output_base)
json_files = [fn for fn in json_files if fn.endswith('.json')]
json_files = [os.path.join(output_base,fn) for fn in json_files]

parallel_zip_files(json_files,verbose=True)

