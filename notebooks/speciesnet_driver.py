#%% Imports

import os

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.wi_utils import generate_md_results_from_predictions_json
from megadetector.utils.wi_utils import generate_instances_json_from_folder

import clipboard # noqa


#%% Constants I set for each job

organization_name ='organization_name'
job_name = 'job_name'
output_base = os.path.join(os.path.expanduser('~/postprocessing'),organization_name,job_name)
os.makedirs(output_base,exist_ok=True)

preview_folder_base = os.path.join(output_base,'preview')

input_folder = '/stuff/input_folder'
assert not input_folder.endswith('/')
model_file = os.path.expanduser('~/models/speciesnet/crop')
country_code = None
lat = None
lon = None
instances_json = None # os.path.join(output_base,'instances.json')

valid_modes = ('all_in_one','modular','md')
mode = 'all_in_one'
assert mode in valid_modes

speciesnet_folder = os.path.expanduser('~/git/cameratrapai')
speciesnet_pt_environment_name = 'speciesnet-package-pytorch'
speciesnet_tf_environment_name = 'speciesnet-package-tf'

md_environment_name = 'cameratraps-detector'
md_folder = os.path.expanduser('~/git/MegaDetector/megadetector')
md_python_path = '{}:{}'.format(
    os.path.expanduser('~/git/yolov5-md'),
    os.path.expanduser('~/git/MegaDetector'))

gpu_number = 0

if gpu_number is not None:
    cuda_prefix = 'export CUDA_VISIBLE_DEVICES={} && '.format(gpu_number)
    
classifier_batch_size = 64


#%% Generate instances.json

if instances_json is not None:

    _ = generate_instances_json_from_folder(folder=input_folder,
                                            country=country_code,
                                            lat=lat,
                                            lon=lon,
                                            output_file=instances_json,
                                            filename_replacements=None)


#%% Run everything using SpeciesNet (all-in-one)

if mode == 'all_in_one':
    
    ensemble_commands = []
    ensemble_commands.append(f'cd {speciesnet_folder} && mamba activate {speciesnet_pt_environment_name}')
    
    if instances_json is not None:
        source_specifier = '--instances_json "{}"'.format(instances_json)
    else:
        source_specifier = '--folders "{}"'.format(input_folder)
        
    ensemble_output_file_all_in_one = os.path.join(output_base,job_name + '-ensemble_output_all_in_one.json')
        
    cmd = '{} python scripts/run_model.py --model "{}"'.format(cuda_prefix,model_file)
    cmd += ' ' + source_specifier
    cmd += ' --predictions_json "{}"'.format(ensemble_output_file_all_in_one)
    ensemble_commands.append(cmd)
    
    ensemble_cmd = '\n\n'.join(ensemble_commands)
    print(ensemble_cmd)
    # clipboard.copy(ensemble_cmd)
    

#%% Run everything using SpeciesNet (modular)

if mode == 'modular':

    detector_output_file_modular = \
        os.path.join(output_base,job_name + '-detector_output_modular.json')
    classifier_output_file_modular = \
        os.path.join(output_base,job_name + '-classifier_output_modular.json')
    ensemble_output_file_modular = \
        os.path.join(output_base,job_name + '-ensemble_output_modular.json')
    
    for fn in [detector_output_file_modular,classifier_output_file_modular,ensemble_output_file_modular]:
        if os.path.exists(fn):
            print('** Warning, file {} exists, this is OK if you are resuming **\n'.format(fn))
    
    if instances_json is not None:
        source_specifier = '--instances_json "{}"'.format(instances_json)
    else:
        source_specifier = '--folders "{}"'.format(input_folder)
    
    ## Run detector
    
    detector_commands = []
    detector_commands.append(f'{cuda_prefix} cd {speciesnet_folder} && mamba activate {speciesnet_pt_environment_name}')
    
    cmd = 'python scripts/run_model.py --detector_only --model "{}"'.format(model_file)
    cmd += ' ' + source_specifier
    cmd += ' --predictions_json "{}"'.format(detector_output_file_modular)
    detector_commands.append(cmd)
    
    detector_cmd = '\n\n'.join(detector_commands)
    # print(detector_cmd); clipboard.copy(detector_cmd)
    
    ## Run classifier
    
    classifier_commands = []
    classifier_commands.append(f'{cuda_prefix} cd {speciesnet_folder} && mamba activate {speciesnet_tf_environment_name}')
    
    cmd = 'python scripts/run_model.py --classifier_only --model "{}"'.format(model_file)
    cmd += ' ' + source_specifier
    cmd += ' --predictions_json "{}"'.format(classifier_output_file_modular)
    cmd += ' --detections_json "{}"'.format(detector_output_file_modular)
    if classifier_batch_size is not None:
        cmd += ' --batch_size {}'.format(classifier_batch_size)
    classifier_commands.append(cmd)
    
    classifier_cmd = '\n\n'.join(classifier_commands)
    # print(classifier_cmd); clipboard.copy(classifier_cmd)
    
    ## Run ensemble
    
    # It doesn't matter here which environment we use
    ensemble_commands = []
    ensemble_commands.append(f'{cuda_prefix} cd {speciesnet_folder} && mamba activate {speciesnet_pt_environment_name}')
    
    cmd = 'python scripts/run_model.py --ensemble_only --model "{}"'.format(model_file)
    cmd += ' ' + source_specifier
    cmd += ' --predictions_json "{}"'.format(ensemble_output_file_modular)
    cmd += ' --detections_json "{}"'.format(detector_output_file_modular)
    cmd += ' --classifications_json "{}"'.format(classifier_output_file_modular)
    ensemble_commands.append(cmd)
    
    ensemble_cmd = '\n\n'.join(ensemble_commands)
    # print(ensemble_cmd); clipboard.copy(ensemble_cmd)
    
    modular_command = '\n\n'.join([detector_cmd,classifier_cmd,ensemble_cmd])
    print(modular_command)
    # clipboard.copy(modular_command)


#%% Run everything using MD + SpeciesNet

if mode == 'md':
    
    detector_output_file_md = os.path.join(output_base,job_name + '-detector_output_md.json')
    detector_output_file_predictions_format_md = insert_before_extension(detector_output_file_md,'predictons-format')
    classifier_output_file_md = os.path.join(output_base,job_name + '-classifier_output_md.json')
    ensemble_output_file_md = os.path.join(output_base,job_name + '-ensemble_output_md.json')
    
    if instances_json is not None:
        source_specifier = '--instances_json "{}"'.format(instances_json)
    else:
        source_specifier = '--folders "{}"'.format(input_folder)
    
    ## Run MegaDetector
    
    megadetector_commands = []
    megadetector_commands.append(f'export PYTHONPATH={md_python_path}')
    megadetector_commands.append(f'cd {md_folder}')
    megadetector_commands.append(f'mamba activate {md_environment_name}')
    cmd = '{} python detection/run_detector_batch.py MDV5A "{}" "{}" --quiet --recursive'.format(
        cuda_prefix, input_folder, detector_output_file_md)
    # Use absolute paths
    # cmd += ' --output_relative_filenames'
    megadetector_commands.append(cmd)
    
    megadetector_cmd = '\n\n'.join(megadetector_commands)
    # print(megadetector_cmd); clipboard.copy(megadetector_cmd)
    
    ## Convert to predictions format
    
    conversion_commands = ['']
    conversion_commands.append(f'cd {md_folder}')
    conversion_commands.append(f'mamba activate {md_environment_name}')
    
    cmd = 'python postprocessing/md_to_wi.py "{}" "{}"'.format(
            detector_output_file_md,detector_output_file_predictions_format_md)
    conversion_commands.append(cmd)
    
    conversion_cmd = '\n\n'.join(conversion_commands)
    # print(conversion_cmd); clipboard.copy(conversion_cmd)
    
    ## Run classifier
    
    classifier_commands = ['']
    classifier_commands.append(f'cd {speciesnet_folder} && mamba activate {speciesnet_tf_environment_name}')
    
    cmd = '{} python scripts/run_model.py --classifier_only --model "{}"'.format(
        cuda_prefix,model_file)
    cmd += ' ' + source_specifier
    cmd += ' --predictions_json "{}"'.format(classifier_output_file_md)
    cmd += ' --detections_json "{}"'.format(detector_output_file_predictions_format_md)
    classifier_commands.append(cmd)
    
    classifier_cmd = '\n\n'.join(classifier_commands)
    # print(classifier_cmd); clipboard.copy(classifier_cmd)
    
    ## Run ensemble
    
    # It doesn't matter here which environment we use
    ensemble_commands = ['']
    ensemble_commands.append(f'cd {speciesnet_folder} && mamba activate {speciesnet_tf_environment_name}')
    
    cmd = '{} python scripts/run_model.py --ensemble_only --model "{}"'.format(
        cuda_prefix,model_file)
    cmd += ' ' + source_specifier
    cmd += ' --predictions_json "{}"'.format(ensemble_output_file_md)
    cmd += ' --detections_json "{}"'.format(detector_output_file_predictions_format_md)
    cmd += ' --classifications_json "{}"'.format(classifier_output_file_md)
    ensemble_commands.append(cmd)
    
    ensemble_cmd = '\n\n'.join(ensemble_commands)
    # print(ensemble_cmd); clipboard.copy(ensemble_cmd)
    
    ## All in one long command
    
    modular_command = '\n\n'.join([megadetector_cmd,conversion_cmd,classifier_cmd,ensemble_cmd])
    print(modular_command)
    # clipboard.copy(modular_command)


#%% Convert output file to MD format 

ensemble_file_to_convert = None

if mode == 'all_in_one':
    ensemble_file_to_convert = ensemble_output_file_all_in_one
elif mode == 'modular':
    ensemble_file_to_convert = ensemble_output_file_modular
elif mode == 'md':
    ensemble_file_to_convert = ensemble_output_file_md
    
assert ensemble_file_to_convert is not None

assert os.path.isfile(ensemble_file_to_convert)
ensemble_output_file_md_format = insert_before_extension(ensemble_file_to_convert,
                                                         'md-format')

generate_md_results_from_predictions_json(predictions_json_file=ensemble_file_to_convert,
                                          md_results_file=ensemble_output_file_md_format,
                                          base_folder=input_folder+'/')


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


#%% Preview

from megadetector.utils.path_utils import open_file
from megadetector.postprocessing.postprocess_batch_results import \
    PostProcessingOptions, process_batch_results

assert os.path.isfile(ensemble_output_file_md_format)

preview_file = ensemble_output_file_md_format
preview_folder = os.path.join(preview_folder_base,mode)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_folder
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.confidence_threshold = 0.2
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.sample_seed = 0
options.max_figures_per_html_file = 2500
options.sort_classification_results_by_count = True

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = 10
options.parallelize_rendering_with_threads = True

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                     'detections_person_vehicle','non_detections']

output_base = os.path.join(preview_folder,
    job_name + '_{:.3f}'.format(options.confidence_threshold))
if render_animals_only:
    output_base = output_base + '_animals_only'

os.makedirs(output_base, exist_ok=True)
print('Processing to {}'.format(output_base))

options.md_results_file = preview_file
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# import clipboard; clipboard.copy(html_output_file)
