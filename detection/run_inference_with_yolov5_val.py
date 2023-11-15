########
# 
# run_inference_with_yolov5_val.py
#
# Runs a folder of images through MegaDetector (or another YOLOv5 model) with YOLOv5's 
# val.py, converting the output to the standard MD format.  The main goal is to leverage
# YOLO's test-time augmentation tools.
#
# YOLOv5's val.py uses each file's base name as a unique identifier, which doesn't work 
# when you have typical camera trap images like:
#
# a/b/c/RECONYX0001.JPG
# d/e/f/RECONYX0001.JPG
#
# ...so this script jumps through a bunch of hoops to put a symlinks in a flat
# folder, run YOLOv5 on that folder, and map the results back to the real files.
#
# Currently requires the user to supply the path where a working YOLOv5 install lives,
# and assumes that the current conda environment is all set up for YOLOv5.
#
# By default, this script uses symlinks to format the input images in a way that YOLOv5's 
# val.py likes.  This requires admin privileges on Windows... actually technically this only 
# requires permissions to create symbolic links, but I've never seen a case where someone has
# that permission and *doesn't* have admin privileges.  If you are running this script on
# Windows and you don't have admin privileges, use --no_use_symlinks.
#
# TODO:
#
# * Multiple GPU support
#
# * Checkpointing
#
# * Support alternative class names at the command line (currently defaults to MD classes,
#   though other class names can be supplied programmatically)
#
########

#%% Imports

import os
import sys
import uuid
import glob
import tempfile
import shutil
import json

from tqdm import tqdm

from md_utils import path_utils
from md_utils import process_utils
from data_management import yolo_output_to_md_output
from detection.run_detector import try_download_known_detector

default_image_size_with_augmentation = int(1280 * 1.3)
default_image_size_with_no_augmentation = 1280


#%% Options class

class YoloInferenceOptions:
        
    ## Required ##
    
    input_folder = None
    model_filename = None
    output_file = None
    
    ## Optional ##
    
    # Required for YOLOv5 models, not for YOLOv8 models
    yolo_working_folder = None
    
    model_type = 'yolov5' # currently 'yolov5' and 'yolov8' are supported

    image_size = default_image_size_with_augmentation
    conf_thres = '0.001'
    batch_size = 1
    device_string = '0'
    augment = True

    symlink_folder = None
    use_symlinks = True
    
    yolo_results_folder = None
    
    remove_symlink_folder = True
    remove_yolo_results_folder = True
    
    # These are deliberately offset from the standard MD categories; YOLOv5
    # needs categories IDs to start at 0.
    #
    # This can also be a string that points to a YOLOv5 dataset.yaml file.
    yolo_category_id_to_name = {0:'animal',1:'person',2:'vehicle'}
    
    # 'error','skip','overwrite'
    overwrite_handling = 'skip'
    
    preview_yolo_command_only = False
            
    
#%% Main function

def run_inference_with_yolo_val(options):

    ##%% Path handling
    
    if options.yolo_working_folder is None:
        assert options.model_type == 'yolov8', \
            'A working folder is required to run YOLOv5 val.py'
    else:
        assert os.path.isdir(options.yolo_working_folder), \
            'Could not find working folder {}'.format(options.yolo_working_folder)
                
    assert os.path.isdir(options.input_folder) or os.path.isfile(options.input_folder), \
        'Could not find input {}'.format(options.input_folder)
    
    # If the model filename is a known model string (e.g. "MDv5A", download the model if necessary)
    model_filename = try_download_known_detector(options.model_filename)
    
    assert os.path.isfile(model_filename), \
        'Could not find model file {}'.format(model_filename)
    
    if os.path.exists(options.output_file):
        if options.overwrite_handling == 'skip':
            print('Warning: output file {} exists, skipping'.format(options.output_file))
            return
        elif options.overwrite_handling == 'overwrite':
            print('Warning: output file {} exists, overwriting'.format(options.output_file))
        elif options.overwrite_handling == 'error':
            raise ValueError('Output file {} exists'.format(options.output_file))
        else:
            raise ValueError('Unknown output handling method {}'.format(options.overwrite_handling))
            
    os.makedirs(os.path.dirname(options.output_file),exist_ok=True)
    
    
    ##%% Other input handling
    
    if isinstance(options.yolo_category_id_to_name,str):
        assert os.path.isfile(options.yolo_category_id_to_name)
        yolo_dataset_file = options.yolo_category_id_to_name
        options.yolo_category_id_to_name = \
            yolo_output_to_md_output.read_classes_from_yolo_dataset_file(yolo_dataset_file)
        print('Loaded {} category mappings from {}'.format(
            len(options.yolo_category_id_to_name),yolo_dataset_file))

    temporary_folder = None
    symlink_folder_is_temp_folder = False
    yolo_folder_is_temp_folder = False
    
    job_id = str(uuid.uuid1())
    
    def get_job_temporary_folder(tf):
        if tf is not None:
            return tf
        tempdir_base = tempfile.gettempdir()
        tf = os.path.join(tempdir_base,'md_to_yolo','md_to_yolo_' + job_id)
        os.makedirs(tf,exist_ok=True)
        return tf
        
    symlink_folder = options.symlink_folder
    yolo_results_folder = options.yolo_results_folder
    
    if symlink_folder is None:
        temporary_folder = get_job_temporary_folder(temporary_folder)
        symlink_folder = os.path.join(temporary_folder,'symlinks')
        symlink_folder_is_temp_folder = True
    
    if yolo_results_folder is None:
        temporary_folder = get_job_temporary_folder(temporary_folder)
        yolo_results_folder = os.path.join(temporary_folder,'yolo_results')
        yolo_folder_is_temp_folder = True
        
    # Attach a GUID to the symlink folder, regardless of whether we created it
    symlink_folder_inner = os.path.join(symlink_folder,job_id)
    
    os.makedirs(symlink_folder_inner,exist_ok=True)
    os.makedirs(yolo_results_folder,exist_ok=True)
    

    ##%% Enumerate images
    
    if os.path.isdir(options.input_folder):
        image_files_absolute = path_utils.find_images(options.input_folder,recursive=True)
    else:
        assert os.path.isfile(options.input_folder)
        with open(options.input_folder,'r') as f:            
            image_files_absolute = json.load(f)
            assert isinstance(image_files_absolute,list)
            for fn in image_files_absolute:
                assert os.path.isfile(fn), 'Could not find image file {}'.format(fn)
    
    
    ##%% Create symlinks to give a unique ID to each image
    
    image_id_to_file = {}  
    image_id_to_error = {}
    
    if options.use_symlinks:
        print('Creating {} symlinks in {}'.format(len(image_files_absolute),symlink_folder_inner))
    else:
        print('Symlinks disabled, copying {} images to {}'.format(len(image_files_absolute),symlink_folder_inner))
        
    # i_image = 0; image_fn = image_files_absolute[i_image]
    for i_image,image_fn in tqdm(enumerate(image_files_absolute),total=len(image_files_absolute)):
        
        ext = os.path.splitext(image_fn)[1]
        
        image_id = str(i_image).zfill(10)
        image_id_to_file[image_id] = image_fn
        symlink_name = image_id + ext
        symlink_full_path = os.path.join(symlink_folder_inner,symlink_name)
        
        try:
            if options.use_symlinks:
                path_utils.safe_create_link(image_fn,symlink_full_path)
            else:
                shutil.copyfile(image_fn,symlink_full_path)
        except Exception as e:
            image_id_to_error[image_id] = str(e)
            print('Warning: error copying/creating link for input file {}: {}'.format(
                image_fn,str(e)))
            continue
        
    # ...for each image


    ##%% Create the dataset file if necessary
    
    # This may have been passed in as a string, but at this point, we should have
    # loaded the dataset file.
    assert isinstance(options.yolo_category_id_to_name,dict)
    
    # Category IDs need to be continuous integers starting at 0
    category_ids = sorted(list(options.yolo_category_id_to_name.keys()))
    assert category_ids[0] == 0
    assert len(category_ids) == 1 + category_ids[-1]
    
    yolo_dataset_file = os.path.join(yolo_results_folder,'dataset.yaml')
    
    with open(yolo_dataset_file,'w') as f:
        f.write('path: {}\n'.format(symlink_folder_inner))
        f.write('train: .\n')
        f.write('val: .\n')
        f.write('test: .\n')
        f.write('\n')
        f.write('nc: {}\n'.format(len(options.yolo_category_id_to_name)))
        f.write('\n')
        f.write('names:\n')
        for category_id in category_ids:
            assert isinstance(category_id,int)
            f.write('  {}: {}\n'.format(category_id,
                                        options.yolo_category_id_to_name[category_id]))


    ##%% Prepare Python command or YOLO CLI command
    
    image_size_string = str(round(options.image_size))
    
    if options.model_type == 'yolov5':
        
        cmd = 'python val.py --task test --data "{}"'.format(yolo_dataset_file)
        cmd += ' --weights "{}"'.format(model_filename)
        cmd += ' --batch-size {} --imgsz {} --conf-thres {}'.format(
            options.batch_size,image_size_string,options.conf_thres)
        cmd += ' --device "{}" --save-json'.format(options.device_string)
        cmd += ' --project "{}" --name "{}" --exist-ok'.format(yolo_results_folder,'yolo_results')
        
        if options.augment:
            cmd += ' --augment'
                
    elif options.model_type == 'yolov8':
        
        if options.augment:
            augment_string = 'augment'
        else:
            augment_string = ''
            
        cmd = 'yolo val {} model="{}" imgsz={} batch={} data="{}" project="{}" name="{}"'.format(
            augment_string,model_filename,image_size_string,options.batch_size,yolo_dataset_file,
            yolo_results_folder,'yolo_results')        
        cmd += ' save_hybrid save_json'        
            
    else:
        
        raise ValueError('Unrecognized model type {}'.format(options.model_type))
        
    # print(cmd); import clipboard; clipboard.copy(cmd)

    
    ##%% Run YOLO command
    
    if options.yolo_working_folder is not None:
        current_dir = os.getcwd()
        os.chdir(options.yolo_working_folder)    
    print('Running YOLO inference command:\n{}\n'.format(cmd))
    
    if options.preview_yolo_command_only:
        if options.remove_symlink_folder:
            try:
                shutil.rmtree(symlink_folder)
            except Exception:
                print('Warning: error removing symlink folder {}'.format(symlink_folder))
                pass
        if options.remove_yolo_results_folder:
            try:
                shutil.rmtree(yolo_results_folder)
            except Exception:
                print('Warning: error removing YOLO results folder {}'.format(yolo_results_folder))
                pass
        
        sys.exit()
        
    execution_result = process_utils.execute_and_print(cmd)
    assert execution_result['status'] == 0, 'Error running {}'.format(options.model_type)
    yolo_console_output = execution_result['output']
    
    yolo_read_failures = []
    for line in yolo_console_output:
        if 'cannot identify image file' in line:
            tokens = line.split('cannot identify image file')
            image_name = tokens[-1].strip()
            assert image_name[0] == "'" and image_name [-1] == "'"
            image_name = image_name[1:-1]
            yolo_read_failures.append(image_name)            
            
    # image_file = yolo_read_failures[0]
    for image_file in yolo_read_failures:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        assert image_id in image_id_to_file
        if image_id not in image_id_to_error:
            image_id_to_error[image_id] = 'YOLO read failure'
    
    if options.yolo_working_folder is not None:
        os.chdir(current_dir)
        
    
    ##%% Convert results to MD format
    
    json_files = glob.glob(yolo_results_folder+ '/yolo_results/*.json')
    assert len(json_files) == 1    
    yolo_json_file = json_files[0]

    image_id_to_relative_path = {}
    for image_id in image_id_to_file:
        fn = image_id_to_file[image_id]
        if os.path.isdir(options.input_folder):
            assert options.input_folder in fn
            relative_path = os.path.relpath(fn,options.input_folder)
        else:
            assert os.path.isfile(options.input_folder)
            # We'll use the absolute path as a relative path, and pass '/'
            # as the base path in this case.
            relative_path = fn
        image_id_to_relative_path[image_id] = relative_path
        
    if os.path.isdir(options.input_folder):
        image_base = options.input_folder
    else:
        assert os.path.isfile(options.input_folder)
        image_base = '/'
        
    yolo_output_to_md_output.yolo_json_output_to_md_output(
        yolo_json_file=yolo_json_file,
        image_folder=image_base,
        output_file=options.output_file,
        yolo_category_id_to_name=options.yolo_category_id_to_name,
        detector_name=os.path.basename(model_filename),
        image_id_to_relative_path=image_id_to_relative_path,
        image_id_to_error=image_id_to_error)


    ##%% Clean up
    
    if options.remove_symlink_folder:
        shutil.rmtree(symlink_folder)
    elif symlink_folder_is_temp_folder:
        print('Warning: using temporary symlink folder {}, but not removing it'.format(
            symlink_folder))
        
    if options.remove_yolo_results_folder:
        shutil.rmtree(yolo_results_folder)
    elif yolo_folder_is_temp_folder:
        print('Warning: using temporary YOLO results folder {}, but not removing it'.format(
            yolo_results_folder))
            
# ...def run_inference_with_yolo_val()


#%% Command-line driver

import argparse,sys
from md_utils.ct_utils import args_to_object

def main():
    
    options = YoloInferenceOptions()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_filename',type=str,
        help='model file name')
    parser.add_argument(
        'input_folder',type=str,
        help='folder on which to recursively run the model, or a .json list of filenames')
    parser.add_argument(
        'output_file',type=str,
        help='.json file where output will be written')
    
    parser.add_argument(
        '--yolo_working_folder',type=str,default=None,
        help='folder in which to execute val.py (not necessary for YOLOv8 inference)')
    parser.add_argument(
        '--image_size', default=None, type=int,
        help='image size for model execution (default {} when augmentation is enabled, else {})'.format(
            default_image_size_with_augmentation,default_image_size_with_no_augmentation))
    parser.add_argument(
        '--conf_thres', default=options.conf_thres, type=float,
        help='confidence threshold for including detections in the output file (default {})'.format(
            options.conf_thres))
    parser.add_argument(
        '--batch_size', default=options.batch_size, type=int,
        help='inference batch size (default {})'.format(options.batch_size))
    parser.add_argument(
        '--device_string', default=options.device_string, type=str,
        help='CUDA device specifier, e.g. "0" or "cpu" (default {})'.format(options.device_string))
    parser.add_argument(
        '--overwrite_handling', default=options.overwrite_handling, type=str,
        help='action to take if the output file exists (skip, error, overwrite) (default {})'.format(
            options.overwrite_handling))
    parser.add_argument(
        '--yolo_dataset_file', default=None, type=str,
        help='YOLOv5 dataset.yml file from which we should load category information ' + \
            '(otherwise defaults to MD categories)')
    parser.add_argument(
        '--model_type', default=options.model_type, type=str,
        help='Model type (yolov5 or yolov8) (default {})'.format(options.model_type))

    parser.add_argument(
        '--symlink_folder', type=str,
        help='temporary folder for symlinks (defaults to a folder in the system temp dir)')
    parser.add_argument(
        '--yolo_results_folder', type=str,
        help='temporary folder for YOLO intermediate output (defaults to a folder in the system temp dir)')
    parser.add_argument(
        '--no_use_symlinks', action='store_true',
        help='copy files instead of creating symlinks when preparing the yolo input folder')
    parser.add_argument(
        '--no_remove_symlink_folder', action='store_true',
        help='don\'t remove the temporary folder full of symlinks')
    parser.add_argument(
        '--no_remove_yolo_results_folder', action='store_true',
        help='don\'t remove the temporary folder full of YOLO intermediate files')
    
    parser.add_argument(
        '--preview_yolo_command_only', action='store_true',
        help='don\'t run inference, just preview the YOLO inference command (still creates symlinks)')
    
    if options.augment:
        default_augment_enabled = 1
    else:
        default_augment_enabled = 0
            
    parser.add_argument(
        '--augment_enabled', default=default_augment_enabled, type=int,
        help='enable/disable augmentation (default {})'.format(default_augment_enabled))
        
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
        
    # If the caller hasn't specified an image size, choose one based on whether augmentation
    # is enabled.
    if args.image_size is None:
        assert options.augment in (0,1)
        if options.augment == 1:
            args.image_size = default_image_size_with_augmentation
        else:
            args.image_size = default_image_size_with_no_augmentation
        augment_enabled_string = 'enabled'
        if not options.augment:
            augment_enabled_string = 'disabled'
        print('Augmentation is {}, using default image size {}'.format(
            augment_enabled_string,args.image_size))
        
    args_to_object(args, options)
    
    if args.yolo_dataset_file is not None:
        options.yolo_category_id_to_name = args.yolo_dataset_file
        
    options.remove_symlink_folder = (not options.no_remove_symlink_folder)
    options.remove_yolo_results_folder = (not options.no_remove_yolo_results_folder)
    options.use_symlinks = (not options.no_use_symlinks)
    options.augment = (options.augment_enabled > 0)        
            
    print(options.__dict__)
    
    run_inference_with_yolo_val(options)
    

if __name__ == '__main__':
    main()


#%% Scrap

if False:

    #%% Test driver (folder)
    
    project_name = 'KRU-test-corrupted'
    input_folder = os.path.expanduser(f'~/data/{project_name}')
    output_folder = os.path.expanduser(f'~/tmp/{project_name}')
    model_filename = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    yolo_working_folder = os.path.expanduser('~/git/yolov5')
    model_name = os.path.splitext(os.path.basename(model_filename))[0]
    
    symlink_folder = os.path.join(output_folder,'symlinks')
    yolo_results_folder = os.path.join(output_folder,'yolo_results')
    
    output_file = os.path.join(output_folder,'{}_{}-md_format.json'.format(
        project_name,model_name))
    
    options = YoloInferenceOptions()
    
    options.yolo_working_folder = yolo_working_folder
    
    options.output_file = output_file
    
    options.augment = False
    options.conf_thres = '0.001'
    options.batch_size = 1
    options.device_string = '0'

    if options.augment:
        options.image_size = round(1280 * 1.3)
    else:
        options.image_size = 1280
    
    options.input_folder = input_folder
    options.model_filename = model_filename
    
    options.yolo_results_folder = yolo_results_folder # os.path.join(output_folder + 'yolo_results')        
    options.symlink_folder = symlink_folder # os.path.join(output_folder,'symlinks')
    options.use_symlinks = False
    
    options.remove_temporary_symlink_folder = False
    options.remove_yolo_results_file = False
    
    cmd = f'python run_inference_with_yolov5_val.py {model_filename} {input_folder} ' + \
          f'{output_file} --yolo_working_folder {yolo_working_folder} ' + \
          f' --image_size {options.image_size} --conf_thres {options.conf_thres} ' + \
          f' --batch_size {options.batch_size} ' + \
          f' --symlink_folder {options.symlink_folder} --yolo_results_folder {options.yolo_results_folder} ' + \
          ' --no_remove_symlink_folder --no_remove_yolo_results_folder'
      
    if not options.use_symlinks:
        cmd += ' --no_use_symlinks'
    if not options.augment:
        cmd += ' --augment_enabled 0'
        
    print(cmd)
    execute_in_python = False
    if execute_in_python:
        run_inference_with_yolo_val(options)
    else:
        import clipboard; clipboard.copy(cmd)
    
    
    #%% Test driver (folder) (YOLOv8 model)
    
    project_name = 'yolov8-inference-test'
    input_folder = os.path.expanduser('~/data/usgs-kissel-training-resized/val')
    dataset_file = os.path.expanduser('~/data/usgs-kissel-training-yolo/dataset.yaml')
    output_folder = os.path.expanduser(f'~/tmp/{project_name}')
    model_filename = os.path.expanduser(
        '~/models/usgs-tegus/usgs-tegus-yolov8x-2023.10.25-b-1-img640-e200-best.pt')
    model_name = os.path.splitext(os.path.basename(model_filename))[0]
    
    assert os.path.isdir(input_folder)
    assert os.path.isfile(dataset_file)
    assert os.path.isfile(model_filename)    
    
    symlink_folder = os.path.join(output_folder,'symlinks')
    yolo_results_folder = os.path.join(output_folder,'yolo_results')
    
    output_file = os.path.join(output_folder,'{}_{}-md_format.json'.format(
        project_name,model_name))
    
    options = YoloInferenceOptions()
    
    options.model_type = 'yolov8'    
    options.yolo_category_id_to_name = dataset_file    
    options.yolo_working_folder = None    
    options.output_file = output_file
    
    options.augment = False
    options.conf_thres = '0.001'
    options.batch_size = 1
    options.device_string = '0'

    if options.augment:
        options.image_size = round(640 * 1.3)
    else:
        options.image_size = 640
    
    options.input_folder = input_folder
    options.model_filename = model_filename
    
    options.yolo_results_folder = yolo_results_folder 
    options.symlink_folder = symlink_folder
    options.use_symlinks = False
    
    options.remove_temporary_symlink_folder = False
    options.remove_yolo_results_file = False
    
    cmd = f'python run_inference_with_yolov5_val.py {model_filename} ' + \
          f'{input_folder} {output_file}' + \
          f' --image_size {options.image_size} --conf_thres {options.conf_thres} ' + \
          f' --batch_size {options.batch_size} --symlink_folder {options.symlink_folder} ' + \
          f'--yolo_results_folder {options.yolo_results_folder} --model_type {options.model_type}' + \
          f' --yolo_dataset_file {options.yolo_category_id_to_name}' + \
          ' --no_remove_symlink_folder --no_remove_yolo_results_folder'
      
    if not options.use_symlinks:
        cmd += ' --no_use_symlinks'
    if not options.augment:
        cmd += ' --augment_enabled 0'
        
    print(cmd)
    execute_in_python = False
    if execute_in_python:
        run_inference_with_yolo_val(options)
    else:
        import clipboard; clipboard.copy(cmd)
    
    
    #%% Preview results
    
    postprocessing_output_folder = os.path.join(output_folder,'yolo-val-preview')
    md_json_file = options.output_file
    
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    with open(md_json_file,'r') as f:
        d = json.load(f)
    
    base_task_name = os.path.basename(md_json_file)
    
    pp_options = PostProcessingOptions()
    pp_options.image_base_dir = input_folder
    pp_options.include_almost_detections = True
    pp_options.num_images_to_sample = None
    pp_options.confidence_threshold = 0.1
    pp_options.almost_detection_confidence_threshold = pp_options.confidence_threshold - 0.025
    pp_options.ground_truth_json_file = None
    pp_options.separate_detections_by_category = True
    # pp_options.sample_seed = 0
    
    pp_options.parallelize_rendering = True
    pp_options.parallelize_rendering_n_cores = 16
    pp_options.parallelize_rendering_with_threads = False
    
    output_base = os.path.join(postprocessing_output_folder,
        base_task_name + '_{:.3f}'.format(pp_options.confidence_threshold))
    
    os.makedirs(output_base, exist_ok=True)
    print('Processing to {}'.format(output_base))
    
    pp_options.api_output_file = md_json_file
    pp_options.output_dir = output_base
    ppresults = process_batch_results(pp_options)
    html_output_file = ppresults.output_html_file
    
    path_utils.open_file(html_output_file)
    
    # ...for each prediction file
    
    
    #%% Compare results
    
    import itertools
    
    from api.batch_processing.postprocessing.compare_batch_results import (
        BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)
    
    options = BatchComparisonOptions()
    
    organization_name = ''
    project_name = ''
    
    options.job_name = f'{organization_name}-comparison'
    options.output_folder = os.path.join(output_folder,'model_comparison')
    options.image_folder = input_folder
    
    options.pairwise_options = []
    
    filenames = [
        f'/home/user/tmp/{project_name}/{project_name}_md_v5a.0.0-md_format.json',
        f'/home/user/postprocessing/{organization_name}/{organization_name}-2023-04-06-v5a.0.0/combined_api_outputs/{organization_name}-2023-04-06-v5a.0.0_detections.json',
        f'/home/user/postprocessing/{organization_name}/{organization_name}-2023-04-06-v5b.0.0/combined_api_outputs/{organization_name}-2023-04-06-v5b.0.0_detections.json'
        ]
    
    descriptions = ['YOLO w/augment','MDv5a','MDv5b']
    
    if False:
        results = []
        
        for fn in filenames:
            with open(fn,'r') as f:
                d = json.load(f)
            results.append(d)
        
    detection_thresholds = [0.1,0.1,0.1]
    
    assert len(detection_thresholds) == len(filenames)
    
    rendering_thresholds = [(x*0.6666) for x in detection_thresholds]
    
    # Choose all pairwise combinations of the files in [filenames]
    for i, j in itertools.combinations(list(range(0,len(filenames))),2):
            
        pairwise_options = PairwiseBatchComparisonOptions()
        
        pairwise_options.results_filename_a = filenames[i]
        pairwise_options.results_filename_b = filenames[j]
        
        pairwise_options.results_description_a = descriptions[i]
        pairwise_options.results_description_b = descriptions[j]
        
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
