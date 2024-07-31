"""

run_inference_with_yolov5_val.py

Runs a folder of images through MegaDetector (or another YOLOv5/YOLOv8 model) with YOLO's
val.py, converting the output to the standard MD format.  The reasons this script exists,
as an alternative to the standard run_detector_batch.py are:
    
* This script provides access to YOLO's test-time augmentation tools.
* This script serves a reference implementation: by any reasonable definition, YOLOv5's
  val.py produces the "correct" result for any image, since it matches what was used in 
  training.
* This script works for any Ultralytics detection model, including YOLOv8 models  

YOLOv5's val.py uses each file's base name as a unique identifier, which doesn't work 
when you have typical camera trap images like:

* a/b/c/RECONYX0001.JPG
* d/e/f/RECONYX0001.JPG

...both of which would just be "RECONYX0001.JPG".  So this script jumps through a bunch of 
hoops to put a symlinks in a flat folder, run YOLOv5 on that folder, and map the results back 
to the real files.

If you are running a YOLOv5 model, this script currently requires the caller to supply the path
where a working YOLOv5 install lives, and assumes that the current conda environment is all set up for 
YOLOv5.  If you are running a YOLOv8 model, the folder doesn't matter, but it assumes that ultralytics
tools are available in the current environment.

By default, this script uses symlinks to format the input images in a way that YOLO's 
val.py likes, as per above.  This requires admin privileges on Windows... actually technically this 
only requires permissions to create symbolic links, but I've never seen a case where someone has
that permission and *doesn't* have admin privileges.  If you are running this script on
Windows and you don't have admin privileges, use --no_use_symlinks, which will make copies of images,
rather than using symlinks.

"""

#%% Imports

import os
import sys
import uuid
import glob
import tempfile
import shutil
import json
import copy

from tqdm import tqdm

from megadetector.utils import path_utils
from megadetector.utils import process_utils
from megadetector.utils import string_utils

from megadetector.utils.ct_utils import is_iterable, split_list_into_fixed_size_chunks
from megadetector.utils.path_utils import path_is_abs
from megadetector.data_management import yolo_output_to_md_output
from megadetector.detection.run_detector import try_download_known_detector
from megadetector.postprocessing.combine_api_outputs import combine_api_output_files

default_image_size_with_augmentation = int(1280 * 1.3)
default_image_size_with_no_augmentation = 1280


#%% Options class

class YoloInferenceOptions:
    """
    Parameters that control the behavior of run_inference_with_yolov5_val(), including 
    the input/output filenames.
    """
    
    def __init__(self):
        
        ## Required-ish ##
        
        #: Folder of images to process (can be None if image_filename_list contains absolute paths)
        self.input_folder = None
        
        #: If this is None, [input_folder] can't be None, we'll process all images in [input_folder].
        #:            
        #: If this is not None, and [input_folder] is not None, this should be a list of relative image 
        #: paths within [input_folder] to process, or a .txt or .json file containing a list of 
        #: relative image paths.
        #:
        #: If this is not None, and [input_folder] is None, this should be a list of absolute image
        #: paths, or a .txt or .json file containing a list of absolute image paths.            
        self.image_filename_list = None
        
        #: Model filename (ending in .pt), or a well-known model name (e.g. "MDV5A")
        self.model_filename = None
        
        #: .json output file, in MD results format
        self.output_file = None
        
        
        ## Optional ##
        
        #: Required for older YOLOv5 inference, not for newer ulytralytics/YOLOv8 inference
        self.yolo_working_folder = None
        
        #: Currently 'yolov5' and 'ultralytics' are supported, and really these are proxies for
        #: "the yolov5 repo" and "the ultralytics repo".
        self.model_type = 'yolov5' 
    
        #: Image size to use; this is a single int, which in ultralytics's terminology means
        #: "scale the long side of the image to this size, and preserve aspect ratio".
        #:
        #: If None, will choose based on whether augmentation is enabled.
        self.image_size = None
        
        #: Detections below this threshold will not be included in the output file
        self.conf_thres = '0.001'
        
        #: Batch size... has no impact on results, but may create memory issues if you set
        #: this to large values
        self.batch_size = 1
        
        #: Device string: typically '0' for GPU 0, '1' for GPU 1, etc., or 'cpu'
        self.device_string = '0'
        
        #: Should we enable test-time augmentation?
        self.augment = True
        
        #: Should we enable half-precision inference?
        self.half_precision_enabled = None
        
        #: Where should we stash the temporary symlinks (or copies) used to give unique identifiers to image 
        # files?
        #:
        #: If this is None, we'll create a folder in system temp space.
        self.symlink_folder = None
        
        #: Should we use symlinks to give unique identifiers to image files (vs. copies)?
        self.use_symlinks = True
        
        #: How should we guarantee that YOLO IDs (base filenames) are unique?  Choices are:
        #:
        #: * 'verify': assume image IDs are unique, but verify and error if they're not
        #: * 'links': create symlinks (or copies, depending on use_symlinks) to enforce uniqueness
        #: * 'auto': check whether IDs are unique, create links if necessary
        self.unique_id_strategy = 'links'
        
        #: Temporary folder to stash intermediate YOLO results.
        #:
        #: If this is None, we'll create a folder in system temp space.    
        self.yolo_results_folder = None
        
        #: Should we remove the symlink folder when we're done?
        self.remove_symlink_folder = True
        
        #: Should we remove the intermediate results folder when we're done?
        self.remove_yolo_results_folder = True
        
        #: These are deliberately offset from the standard MD categories; YOLOv5
        #: needs categories IDs to start at 0.
        #:
        #: This can also be a string that points to a YOLO dataset.yaml file.
        self.yolo_category_id_to_name = {0:'animal',1:'person',2:'vehicle'}
        
        #: What should we do if the output file already exists?
        #:
        #: Can be 'error', 'skip', or 'overwrite'.
        self.overwrite_handling = 'skip'
        
        #: If True, we'll do a dry run that lets you preview the YOLO val command, without
        #: actually running it.
        self.preview_yolo_command_only = False
        
        #: By default, if any errors occur while we're copying images or creating symlinks, it's
        #: game over.  If this is True, those errors become warnings, and we plow ahead.
        self.treat_copy_failures_as_warnings = False
        
        #: Save YOLO console output
        self.save_yolo_debug_output = False
        
        #: Whether to search for images recursively within [input_folder]
        #:
        #: Ignored if a list of files is provided.
        self.recursive = True
        
        #: Maximum number of images to run in a single chunk
        self.checkpoint_frequency = None
        
    # ...def __init__()
    
# ...YoloInferenceOptions()


#%% Support functions

def _clean_up_temporary_folders(options,
                                symlink_folder,yolo_results_folder,
                                symlink_folder_is_temp_folder,yolo_folder_is_temp_folder):
    """
    Remove temporary symlink/results folders, unless the caller requested that we leave them in place.
    """
    
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

    
#%% Main function

def run_inference_with_yolo_val(options):
    """
    Runs a folder of images through MegaDetector (or another YOLOv5/YOLOv8 model) with YOLO's
    val.py, converting the output to the standard MD format.
    
    Args: 
        options (YoloInferenceOptions): all the parameters used to control this process,
            including filenames; see YoloInferenceOptions for details            
    """
        
    ##%% Input and path handling
    
    default_options = YoloInferenceOptions()
    
    for k in options.__dict__.keys():
        if k not in default_options.__dict__:
            print('Warning: unexpected variable {} in options object'.format(k))
            
    if options.model_type == 'yolov8':
        
        print('Warning: model type "yolov8" supplied, "ultralytics" is the preferred model type string for YOLOv8 models')
        options.model_type = 'ultralytics'
        
    if (options.model_type == 'yolov5') and ('yolov8' in options.model_filename.lower()):
        print('\n\n*** Warning: model type set as "yolov5", but your model filename contains "yolov8"... did you mean to use --model_type yolov8?" ***\n\n')        
    
    if options.yolo_working_folder is None:
        assert options.model_type == 'ultralytics', \
            'A working folder is required to run YOLOv5 val.py'
    else:
        assert os.path.isdir(options.yolo_working_folder), \
            'Could not find working folder {}'.format(options.yolo_working_folder)
                
    if options.half_precision_enabled is not None:
        assert options.half_precision_enabled in (0,1), \
            'Invalid value {} for --half_precision_enabled (should be 0 or 1)'.format(
                options.half_precision_enabled)
    
    # If the model filename is a known model string (e.g. "MDv5A", download the model if necessary)
    model_filename = try_download_known_detector(options.model_filename)
    
    assert os.path.isfile(model_filename), \
        'Could not find model file {}'.format(model_filename)
    
    assert (options.input_folder is not None) or (options.image_filename_list is not None), \
        'You must specify a folder and/or a file list'
        
    if options.input_folder is not None:
        assert os.path.isdir(options.input_folder), 'Could not find input folder {}'.format(
            options.input_folder)
    
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
    
    if options.input_folder is not None:
        options.input_folder = options.input_folder.replace('\\','/')
                
        
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
    
    image_files_relative = None
    image_files_absolute = None
    
    # If the caller just provided a folder, not a list of files...
    if options.image_filename_list is None:
        
        assert options.input_folder is not None and os.path.isdir(options.input_folder), \
            'Could not find input folder {}'.format(options.input_folder)
        image_files_relative = path_utils.find_images(options.input_folder,
                                                      recursive=options.recursive,
                                                      return_relative_paths=True,
                                                      convert_slashes=True)
        image_files_absolute = [os.path.join(options.input_folder,fn) for \
                                fn in image_files_relative]
            
    else:
        
        # If the caller provided a list of image files (rather than a filename pointing 
        # to a list of image files)...
        if is_iterable(options.image_filename_list) and not isinstance(options.image_filename_list,str):
            
            image_files_relative = options.image_filename_list
            
        # If the caller provided a filename pointing to a list of image files...
        else:
            
            assert isinstance(options.image_filename_list,str), \
                'Unrecognized image filename list object type: {}'.format(options.image_filename_list)
            assert os.path.isfile(options.image_filename_list), \
                'Could not find image filename list file: {}'.format(options.image_filename_list)
            ext = os.path.splitext(options.image_filename_list)[-1].lower()
            assert ext in ('.json','.txt'), \
                'Unrecognized image filename list file extension: {}'.format(options.image_filename_list)
            if ext == '.json':
                with open(options.image_filename_list,'r') as f:
                    image_files_relative = json.load(f)
                    assert is_iterable(image_files_relative)
            else:
                assert ext == '.txt'
                with open(options.image_filename_list,'r') as f:
                    image_files_relative = f.readlines()
                    image_files_relative = [s.strip() for s in image_files_relative]
        
        # ...whether the image filename list was supplied as list vs. a filename
        
        if options.input_folder is None:
            
            image_files_absolute = image_files_relative
            
        else:
            
            # The list should be relative filenames
            for fn in image_files_relative:
                assert not path_is_abs(fn), \
                    'When providing a folder and a list, paths in the list should be relative'
                
            image_files_absolute = \
                [os.path.join(options.input_folder,fn) for fn in image_files_relative]
        
        for fn in image_files_absolute:
            assert os.path.isfile(fn), 'Could not find image file {}'.format(fn)
    
    # ...whether the caller supplied a list of filenames
    
    image_files_absolute = [fn.replace('\\','/') for fn in image_files_absolute]
    
    del image_files_relative
    
    
    ##%% Recurse if necessary to handle checkpoints
    
    if options.checkpoint_frequency is not None and options.checkpoint_frequency > 0:
        
        chunks = split_list_into_fixed_size_chunks(image_files_absolute,options.checkpoint_frequency)
        
        chunk_output_files = []
        
        # i_chunk = 0; chunk_files_abs = chunks[i_chunk]
        for i_chunk,chunk_files_abs in enumerate(chunks):
            
            print('Processing {} images from chunk {} of {}'.format(
                len(chunk_files_abs),i_chunk,len(chunks)))
    
            chunk_options = copy.deepcopy(options)
            
            # Run each chunk without checkpointing
            chunk_options.checkpoint_frequency = None
            
            if options.input_folder is not None:
                chunk_files_relative = \
                    [os.path.relpath(fn,options.input_folder) for fn in chunk_files_abs]
                chunk_options.image_filename_list = chunk_files_relative
            else:
                chunk_options.image_filename_list = chunk_files_abs
                
            chunk_options.image_filename_list = \
                [fn.replace('\\','/') for fn in chunk_options.image_filename_list]
                
            chunk_string = 'chunk_{}'.format(str(i_chunk).zfill(5))
            chunk_options.yolo_results_folder = yolo_results_folder + '_' + chunk_string
            chunk_options.symlink_folder = symlink_folder + '_' + chunk_string
            
            # Put the output file in the parent job's scratch folder
            chunk_output_file = os.path.join(yolo_results_folder,chunk_string + '_results_md_format.json')
            chunk_output_files.append(chunk_output_file)
            chunk_options.output_file = chunk_output_file
            
            if os.path.isfile(chunk_output_file):
                
                print('Chunk output file {} exists, checking completeness'.format(chunk_output_file))
                
                with open(chunk_output_file,'r') as f:
                    chunk_results = json.load(f)
                images_in_this_chunk_results_file = [im['file'] for im in chunk_results['images']]                    
                assert len(images_in_this_chunk_results_file) == len(chunk_options.image_filename_list), \
                    'Expected {} images in chunk results file {}, found {}, possibly this is left over from a previous job?'.format(
                        len(chunk_options.image_filename_list),chunk_output_file,
                        len(images_in_this_chunk_results_file))
                for fn in images_in_this_chunk_results_file:
                    assert fn in chunk_options.image_filename_list, \
                        'Unexpected image {} in chunk results file {}, possibly this is left over from a previous job?'.format(
                            fn,chunk_output_file)
                
                print('Chunk output file {} exists and is complete, skipping this chunk'.format(
                    chunk_output_file))
                
            # ...if the outptut file exists
            
            else:
                
                run_inference_with_yolo_val(chunk_options)
                
            # ...if we do/don't have to run this chunk
            
            assert os.path.isfile(chunk_options.output_file)
            
        # ...for each chunk
    
        # Merge
        _ = combine_api_output_files(input_files=chunk_output_files,
                                 output_file=options.output_file,
                                 require_uniqueness=True,
                                 verbose=True)
        
        # Validate
        with open(options.output_file,'r') as f:
            combined_results = json.load(f)
        assert len(combined_results['images']) == len(image_files_absolute), \
            'Expected {} images in merged output file, found {}'.format(
                len(image_files_absolute),len(combined_results['images']))
        
        # Clean up
        _clean_up_temporary_folders(options,
                                    symlink_folder,yolo_results_folder,
                                    symlink_folder_is_temp_folder,yolo_folder_is_temp_folder)
        
        return
    
    # ...if we need to make recursive calls for file chunks
    
    
    ##%% Create symlinks (or copy images) to give a unique ID to each image
    
    # Maps YOLO image IDs (base filename without extension as it will appear in YOLO .json output)
    # to the *original full path* for each image (not the symlink path).
    image_id_to_file = {}  
    
    # Maps YOLO image IDs (base filename without extension as it will appear in YOLO .json output)
    # to errors, including errors that happen before we run the model at all (e.g. file access errors).
    image_id_to_error = {}
    
    create_links = True
    
    if options.unique_id_strategy == 'links':
        
        create_links = True
        
    else:
        
        assert options.unique_id_strategy in ('auto','verify'), \
            'Unknown unique ID strategy {}'.format(options.unique_id_strategy)
            
        image_ids_are_unique = True
        
        for i_image,image_fn in tqdm(enumerate(image_files_absolute),total=len(image_files_absolute)):        
            
            image_id = os.path.splitext(os.path.basename(image_fn))[0]
            
            # Is this image ID unique?
            if image_id in image_id_to_file:
                if options.unique_id_strategy == 'verify':
                    raise ValueError('"verify" specified for image uniqueness, but ' +
                                     'image ID {} occurs more than once:\n\n{}\n\n{}'.format(
                                         image_id,image_fn,image_id_to_file[image_id]))
                else:
                    assert options.unique_id_strategy == 'auto'
                    image_ids_are_unique = False
                    image_id_to_file = {}
                    break
                
            image_id_to_file[image_id] = image_fn
        
        # ...for each image
        
        if image_ids_are_unique:
            
            print('"{}" specified for image uniqueness and images are unique, skipping links'.format(
                options.unique_id_strategy))
            assert len(image_id_to_file) == len(image_files_absolute)
            create_links = False
            
        else:
            
            assert options.unique_id_strategy == 'auto'
            create_links = True
            link_type = 'copies'
            if options.use_symlinks:
                link_type = 'links'
            print('"auto" specified for image uniqueness and images are not unique, defaulting to {}'.format(
                link_type))
            
    # ...which unique ID strategy?
    
    if create_links:
        
        if options.use_symlinks:
            print('Creating {} symlinks in {}'.format(len(image_files_absolute),symlink_folder_inner))
        else:
            print('Symlinks disabled, copying {} images to {}'.format(len(image_files_absolute),symlink_folder_inner))
            
        link_full_paths = []
        
        # i_image = 0; image_fn = image_files_absolute[i_image]
        for i_image,image_fn in tqdm(enumerate(image_files_absolute),total=len(image_files_absolute)):
            
            ext = os.path.splitext(image_fn)[1]
            image_fn_without_extension = os.path.splitext(image_fn)[0]
            
            # YOLO .json output identifies images by the base filename without the extension
            image_id = str(i_image).zfill(10)
            image_id_to_file[image_id] = image_fn
            symlink_name = image_id + ext
            symlink_full_path = os.path.join(symlink_folder_inner,symlink_name)
            link_full_paths.append(symlink_full_path)
            
            # If annotation files exist, link those too; only useful if we're reading the computed
            # mAP value, but it doesn't hurt.
            annotation_fn = image_fn_without_extension + '.txt'
            annotation_file_exists = False
            if os.path.isfile(annotation_fn):
                annotation_file_exists = True
                annotation_symlink_name = image_id + '.txt'
                annotation_symlink_full_path = os.path.join(symlink_folder_inner,annotation_symlink_name)                
            
            try:
                
                if options.use_symlinks:
                    path_utils.safe_create_link(image_fn,symlink_full_path)
                    if annotation_file_exists:
                        path_utils.safe_create_link(annotation_fn,annotation_symlink_full_path)
                else:
                    shutil.copyfile(image_fn,symlink_full_path)
                    if annotation_file_exists:
                        shutil.copyfile(annotation_fn,annotation_symlink_full_path)
                    
            except Exception as e:
                
                error_string = str(e)
                image_id_to_error[image_id] = error_string
                
                # Always break if the user is trying to create symlinks on Windows without
                # permission, 100% of images will always fail in this case.
                if ('a required privilege is not held by the client' in error_string.lower()) or \
                   (not options.treat_copy_failures_as_warnings):
                       
                       print('\nError copying/creating link for input file {}: {}'.format(
                           image_fn,error_string))
                       
                       raise
                       
                else:
                    
                    print('Warning: error copying/creating link for input file {}: {}'.format(
                        image_fn,error_string))
                    continue
            
            # ...except
            
        # ...for each image
                
    # ...if we need to create links/copies

    
    ##%% Create the dataset file if necessary
    
    # This may have been passed in as a string, but at this point, we should have
    # loaded the dataset file.
    assert isinstance(options.yolo_category_id_to_name,dict)
    
    # Category IDs need to be continuous integers starting at 0
    category_ids = sorted(list(options.yolo_category_id_to_name.keys()))
    assert category_ids[0] == 0
    assert len(category_ids) == 1 + category_ids[-1]
        
    yolo_dataset_file = os.path.join(yolo_results_folder,'dataset.yaml')
    yolo_image_list_file = os.path.join(yolo_results_folder,'images.txt')
    
    
    with open(yolo_image_list_file,'w') as f:
        
        if create_links:
            image_files_to_write = link_full_paths
        else:
            image_files_to_write = image_files_absolute
            
        for fn_abs in image_files_to_write:
            # At least in YOLOv5 val (need to verify for YOLOv8 val), filenames in this 
            # text file are treated as relative to the text file itself if they start with
            # "./", otherwise they're treated as absolute paths.  Since we don't want to put this
            # text file in the image folder, we'll use absolute paths.
            # fn_relative = os.path.relpath(fn_abs,options.input_folder)
            # f.write(fn_relative + '\n')
            f.write(fn_abs + '\n')
    
    if create_links:
        inference_folder = symlink_folder_inner
    else:
        # This doesn't matter, but it has to be a valid path
        inference_folder = options.yolo_results_folder
        
    with open(yolo_dataset_file,'w') as f:
        
        f.write('path: {}\n'.format(inference_folder))        
        # These need to be valid paths, even if you're not using them, and "." is always safe
        f.write('train: .\n')
        f.write('val: .\n')
        f.write('test: {}\n'.format(yolo_image_list_file))
        f.write('\n')
        f.write('nc: {}\n'.format(len(options.yolo_category_id_to_name)))
        f.write('\n')
        f.write('names:\n')
        for category_id in category_ids:
            assert isinstance(category_id,int)
            f.write('  {}: {}\n'.format(category_id,
                                        options.yolo_category_id_to_name[category_id]))


    ##%% Prepare Python command or YOLO CLI command
    
    if options.image_size is None:
        if options.augment:
            image_size = default_image_size_with_augmentation
        else:
            image_size = default_image_size_with_no_augmentation
    else:
        image_size = options.image_size
    
    image_size_string = str(round(image_size))
    
    if options.model_type == 'yolov5':
        
        cmd = 'python val.py --task test --data "{}"'.format(yolo_dataset_file)
        cmd += ' --weights "{}"'.format(model_filename)
        cmd += ' --batch-size {} --imgsz {} --conf-thres {}'.format(
            options.batch_size,image_size_string,options.conf_thres)
        cmd += ' --device "{}" --save-json'.format(options.device_string)
        cmd += ' --project "{}" --name "{}" --exist-ok'.format(yolo_results_folder,'yolo_results')
        
        # This is the NMS IoU threshold
        # cmd += ' --iou-thres 0.6'
        
        if options.augment:
            cmd += ' --augment'
                
        # --half is a store_true argument for YOLOv5's val.py
        if (options.half_precision_enabled is not None) and (options.half_precision_enabled == 1):
            cmd += ' --half'
        
        # Sometimes useful for debugging
        # cmd += ' --save_conf --save_txt'
        
    elif options.model_type == 'ultralytics':
                
        if options.augment:
            augment_string = 'augment'
        else:
            augment_string = ''
            
        cmd = 'yolo val {} model="{}" imgsz={} batch={} data="{}" project="{}" name="{}" device="{}"'.\
            format(augment_string,model_filename,image_size_string,options.batch_size,
                   yolo_dataset_file,yolo_results_folder,'yolo_results',options.device_string)
        cmd += ' save_json exist_ok'
        
        if (options.half_precision_enabled is not None):
            if options.half_precision_enabled == 1:
                cmd += ' --half=True'
            else:
                assert options.half_precision_enabled == 0
                cmd += ' --half=False'
        
        # Sometimes useful for debugging
        # cmd += ' save_conf save_txt'
            
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
                print('Removing YOLO symlink folder {}'.format(symlink_folder))
                shutil.rmtree(symlink_folder)
            except Exception:
                print('Warning: error removing symlink folder {}'.format(symlink_folder))
                pass
        if options.remove_yolo_results_folder:
            try:
                print('Removing YOLO results folder {}'.format(yolo_results_folder))
                shutil.rmtree(yolo_results_folder)
            except Exception:
                print('Warning: error removing YOLO results folder {}'.format(yolo_results_folder))
                pass
        
        # sys.exit()
        return
    
    execution_result = process_utils.execute_and_print(cmd,encoding='utf-8',verbose=True)
    assert execution_result['status'] == 0, 'Error running {}'.format(options.model_type)
    yolo_console_output = execution_result['output']
      
    if options.save_yolo_debug_output:
        
        with open(os.path.join(yolo_results_folder,'yolo_console_output.txt'),'w') as f:
            for s in yolo_console_output:
                f.write(s + '\n')
        with open(os.path.join(yolo_results_folder,'image_id_to_file.json'),'w') as f:
            json.dump(image_id_to_file,f,indent=1)
        with open(os.path.join(yolo_results_folder,'image_id_to_error.json'),'w') as f:
            json.dump(image_id_to_error,f,indent=1)
                
        
    # YOLO console output contains lots of ANSI escape codes, remove them for easier parsing
    yolo_console_output = [string_utils.remove_ansi_codes(s) for s in yolo_console_output]
    
    # Find errors that occurred during the initial corruption check; these will not be included in the
    # output.  Errors that occur during inference will be handled separately.
    yolo_read_failures = []
    
    for line in yolo_console_output:
        # Lines look like:
        #
        # For ultralytics val:
        #
        # val: WARNING ⚠️ /a/b/c/d.jpg: ignoring corrupt image/label: [Errno 13] Permission denied: '/a/b/c/d.jpg'
        # line = "val: WARNING ⚠️ /a/b/c/d.jpg: ignoring corrupt image/label: [Errno 13] Permission denied: '/a/b/c/d.jpg'"
        #
        # For yolov5 val.py:
        #
        # test: WARNING: a/b/c/d.jpg: ignoring corrupt image/label: cannot identify image file '/a/b/c/d.jpg'
        # line = "test: WARNING: a/b/c/d.jpg: ignoring corrupt image/label: cannot identify image file '/a/b/c/d.jpg'"
        if 'cannot identify image file' in line:
            tokens = line.split('cannot identify image file')
            image_name = tokens[-1].strip()
            assert image_name[0] == "'" and image_name [-1] == "'"
            image_name = image_name[1:-1]
            yolo_read_failures.append(image_name)            
        elif 'ignoring corrupt image/label' in line:
            assert 'WARNING' in line
            if '⚠️' in line:
                assert line.startswith('val'), \
                    'Unrecognized line in YOLO output: {}'.format(line)
                tokens = line.split('ignoring corrupt image/label')
                image_name = tokens[0].split('⚠️')[-1].strip()
            else:
                assert line.startswith('test'), \
                    'Unrecognized line in YOLO output: {}'.format(line)
                tokens = line.split('ignoring corrupt image/label')
                image_name = tokens[0].split('WARNING:')[-1].strip()
            assert image_name.endswith(':')
            image_name = image_name[0:-1]
            yolo_read_failures.append(image_name)
                    
    # image_file = yolo_read_failures[0]
    for image_file in yolo_read_failures:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        assert image_id in image_id_to_file, 'Unexpected image ID {}'.format(image_id)
        if image_id not in image_id_to_error:
            image_id_to_error[image_id] = 'YOLO read failure'
    
    if options.yolo_working_folder is not None:
        os.chdir(current_dir)
        
    
    ##%% Convert results to MD format
    
    json_files = glob.glob(yolo_results_folder + '/yolo_results/*.json')
    assert len(json_files) == 1    
    yolo_json_file = json_files[0]

    # Map YOLO image IDs to paths
    image_id_to_relative_path = {}
    for image_id in image_id_to_file:
        fn = image_id_to_file[image_id].replace('\\','/')
        assert path_is_abs(fn)
        if options.input_folder is not None:
            assert os.path.isdir(options.input_folder)
            assert options.input_folder in fn, 'Internal error: base folder {} not in file {}'.format(
                options.input_folder,fn)
            relative_path = os.path.relpath(fn,options.input_folder)
        else:
            # We'll use the absolute path as a relative path, and pass '/'
            # as the base path in this case.
            relative_path = fn
        image_id_to_relative_path[image_id] = relative_path
        
    # Are we working with a base folder?
    if options.input_folder is not None:
        assert os.path.isdir(options.input_folder)
        image_base = options.input_folder
    else:
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
    
    _clean_up_temporary_folders(options,
                                symlink_folder,yolo_results_folder,
                                symlink_folder_is_temp_folder,yolo_folder_is_temp_folder)
    
# ...def run_inference_with_yolo_val()


#%% Command-line driver

import argparse
from megadetector.utils.ct_utils import args_to_object

def main():
    
    options = YoloInferenceOptions()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_filename',type=str,
        help='model file name')
    parser.add_argument(
        'input_folder',type=str,
        help='folder on which to recursively run the model, or a .json or .txt file containing a list of absolute image paths')
    parser.add_argument(
        'output_file',type=str,
        help='.json file where output will be written')
    
    parser.add_argument(
        '--image_filename_list',type=str,default=None,
        help='.json or .txt file containing a list of relative image filenames within [input_folder]')
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
        '--half_precision_enabled', default=None, type=int,
        help='use half-precision-inference (1 or 0) (default is the underlying model\'s default, probably full for YOLOv8 and half for YOLOv5')
    parser.add_argument(
        '--device_string', default=options.device_string, type=str,
        help='CUDA device specifier, typically "0" or "1" for CUDA devices, "mps" for M1/M2 devices, or "cpu" (default {})'.format(
            options.device_string))
    parser.add_argument(
        '--overwrite_handling', default=options.overwrite_handling, type=str,
        help='action to take if the output file exists (skip, error, overwrite) (default {})'.format(
            options.overwrite_handling))
    parser.add_argument(
        '--yolo_dataset_file', default=None, type=str,
        help='YOLOv5 dataset.yaml file from which we should load category information ' + \
            '(otherwise defaults to MD categories)')
    parser.add_argument(
        '--model_type', default=options.model_type, type=str,
        help='model type ("yolov5" or "ultralytics" ("yolov8" behaves the same as "ultralytics")) (default {})'.format(
            options.model_type))

    parser.add_argument('--unique_id_strategy', default=options.unique_id_strategy, type=str,
        help='how should we ensure that unique filenames are passed to the YOLO val script, ' + \
             'can be "verify", "auto", or "links", see options class docs for details (default {})'.format(
                 options.unique_id_strategy))
    parser.add_argument(
        '--symlink_folder', default=None, type=str,
        help='temporary folder for symlinks (defaults to a folder in the system temp dir)')
    parser.add_argument(
        '--yolo_results_folder', default=None, type=str,
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
        '--save_yolo_debug_output', action='store_true',
        help='write yolo console output to a text file in the results folder, along with additional debug files')
    parser.add_argument(
        '--checkpoint_frequency', default=options.checkpoint_frequency, type=int,
        help='break the job into chunks with no more than this many images (default {})'.format(
            options.checkpoint_frequency))    
    
    parser.add_argument(
        '--nonrecursive', action='store_true',
        help='Disable recursive folder processing')
    
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
        assert args.augment_enabled in (0,1), \
            'Illegal augment_enabled value {}'.format(args.augment_enabled)
        if args.augment_enabled == 1:
            args.image_size = default_image_size_with_augmentation
        else:
            args.image_size = default_image_size_with_no_augmentation
        augment_enabled_string = 'enabled'
        if not args.augment_enabled:
            augment_enabled_string = 'disabled'
        print('Augmentation is {}, using default image size {}'.format(
            augment_enabled_string,args.image_size))
        
    args_to_object(args, options)
    
    if args.yolo_dataset_file is not None:
        options.yolo_category_id_to_name = args.yolo_dataset_file
    
    # The function convention is that input_folder should be None when we want to use a list of 
    # absolute paths, but the CLI convention is that the required argument is always valid, whether 
    # it's a folder or a list of absolute paths.
    if os.path.isfile(options.input_folder):
        assert options.image_filename_list is None, \
            'image_filename_list should not be specified when input_folder is a file'
        options.image_filename_list = options.input_folder
        options.input_folder = None        
        
    options.recursive = (not options.nonrecursive)
    options.remove_symlink_folder = (not options.no_remove_symlink_folder)
    options.remove_yolo_results_folder = (not options.no_remove_yolo_results_folder)
    options.use_symlinks = (not options.no_use_symlinks)
    options.augment = (options.augment_enabled > 0)        
    
    del options.nonrecursive
    del options.no_remove_symlink_folder
    del options.no_remove_yolo_results_folder
    del options.no_use_symlinks
    del options.augment_enabled
    del options.yolo_dataset_file
        
    print(options.__dict__)
    
    run_inference_with_yolo_val(options)    

if __name__ == '__main__':
    main()


#%% Interactive driver

if False:

    #%% Run inference on a folder
    
    input_folder = r'g:\temp\tegu-val-mini'.replace('\\','/')
    model_filename = r'g:\temp\usgs-tegus-yolov5x-231003-b8-img1280-e3002-best.pt'
    output_folder = r'g:\temp\tegu-scratch'
    yolo_working_folder = r'c:\git\yolov5-tegus'
    dataset_file = r'g:\temp\dataset.yaml'
    
    # This only impacts the output file name, it's not passed to the inference function
    job_name = 'yolo-inference-test'
    
    model_name = os.path.splitext(os.path.basename(model_filename))[0]
    
    symlink_folder = os.path.join(output_folder,'symlinks')
    yolo_results_folder = os.path.join(output_folder,'yolo_results')
    
    output_file = os.path.join(output_folder,'{}_{}-md_format.json'.format(
        job_name,model_name))
    
    options = YoloInferenceOptions()
    
    options.yolo_working_folder = yolo_working_folder
    options.input_folder = input_folder
    options.output_file = output_file
    
    pass_image_filename_list = False    
    pass_relative_paths = True
    
    if pass_image_filename_list:
        if pass_relative_paths:
            options.image_filename_list =  [
                r"val#american_cardinal#american_cardinal#CaCa#31W.01_C83#2017-2019#C90 and C83_31W.01#(05) 18AUG17 - 05SEP17 FTC AEG#MFDC1949_000065.JPG",
                r"val#american_cardinal#american_cardinal#CaCa#31W.01_C83#2017-2019#C90 and C83_31W.01#(04) 27JUL17 - 18AUG17 FTC AEG#MFDC1902_000064.JPG"
            ]   
        else:
            options.image_filename_list =  [
                r"g:/temp/tegu-val-mini/val#american_cardinal#american_cardinal#CaCa#31W.01_C83#2017-2019#C90 and C83_31W.01#(05) 18AUG17 - 05SEP17 FTC AEG#MFDC1949_000065.JPG",
                r"g:/temp/tegu-val-mini/val#american_cardinal#american_cardinal#CaCa#31W.01_C83#2017-2019#C90 and C83_31W.01#(04) 27JUL17 - 18AUG17 FTC AEG#MFDC1902_000064.JPG"
            ]
    else:
        options.image_filename_list = None    
    
    options.yolo_category_id_to_name = dataset_file
    options.augment = False
    options.conf_thres = '0.001'
    options.batch_size = 1
    options.device_string = '0'
    options.unique_id_strategy = 'auto'
    options.overwrite_handling = 'overwrite'

    if options.augment:
        options.image_size = round(1280 * 1.3)
    else:
        options.image_size = 1280
    
    options.model_filename = model_filename
    
    options.yolo_results_folder = yolo_results_folder # os.path.join(output_folder + 'yolo_results')        
    options.symlink_folder = symlink_folder # os.path.join(output_folder,'symlinks')
    options.use_symlinks = False
    
    options.remove_symlink_folder = True
    options.remove_yolo_results_folder = True
    
    options.checkpoint_frequency = 5
    
    cmd = f'python run_inference_with_yolov5_val.py {model_filename} {input_folder} ' + \
          f'{output_file} --yolo_working_folder {yolo_working_folder} ' + \
          f' --image_size {options.image_size} --conf_thres {options.conf_thres} ' + \
          f' --batch_size {options.batch_size} ' + \
          f' --symlink_folder {options.symlink_folder} --yolo_results_folder {options.yolo_results_folder} ' + \
          f' --yolo_dataset_file {options.yolo_category_id_to_name} ' + \
          f' --unique_id_strategy {options.unique_id_strategy} --overwrite_handling {options.overwrite_handling}'
      
    if not options.remove_symlink_folder:
        cmd += ' --no_remove_symlink_folder'
    if not options.remove_yolo_results_folder: 
        cmd += ' --no_remove_yolo_results_folder'
    if options.checkpoint_frequency is not None:
        cmd += f' --checkpoint_frequency {options.checkpoint_frequency}'
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

