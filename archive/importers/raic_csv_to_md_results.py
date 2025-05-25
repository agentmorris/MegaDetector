"""

raic_csv_to_md_results.py

Converts classification+detection results in the .csv format provided to the Snapshot
Serengeti program by the RAIC team to the MD results format.

The input format is two .csv files:
    
* One with results, with columns [unnamed], filename, category, x_center, y_center,
  width, height, confidence, datetime

* One with class IDs and names, with columns CLASS, SPECIES

Filenames are relative paths to .txt files, but with slashes replaced by underscores, e.g. this 
file:
    
    B04_R1/I__00122.JPG
    
...appears in the .csv file as:
    
    B04_R1_I__00122.txt
    
Image coordinates are in absolute floating-point units, with an upper-left origin.
  
Unknowns at the time I'm writing this:
    
* I don't know what the unnamed column is, but it looks like an ID I can safely ignore.

* I believe that MegaDetector was run, then a classifier was run, but there is a 
  single "confidence" column in the output.  I am writing out the results as if they were a 
  single multi-class detector.  This is suspicious given the lack of a human class, which suggests
  that this is intended to be run in conjunection with MD.

* There is no concept of "empty" in this file format, so by default I assume that images with
  no annotations in the .csv file were processed and determine to have no detections above some
  (unknown) threshold.
  
* I'm not currently handling EXIF rotations, as part of the effort to simplify this file
  for conversion to R (see below).
  
Note to self: this file should not take dependencies on other components of the MD
repo, at the risk of creating some redundancy.  I am going to convert this to R,
which will be easier if it's not using any non-standard libraries.  Anything in the 
"interactive driver" cells gets a pass.

"""

#%% Imports and constants

import os
import glob
import json
import sys
import argparse

import pandas as pd
from PIL import Image


#%% Functions from the MD python package

# ...that I'm choosing to copy and paste to facilitate a conversion of this
# script to R.

# Should all be lower-case
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')

def _is_image_file(s, img_extensions=IMG_EXTENSIONS):
    """
    Checks a file's extension against a hard-coded set of image file
    extensions.  Uses case-insensitive comparison.
    
    Does not check whether the file exists, only determines whether the filename
    implies it's an image file.
    
    Args:
        s (str): filename to evaluate for image-ness
        img_extensions (list, optional): list of known image file extensions
        
    Returns:
        bool: True if [s] appears to be an image file, else False
    """
    
    ext = os.path.splitext(s)[1]
    return ext.lower() in img_extensions


def _find_image_strings(strings):
    """
    Given a list of strings that are potentially image file names, looks for
    strings that actually look like image file names (based on extension).
    
    Args:
        strings (list): list of filenames to check for image-ness
        
    Returns:
        list: the subset of [strings] that appear to be image filenames
    """
    
    return [s for s in strings if _is_image_file(s)]


def _find_images(dirname, 
                recursive=False, 
                return_relative_paths=False, 
                convert_slashes=True):
    """
    Finds all files in a directory that look like image file names. Returns
    absolute paths unless return_relative_paths is set.  Uses the OS-native
    path separator unless convert_slashes is set, in which case will always
    use '/'.
    
    Args:
        dirname (str): the folder to search for images
        recursive (bool, optional): whether to search recursively
        return_relative_paths (str, optional): return paths that are relative
            to [dirname], rather than absolute paths
        convert_slashes (bool, optional): force forward slashes in return values

    Returns:
        list: list of image filenames found in [dirname]
    """
    
    assert os.path.isdir(dirname), '{} is not a folder'.format(dirname)
    
    if recursive:
        strings = glob.glob(os.path.join(dirname, '**', '*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirname, '*.*'))
    
    image_files = _find_image_strings(strings)
    
    if return_relative_paths:
        image_files = [os.path.relpath(fn,dirname) for fn in image_files]
    
    image_files = sorted(image_files)
    
    if convert_slashes:
        image_files = [fn.replace('\\', '/') for fn in image_files]
        
    return image_files


#%% Main conversion function

def raic_csv_to_md_results(result_csv_file,
                           class_mapping_csv_file,
                           image_folder,
                           output_file=None,
                           unannotated_image_handling='empty'):
    """
    Converts a pair of .csv files (see file header for details) to MD results format.
    
    Currently errors if image filenames are ambiguous, or if any images referred to in 
    the results are not available.
    
    Args:
        result_csv_file (str): the results file to read (.csv)
        class_mapping_csv_file (str): the class mapping file (.csv)
        image_folder (str): the folder containing all the images referred to in
            [result_csv_file]
        output_file (str, optional): the .json file to which we should write results.  Defaults
            to [result_csv_file].json
        unannotated_image_handling (str, optional): can be "empty" (default) to assume
            that images without annotations are empty, "warning", "error", or "skip"
            
    Returns:
        str: the output file written, identical to [output_file] if [output_file] was not None
    """
    
    # Validate arguments    
    assert os.path.isfile(result_csv_file), \
        'Result file {} not found'.format(result_csv_file)
    assert os.path.isfile(class_mapping_csv_file), \
        'Class mapping file {} not found'.format(class_mapping_csv_file)
    assert os.path.isdir(image_folder), \
        'Image folder {} not found'.format(image_folder)
    
    if output_file is None:
        output_file = result_csv_file + '.json'
        
    image_files_relative = _find_images(image_folder,
                                        recursive=True,
                                        return_relative_paths=True,
                                        convert_slashes=True)
    image_file_base_flattened_to_image_file_relative = {}
    for fn in image_files_relative:
        # Convert, e.g. B04_R1/I__00108.JPG to B04_R1_I__00108
        fn_flattened = fn.replace('/','_')
        fn_flattened_base = os.path.splitext(fn_flattened)[0]
        image_file_base_flattened_to_image_file_relative[fn_flattened_base] = \
            fn    
        
    # Read the .csv files
    df_results = pd.read_csv(result_csv_file)
    df_class_mapping = pd.read_csv(class_mapping_csv_file)
    
    assert 'CLASS' in df_class_mapping.columns and 'SPECIES' in df_class_mapping.columns, \
        'Unexpected column names in class mapping file {}'.format(class_mapping_csv_file)
    
    category_id_to_name = {}
    for i_row,row in df_class_mapping.iterrows():
        class_id = int(row['CLASS'])
        assert class_id not in category_id_to_name, \
            'Class ID {} occurs more than once in class mapping file {}'.format(
                class_id,class_mapping_csv_file)
        category_id_to_name[class_id] = row['SPECIES']
    
    if len(category_id_to_name) != len(set(category_id_to_name.values())):
        print('Warning: one or more categories are used more than once in class mapping file {}'.format(
            class_mapping_csv_file))
        
    # Convert results
    
    fn_relative_to_im = {}
    
    # i_row = 0; row = df_results.iloc[i_row]
    for i_row,row in df_results.iterrows():
        
        # Map the .txt filename base to a relative path
        bn = row['filename']
        assert bn.lower().endswith('.txt')
        bn_no_ext = os.path.splitext(bn)[0]
        assert bn_no_ext in image_file_base_flattened_to_image_file_relative, \
            'No image found for result row {}'.format(row['filename'])
        
        image_fn_relative = image_file_base_flattened_to_image_file_relative[bn_no_ext]
        
        # Have we seen another detection for this image?
        if image_fn_relative in fn_relative_to_im:
            
            im = fn_relative_to_im[image_fn_relative]
        
        # If not, load this image so we can read its size
        else:
            
            image_fn_abs = os.path.join(image_folder,image_fn_relative)
            image = Image.open(image_fn_abs)
            w = image.size[0]
            h = image.size[1]
            
            im = {}
            im['file'] = image_fn_relative
            im['width'] = w
            im['height'] = h
            im['detections'] = []
            im['datetime'] = str(row['datetime'])
            fn_relative_to_im[image_fn_relative] = im
            
        # Convert annotation
        x_center_abs = row['x_center']
        y_center_abs = row['y_center']
        box_width_abs = row['width']
        box_height_abs = row['height']
        
        # Convert to relative coordinates
        box_left_abs = x_center_abs - (box_width_abs/2.0)
        box_top_abs = y_center_abs - (box_height_abs/2.0)
        bbox_normalized = [box_left_abs/im['width'],
                           box_top_abs/im['height'],
                           box_width_abs/im['width'],
                           box_height_abs/im['height']]
        
        category_id = str(int(row['category']))
        confidence = row['confidence']        
        assert isinstance(confidence,float) and confidence <= 1.0 and confidence >= 0.0
        
        det = {}
        im['detections'].append(det)
        det['category'] = category_id
        det['conf'] = confidence
        det['bbox'] = bbox_normalized
                
    # ...for each row
        
    n_empty_images = 0
    
    # Handle images without annotations
    for fn_relative in image_files_relative:
        
        if fn_relative not in fn_relative_to_im:
            if unannotated_image_handling == 'empty':
                im = {}
                im['file'] = fn_relative
                im['detections'] = []                
                fn_relative_to_im[fn_relative] = im
                n_empty_images += 1
                # Don't bother to read width and height here
            elif unannotated_image_handling == 'warning':
                print('Warning: image {} is not represented in the .csv results file'.format(fn_relative))
            elif unannotated_image_handling == 'error':
                raise ValueError('Image {} is not represented in the .csv results file'.format(fn_relative))
            elif unannotated_image_handling == 'skip':
                continue
            
    # ...for each image file
    
    if n_empty_images > 0:
        print('Warning: assuming {} of {} images without annotations are empty'.format(
            n_empty_images,len(image_files_relative)))
        
    images = list(fn_relative_to_im.values())
            
    # The MD output format uses string-ints for category IDs, right now we have ints
    detection_categories = {}
    for category_id_int in category_id_to_name:
        detection_categories[str(category_id_int)] = category_id_to_name[category_id_int]
    
    info = {}
    info['format_version'] = '1.4'
    info['detector'] = 'RAIC .csv converter'
    
    d = {}
    d['images'] = images
    d['detection_categories'] = detection_categories
    d['info'] = info
        
    with open(output_file,'w') as f:
        json.dump(d,f,indent=1)
        
    return output_file

# ...def raic_csv_to_md_results(...)

    
#%% Interactive driver

if False:
    
    pass

    #%% Test conversion
    
    base_folder = r'G:\temp\S24_B04_R1_output_annotations_for_Dan'
    result_csv_file = os.path.join(base_folder,'S24_B04_R1_output_annotations_for_Dan.csv')
    class_mapping_csv_file = os.path.join(base_folder,'categories_key.csv')
    
    # This is wrong, B04_R1 has to be part of the image paths
    # image_folder = os.path.join(base_folder,'B04_R1')
    
    image_folder = base_folder 
    
    output_file = None
    unannotated_image_handling='empty'
    
    output_file = raic_csv_to_md_results(result_csv_file=result_csv_file, 
                                         class_mapping_csv_file=class_mapping_csv_file,
                                         image_folder=image_folder,
                                         output_file=output_file,
                                         unannotated_image_handling=unannotated_image_handling)

    #%% Validate results file
    
    from megadetector.postprocessing.validate_batch_results import \
        ValidateBatchResultsOptions, validate_batch_results
        
    validation_options = ValidateBatchResultsOptions()
    validation_options.check_image_existence = True
    validation_options.relative_path_base = image_folder
    validation_options.return_data = True
    
    results = validate_batch_results(output_file,validation_options)
    assert len(results['validation_results']['errors']) == 0
    assert len(results['validation_results']['warnings']) == 0
    
    
    #%% Preview results
    
    from megadetector.postprocessing.postprocess_batch_results import \
        PostProcessingOptions, process_batch_results
        
    postprocessing_options = PostProcessingOptions()
    
    postprocessing_options.md_results_file = output_file    
    postprocessing_options.output_dir = r'g:\temp\serengeti-conversion-preview'
    postprocessing_options.image_base_dir = image_folder
    postprocessing_options.confidence_threshold = 0.2
    postprocessing_options.num_images_to_sample = None
    postprocessing_options.viz_target_width = 1280
    postprocessing_options.line_thickness = 4    
    postprocessing_options.parallelize_rendering_n_cores = 10
    postprocessing_options.parallelize_rendering_with_threads = True
        
    postprocessing_results = process_batch_results(postprocessing_options)
    
    from megadetector.utils.path_utils import open_file
    open_file(postprocessing_results.output_html_file)
    
    
#%% Command-line driver

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('result_csv_file', type=str, 
                        help='csv file containing AI results')
    parser.add_argument('class_mapping_csv_file', type=str, 
                        help='csv file containing class mappings (with columns CLASS, SPECIES)')
    parser.add_argument('image_folder', type=str, 
                        help='folder containing the images referred to in [result_csv_file]')
    parser.add_argument('--output_file', type=str, default=None,
                        help='.json file to which we should write results (defaults to [result_csv_file].json)')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()
    raic_csv_to_md_results(result_csv_file=args.result_csv_file,
                           class_mapping_csv_file=args.class_mapping_csv_file,
                           image_folder=args.image_folder,
                           output_file=args.output_file)

if __name__ == '__main__':    
    main()
