"""

yolo_to_coco.py

Converts a folder of YOLO-formatted annotation files to a COCO-formatted dataset. 

"""

#%% Imports and constants

import json
import os

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial

from tqdm import tqdm

from megadetector.utils.path_utils import find_images
from megadetector.utils.path_utils import recursive_file_list
from megadetector.utils.path_utils import find_image_strings
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.visualization.visualization_utils import open_image
from megadetector.data_management.yolo_output_to_md_output import read_classes_from_yolo_dataset_file


#%% Support functions

def _filename_to_image_id(fn):
    """
    Image IDs can't have spaces in them, replace spaces with underscores
    """
    return fn.replace(' ','_').replace('\\','/')


def _process_image(fn_abs,input_folder,category_id_to_name):
    """
    Internal support function for processing one image's labels.
    """
    
    # Create the image object for this image
    #
    # Always use forward slashes in image filenames and IDs
    fn_relative = os.path.relpath(fn_abs,input_folder).replace('\\','/')
    image_id = _filename_to_image_id(fn_relative)
    
    # This is done in a separate loop now
    #
    # assert image_id not in image_ids, \
    #    'Oops, you have hit a very esoteric case where you have the same filename ' + \
    #    'with both spaces and underscores, this is not currently handled.'
    # image_ids.add(image_id)
    
    im = {}
    im['file_name'] = fn_relative
    im['id'] = image_id
    
    annotations_this_image = []
    
    try:        
        pil_im = open_image(fn_abs)
        im_width, im_height = pil_im.size
        im['width'] = im_width
        im['height'] = im_height
        im['error'] = None
    except Exception as e:
        print('Warning: error reading {}:\n{}'.format(fn_relative,str(e)))
        im['width'] = -1
        im['height'] = -1
        im['error'] = str(e)
        return (im,annotations_this_image)
        
    # Is there an annotation file for this image?
    annotation_file = os.path.splitext(fn_abs)[0] + '.txt'
    if not os.path.isfile(annotation_file):
        annotation_file = os.path.splitext(fn_abs)[0] + '.TXT'
    
    if os.path.isfile(annotation_file):
        
        with open(annotation_file,'r') as f:
            lines = f.readlines()
        lines = [s.strip() for s in lines]
        
        # s = lines[0]
        annotation_number = 0
        
        for s in lines:
            
            if len(s.strip()) == 0:
                continue
            
            tokens = s.split()
            assert len(tokens) == 5
            category_id = int(tokens[0])
            assert category_id in category_id_to_name, \
                'Unrecognized category ID {} in annotation file {}'.format(
                    category_id,annotation_file)
            ann = {}
            ann['id'] = im['id'] + '_' + str(annotation_number)
            ann['image_id'] = im['id']
            ann['category_id'] = category_id
            ann['sequence_level_annotation'] = False
            
            # COCO: [x_min, y_min, width, height] in absolute coordinates
            # YOLO: [class, x_center, y_center, width, height] in normalized coordinates
            
            yolo_bbox = [float(x) for x in tokens[1:]]
            
            normalized_x_center = yolo_bbox[0]
            normalized_y_center = yolo_bbox[1]
            normalized_width = yolo_bbox[2]
            normalized_height = yolo_bbox[3]
            
            absolute_x_center = normalized_x_center * im_width 
            absolute_y_center = normalized_y_center * im_height
            absolute_width = normalized_width * im_width
            absolute_height = normalized_height * im_height
            absolute_x_min = absolute_x_center - absolute_width / 2
            absolute_y_min = absolute_y_center - absolute_height / 2
            
            coco_bbox = [absolute_x_min, absolute_y_min, absolute_width, absolute_height]
            
            ann['bbox'] = coco_bbox
            annotation_number += 1
            
            annotations_this_image.append(ann)                
            
        # ...for each annotation 
        
    # ...if this image has annotations
    
    return (im,annotations_this_image)

# ...def _process_image(...)


def load_yolo_class_list(class_name_file):
    """
    Loads a dictionary mapping zero-indexed IDs to class names from the text/yaml file
    [class_name_file].    
    
    Args:
        class_name_file (str or list): this can be:
            - a .yaml or .yaml file in YOLO's dataset.yaml format
            - a .txt or .data file containing a flat list of class names
            - a list of class names
            
    Returns:
        dict: A dict mapping zero-indexed integer IDs to class names
    """
    
    # class_name_file can also be a list of class names
    if isinstance(class_name_file,list):
        category_id_to_name = {}
        for i_name,name in enumerate(class_name_file):
            category_id_to_name[i_name] = name
        return category_id_to_name
            
    ext = os.path.splitext(class_name_file)[1][1:]
    assert ext in ('yml','txt','yaml','data'), 'Unrecognized class name file type {}'.format(
        class_name_file)
    
    if ext in ('txt','data'):
        
        with open(class_name_file,'r') as f:
            lines = f.readlines()
        assert len(lines) > 0, 'Empty class name file {}'.format(class_name_file)
        class_names = [s.strip() for s in lines]
        assert len(lines[0]) > 0, 'Empty class name file {} (empty first line)'.format(class_name_file)
        
        # Blank lines should only appear at the end
        b_found_blank = False
        for s in lines:
            if len(s) == 0:
                b_found_blank = True
            elif b_found_blank:
                raise ValueError('Invalid class name file {}, non-blank line after the last blank line'.format(
                    class_name_file))
    
        category_id_to_name = {}        
        for i_category_id,category_name in enumerate(class_names):
            assert len(category_name) > 0
            category_id_to_name[i_category_id] = category_name
            
    else:
        
        assert ext in ('yml','yaml')
        category_id_to_name = read_classes_from_yolo_dataset_file(class_name_file)
        
    return category_id_to_name

# ...load_yolo_class_list(...)


def validate_label_file(label_file,category_id_to_name=None,verbose=False):
    """"
    Verifies that [label_file] is a valid YOLO label file.  Does not check the extension.
    
    Args:
        label_file (str): the .txt file to validate
        category_id_to_name (dict, optional): a dict mapping integer category IDs to names;
            if this is not None, this function errors if the file uses a category that's not
            in this dict
        verbose (bool, optional): enable additional debug console output
    
    Returns:
        dict: a dict with keys 'file' (the same as [label_file]) and 'errors' (a list of 
        errors (if any) that we found in this file)
    """
    
    label_result = {}
    label_result['file'] = label_file
    label_result['errors'] = []
    
    try:
        with open(label_file,'r') as f:
            lines = f.readlines()
    except Exception as e:
        label_result['errors'].append('Read error: {}'.format(str(e)))
        return label_result
    
    # i_line 0; line = lines[i_line]
    for i_line,line in enumerate(lines):
        s = line.strip()
        if len(s) == 0 or s[0] == '#':
            continue
        
        try:
        
            tokens = s.split()
            assert len(tokens) == 5, '{} tokens'.format(len(tokens))                
            
            if category_id_to_name is not None:
                category_id = int(tokens[0])
                assert category_id in category_id_to_name, \
                    'Unrecognized category ID {}'.format(category_id)
        
            yolo_bbox = [float(x) for x in tokens[1:]]
            
        except Exception as e:
            label_result['errors'].append('Token error at line {}: {}'.format(i_line,str(e)))
            continue
                            
        normalized_x_center = yolo_bbox[0]
        normalized_y_center = yolo_bbox[1]
        normalized_width = yolo_bbox[2]
        normalized_height = yolo_bbox[3]
        
        normalized_x_min = normalized_x_center - normalized_width / 2.0
        normalized_x_max = normalized_x_center + normalized_width / 2.0
        normalized_y_min = normalized_y_center - normalized_height / 2.0
        normalized_y_max = normalized_y_center + normalized_height / 2.0
        
        if normalized_x_min < 0 or normalized_y_min < 0 or \
            normalized_x_max > 1 or normalized_y_max > 1:
            label_result['errors'].append('Invalid bounding box: {} {} {} {}'.format(
                normalized_x_min,normalized_y_min,normalized_x_max,normalized_y_max))
        
    # ...for each line
    
    if verbose:
        if len(label_result['errors']) > 0:
            print('Errors for {}:'.format(label_file))
            for error in label_result['errors']:
                print(error)
                
    return label_result
    
# ...def validate_label_file(...)

    
def validate_yolo_dataset(input_folder, class_name_file, n_workers=1, pool_type='thread', verbose=False):
    """
    Verifies all the labels in a YOLO dataset folder.  
    
    Looks for:
        
    * Image files without label files
    * Text files without image files
    * Illegal classes in label files
    * Invalid boxes in label files

    Args:
        input_folder (str): the YOLO dataset folder to validate
        class_name_file (str or list): a list of classes, a flat text file, or a yolo 
            dataset.yml/.yaml file.  If it's a dataset.yml file, that file should point to 
            input_folder as the base folder, though this is not explicitly checked.
        n_workers (int, optional): number of concurrent workers, set to <= 1 to disable
            parallelization
        pool_type (str, optional): 'thread' or 'process', worker type to use for parallelization;
            not used if [n_workers] <= 1
        verbose (bool, optional): enable additional debug console output
    
    Returns:
        dict: validation results, as a dict with fields:        
        
        - image_files_without_label_files (list)
        - label_files_without_image_files (list)
        - label_results (list of dicts with field 'filename', 'errors') (list)
    """
    
    # Validate arguments
    assert os.path.isdir(input_folder), 'Could not find input folder {}'.format(input_folder)
    if n_workers > 1:
        assert pool_type in ('thread','process'), 'Illegal pool type {}'.format(pool_type)
        
    category_id_to_name = load_yolo_class_list(class_name_file)
    
    print('Enumerating files in {}'.format(input_folder))
    
    all_files = recursive_file_list(input_folder,recursive=True,return_relative_paths=False,
                                    convert_slashes=True)
    label_files = [fn for fn in all_files if fn.endswith('.txt')]
    image_files = find_image_strings(all_files)
    print('Found {} images files and {} label files in {}'.format(
        len(image_files),len(label_files),input_folder))
    
    label_files_set = set(label_files)
    
    image_files_without_extension = set()
    for fn in image_files:
        image_file_without_extension = os.path.splitext(fn)[0]
        assert image_file_without_extension not in image_files_without_extension, \
            'Duplicate image file, likely with different extensions: {}'.format(fn)
        image_files_without_extension.add(image_file_without_extension)
        
    print('Looking for missing image/label files')
    
    image_files_without_label_files = []
    label_files_without_images = []
    
    for image_file in tqdm(image_files):
        expected_label_file = os.path.splitext(image_file)[0] + '.txt'
        if expected_label_file not in label_files_set:
            image_files_without_label_files.append(image_file)
            
    for label_file in tqdm(label_files):
        expected_image_file_without_extension = os.path.splitext(label_file)[0]
        if expected_image_file_without_extension not in image_files_without_extension:
            label_files_without_images.append(label_file)
            
    print('Found {} image files without labels, {} labels without images'.format(
        len(image_files_without_label_files),len(label_files_without_images)))

    print('Validating label files')
    
    if n_workers <= 1:
        
        label_results = []        
        for fn_abs in tqdm(label_files):                
            label_results.append(validate_label_file(fn_abs,
                                                      category_id_to_name=category_id_to_name,
                                                      verbose=verbose))
            
    else:
        
        assert pool_type in ('process','thread'), 'Illegal pool type {}'.format(pool_type)
        
        if pool_type == 'thread':
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)
        
        print('Starting a {} pool of {} workers'.format(pool_type,n_workers))
        
        p = partial(validate_label_file,
                    category_id_to_name=category_id_to_name,
                    verbose=verbose)
        label_results = list(tqdm(pool.imap(p, label_files),
                                  total=len(label_files)))        
    
    assert len(label_results) == len(label_files)
    
    validation_results = {}
    validation_results['image_files_without_label_files'] = image_files_without_label_files
    validation_results['label_files_without_images'] = label_files_without_images
    validation_results['label_results'] = label_results
    
    return validation_results
    
# ...validate_yolo_dataset(...)    


#%% Main conversion function

def yolo_to_coco(input_folder,
                 class_name_file,
                 output_file=None,
                 empty_image_handling='no_annotations',
                 empty_image_category_name='empty',
                 error_image_handling='no_annotations',
                 allow_images_without_label_files=True,
                 n_workers=1,
                 pool_type='thread',
                 recursive=True,
                 exclude_string=None,
                 include_string=None,
                 overwrite_handling='overwrite'):
    """
    Converts a YOLO-formatted dataset to a COCO-formatted dataset.
    
    All images will be assigned an "error" value, usually None.    
    
    Args:
        input_folder (str): the YOLO dataset folder to validate
        class_name_file (str or list): a list of classes, a flat text file, or a yolo 
            dataset.yml/.yaml file.  If it's a dataset.yml file, that file should point to 
            input_folder as the base folder, though this is not explicitly checked.
        output_file (str, optional): .json file to which we should write COCO .json data
        empty_image_handling (str, optional): how to handle images with no boxes; whether
            this includes images with no .txt files depending on the value of 
            [allow_images_without_label_files].  Can be:
                
            - 'no_annotations': include the image in the image list, with no annotations
            - 'empty_annotations': include the image in the image list, and add an annotation without
              any bounding boxes, using a category called [empty_image_category_name].
            - 'skip': don't include the image in the image list
            - 'error': there shouldn't be any empty images            
        error_image_handling (str, optional): how to handle images that don't load properly; can
            be:
            
            - 'skip': don't include the image at all
            - 'no_annotations': include with no annotations
            
        n_workers (int, optional): number of concurrent workers, set to <= 1 to disable
            parallelization
        pool_type (str, optional): 'thread' or 'process', worker type to use for parallelization;
            not used if [n_workers] <= 1
        recursive (bool, optional): whether to recurse into [input_folder]
        exclude_string (str, optional): exclude any images whose filename contains a string
        include_string (str, optional): include only images whose filename contains a string
        overwrite_handling (bool, optional): behavior if output_file exists ('load', 'overwrite', or 
            'error')
    
    Returns:
        dict: COCO-formatted data, the same as what's written to [output_file]
    """
    
    ## Validate input
    
    assert os.path.isdir(input_folder)
    assert os.path.isfile(class_name_file)
    
    assert empty_image_handling in \
        ('no_annotations','empty_annotations','skip','error'), \
            'Unrecognized empty image handling spec: {}'.format(empty_image_handling)
     
    if (output_file is not None) and os.path.isfile(output_file):
        
            if overwrite_handling == 'overwrite':
                print('Warning: output file {} exists, over-writing'.format(output_file))
            elif overwrite_handling == 'load':
                print('Output file {} exists, loading and returning'.format(output_file))
                with open(output_file,'r') as f:
                    d = json.load(f)
                return d
            elif overwrite_handling == 'error':
                raise ValueError('Output file {} exists'.format(output_file))
            else:
                raise ValueError('Unrecognized overwrite_handling value: {}'.format(overwrite_handling))
                
                
    ## Read class names
    
    category_id_to_name = load_yolo_class_list(class_name_file)
    
    
    # Find or create the empty image category, if necessary
    empty_category_id = None
    
    if (empty_image_handling == 'empty_annotations'):
        category_name_to_id = invert_dictionary(category_id_to_name)
        if empty_image_category_name in category_name_to_id:
            empty_category_id = category_name_to_id[empty_image_category_name]
            print('Using existing empty image category with name {}, ID {}'.format(
                empty_image_category_name,empty_category_id))            
        else:
            empty_category_id = len(category_id_to_name)
            print('Adding an empty category with name {}, ID {}'.format(
                empty_image_category_name,empty_category_id))
            category_id_to_name[empty_category_id] = empty_image_category_name
            
            
    ## Enumerate images
    
    print('Enumerating images...')
    
    image_files_abs = find_images(input_folder,recursive=recursive,convert_slashes=True)

    n_files_original = len(image_files_abs)
    
    # Optionally include/exclude images matching specific strings
    if exclude_string is not None:
        image_files_abs = [fn for fn in image_files_abs if exclude_string not in fn]
    if include_string is not None:
        image_files_abs = [fn for fn in image_files_abs if include_string in fn]
    
    if len(image_files_abs) != n_files_original or exclude_string is not None or include_string is not None:
        n_excluded = n_files_original - len(image_files_abs)
        print('Excluded {} of {} images based on filenames'.format(n_excluded,n_files_original))
        
    categories = []
    
    for category_id in category_id_to_name:
        categories.append({'id':category_id,'name':category_id_to_name[category_id]})
        
    info = {}
    info['version'] = '1.0'
    info['description'] = 'Converted from YOLO format'
    
    image_ids = set()
    
    
    ## If we're expected to have labels for every image, check before we process all the images
    
    if not allow_images_without_label_files:
        print('Verifying that label files exist')
        for image_file_abs in tqdm(image_files_abs):
            label_file_abs = os.path.splitext(image_file_abs)[0] + '.txt'
            assert os.path.isfile(label_file_abs), \
                'No annotation file for {}'.format(image_file_abs)
    
    
    ## Initial loop to make sure image IDs will be unique
    
    print('Validating image IDs...')
    
    for fn_abs in tqdm(image_files_abs):
        
        fn_relative = os.path.relpath(fn_abs,input_folder)
        image_id = _filename_to_image_id(fn_relative)
        assert image_id not in image_ids, \
            'Oops, you have hit a very esoteric case where you have the same filename ' + \
            'with both spaces and underscores, this is not currently handled.'
        image_ids.add(image_id)
    
    
    ## Main loop to process labels
    
    print('Processing labels...')
    
    if n_workers <= 1:
        
        image_results = []        
        for fn_abs in tqdm(image_files_abs):                
            image_results.append(_process_image(fn_abs,input_folder,category_id_to_name))
            
    else:
        
        assert pool_type in ('process','thread'), 'Illegal pool type {}'.format(pool_type)
        
        if pool_type == 'thread':
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)
        
        print('Starting a {} pool of {} workers'.format(pool_type,n_workers))
        
        p = partial(_process_image,input_folder=input_folder,
                    category_id_to_name=category_id_to_name)
        image_results = list(tqdm(pool.imap(p, image_files_abs),
                                  total=len(image_files_abs)))
                
    
    assert len(image_results) == len(image_files_abs)
    
    
    ## Re-assembly of results into a COCO dict
    
    print('Assembling labels...')
    
    images = []
    annotations = []
    
    for image_result in tqdm(image_results):
    
        im = image_result[0]
        annotations_this_image = image_result[1]
           
        # If we have annotations for this image
        if len(annotations_this_image) > 0:
            assert im['error'] is None
            images.append(im)
            for ann in annotations_this_image:
                annotations.append(ann)
                
        # If this image failed to read
        elif im['error'] is not None:
            
            if error_image_handling == 'skip':
                pass
            elif error_image_handling == 'no_annotations':
                images.append(im)            
                
        # If this image read successfully, but there are no annotations
        else:
            
            if empty_image_handling == 'skip':
                pass
            elif empty_image_handling == 'no_annotations':
                images.append(im)
            elif empty_image_handling == 'empty_annotations':
                assert empty_category_id  is not None
                ann = {}
                ann['id'] = im['id'] + '_0'
                ann['image_id'] = im['id']
                ann['category_id'] = empty_category_id
                ann['sequence_level_annotation'] = False
                # This would also be a reasonable thing to do, but it's not the convention
                # we're adopting.
                # ann['bbox'] = [0,0,0,0]
                annotations.append(ann)
                images.append(im)        
        
    # ...for each image result
    
    print('Read {} annotations for {} images'.format(len(annotations),
                                                     len(images)))
    
    d = {}
    d['images'] = images
    d['annotations'] = annotations
    d['categories'] = categories
    d['info'] = info

    if output_file is not None:
        print('Writing to {}'.format(output_file))
        with open(output_file,'w') as f:
            json.dump(d,f,indent=1)

    return d

# ...def yolo_to_coco()


#%% Interactive driver

if False:
    
    pass

    #%% Convert YOLO folders to COCO
    
    preview_folder = '/home/user/data/noaa-fish/val-coco-conversion-preview'
    input_folder = '/home/user/data/noaa-fish/val'
    output_file = '/home/user/data/noaa-fish/val.json'
    class_name_file = '/home/user/data/noaa-fish/AllImagesWithAnnotations/classes.txt'

    d = yolo_to_coco(input_folder,class_name_file,output_file)
        
    input_folder = '/home/user/data/noaa-fish/train'
    output_file = '/home/user/data/noaa-fish/train.json'
    class_name_file = '/home/user/data/noaa-fish/AllImagesWithAnnotations/classes.txt'

    d = yolo_to_coco(input_folder,class_name_file,output_file)
    
    
    #%% Check DB integrity

    from megadetector.data_management.databases import integrity_check_json_db

    options = integrity_check_json_db.IntegrityCheckOptions()
    options.baseDir = input_folder
    options.bCheckImageSizes = False
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True

    _, _, _ = integrity_check_json_db.integrity_check_json_db(output_file, options)


    #%% Preview some images

    from megadetector.visualization import visualize_db

    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = None
    viz_options.trim_to_images_with_bboxes = False
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True

    html_output_file, _ = visualize_db.visualize_db(db_path=output_file,
                                                        output_dir=preview_folder,
                                                        image_base_dir=input_folder,
                                                        options=viz_options)
    
    from megadetector.utils.path_utils import open_file
    open_file(html_output_file)


#%% Command-line driver

# TODO
