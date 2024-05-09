"""

labelme_to_coco.py

Converts a folder of labelme-formatted .json files to COCO.

"""

#%% Constants and imports

import json
import os
import uuid

from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from tqdm import tqdm

from megadetector.utils import path_utils
from megadetector.visualization.visualization_utils import open_image


#%% Support functions

def _add_category(category_name,category_name_to_id,candidate_category_id=0):
    """
    Adds the category [category_name] to the dict [category_name_to_id], by default
    using the next available integer index.    
    """
    
    if category_name in category_name_to_id:
        return category_name_to_id[category_name]
    while candidate_category_id in category_name_to_id.values():
        candidate_category_id += 1
    category_name_to_id[category_name] = candidate_category_id
    return candidate_category_id


def _process_labelme_file(image_fn_relative,input_folder,use_folders_as_labels,
                          no_json_handling,validate_image_sizes,
                          category_name_to_id,allow_new_categories=True):
    """
    Internal function for processing each image; this support function facilitates parallelization.
    """
    
    result = {}
    result['im'] = None
    result['annotations_this_image'] = None
    result['status'] = None
    
    image_fn_abs = os.path.join(input_folder,image_fn_relative)
    json_fn_abs = os.path.splitext(image_fn_abs)[0] + '.json'
    
    im = {}
    im['id'] = image_fn_relative
    im['file_name'] = image_fn_relative
    
    # If there's no .json file for this image...
    if not os.path.isfile(json_fn_abs):
        
        # Either skip it...
        if no_json_handling == 'skip':
            print('Skipping image {} (no .json file)'.format(image_fn_relative))
            result['status'] = 'skipped (no .json file)'
            return result
        
        # ...or error
        elif no_json_handling == 'error':
            raise ValueError('Image file {} has no corresponding .json file'.format(
                image_fn_relative))
        
        # ...or treat it as empty.
        elif no_json_handling == 'empty':
            try:
                pil_im = open_image(image_fn_abs)
            except Exception:
                print('Warning: error opening image {}, skipping'.format(image_fn_abs))
                result['status'] = 'image load error'
                return result
            im['width'] = pil_im.width
            im['height'] = pil_im.height
            
            # Just in case we need to differentiate between "no .json file" and "a .json file with no annotations"
            im['no_labelme_json'] = True
            shapes = []
        else:
            raise ValueError('Unrecognized specifier {} for handling images with no .json files'.format(
                no_json_handling))            
    
    # If we found a .json file for this image...
    else:
        
        # Read the .json file
        with open(json_fn_abs,'r') as f:
            labelme_data = json.load(f)            
        im['width'] = labelme_data['imageWidth']
        im['height'] = labelme_data['imageHeight']
        
        if validate_image_sizes:
            try:
                pil_im = open_image(image_fn_abs)
            except Exception:
                print('Warning: error opening image {} for size validation, skipping'.format(image_fn_abs))
                result['status'] = 'skipped (size validation error)'
                return result
            if not (im['width'] == pil_im.width and im['height'] == pil_im.height):
                print('Warning: image size validation error for file {}'.format(image_fn_relative))
                im['width'] = pil_im.width
                im['height'] = pil_im.height
                im['labelme_width'] = labelme_data['imageWidth']
                im['labelme_height'] = labelme_data['imageHeight']

        shapes = labelme_data['shapes']

        if ('flags' in labelme_data) and (len(labelme_data['flags']) > 0):
            im['flags'] = labelme_data['flags']

    annotations_this_image = []
    
    if len(shapes) == 0:
        
        if allow_new_categories:
            category_id = _add_category('empty',category_name_to_id)
        else:
            assert 'empty' in category_name_to_id
            category_id = category_name_to_id['empty']
            
        ann = {}
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']
        ann['category_id'] = category_id
        ann['sequence_level_annotation'] = False
        annotations_this_image.append(ann)
        
    else:
        
        for shape in shapes:
            
            if shape['shape_type'] != 'rectangle':
                print('Only rectangles are supported, skipping an annotation of type {} in {}'.format(
                    shape['shape_type'],image_fn_relative))
                continue
            
            if use_folders_as_labels:
                category_name = os.path.basename(os.path.dirname(image_fn_abs))
            else:
                category_name = shape['label']                
            
            if allow_new_categories:
                category_id = _add_category(category_name,category_name_to_id)
            else:
                assert category_name in category_name_to_id
                category_id = category_name_to_id[category_name]
            
            points = shape['points']
            if len(points) != 2:
                print('Warning: illegal rectangle with {} points for {}'.format(
                    len(points),image_fn_relative))
                continue
            
            p0 = points[0]
            p1 = points[1]
            x0 = min(p0[0],p1[0])
            x1 = max(p0[0],p1[0])
            y0 = min(p0[1],p1[1])
            y1 = max(p0[1],p1[1])
            
            bbox = [x0,y0,abs(x1-x0),abs(y1-y0)]
            ann = {}
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']
            ann['category_id'] = category_id
            ann['sequence_level_annotation'] = False
            ann['bbox'] = bbox
            annotations_this_image.append(ann)
            
        # ...for each shape
        
    result['im'] = im
    result['annotations_this_image'] = annotations_this_image

    return result
    
# ...def _process_labelme_file(...)


#%% Main function

def labelme_to_coco(input_folder,
                    output_file=None,
                    category_id_to_category_name=None,
                    empty_category_name='empty',
                    empty_category_id=None,
                    info_struct=None,
                    relative_paths_to_include=None,
                    relative_paths_to_exclude=None,
                    use_folders_as_labels=False,
                    recursive=True,
                    no_json_handling='skip',
                    validate_image_sizes=True,
                    max_workers=1, 
                    use_threads=True):
    """
    Finds all images in [input_folder] that have corresponding .json files, and converts
    to a COCO .json file.
    
    Currently only supports bounding box annotations and image-level flags (i.e., does not
    support point or general polygon annotations).
    
    Labelme's image-level flags don't quite fit the COCO annotations format, so they are attached
    to image objects, rather than annotation objects.
    
    If output_file is None, just returns the resulting dict, does not write to file.    
    
    if use_folders_as_labels is False (default), the output labels come from the labelme
    .json files.  If use_folders_as_labels is True, the lowest-level folder name containing
    each .json file will determine the output label.  E.g., if use_folders_as_labels is True,
    and the folder contains:
        
    images/train/lion/image0001.json
    
    ...all boxes in image0001.json will be given the label "lion", regardless of the labels in the 
    file.  Empty images in the "lion" folder will still be given the label "empty" (or 
    [empty_category_name]).
    
    Args:
        input_folder (str): input folder to search for images and Labelme .json files
        output_file (str, optional): output file to which we should write COCO-formatted data; if None
            this function just returns the COCO-formatted dict
        category_id_to_category_name (dict, optional): dict mapping category IDs to category names;
            really used to map Labelme category names to COCO category IDs.  IDs will be auto-generated
            if this is None.
        empty_category_id (int, optional): category ID to use for the not-very-COCO-like "empty" category;
            also see the no_json_handling parameter.
        info_struct (dict, optional): dict to stash in the "info" field of the resulting COCO dict
        relative_paths_to_include (list, optional): allowlist of relative paths to include in the COCO
            dict; there's no reason to specify this along with relative_paths_to_exclude.
        relative_paths_to_exclude (list, optional): blocklist of relative paths to exclude from the COCO
            dict; there's no reason to specify this along with relative_paths_to_include.
        use_folders_as_labels (bool, optional): if this is True, class names will be pulled from folder names,
            useful if you have images like a/b/cat/image001.jpg, a/b/dog/image002.jpg, etc.
        recursive (bool, optional): whether to recurse into [input_folder]
        no_json_handling (str, optional): how to deal with image files that have no corresponding .json files, 
            can be:
                
                - 'skip': ignore image files with no corresponding .json files
                - 'empty': treat image files with no corresponding .json files as empty
                - 'error': throw an error when an image file has no corresponding .json file
        validate_image_sizes (bool, optional): whether to load images to verify that the sizes specified
            in the labelme files are correct
        max_workers (int, optional): number of workers to use for parallelization, set to <=1 to disable
            parallelization
        use_threads (bool, optional): whether to use threads (True) or processes (False) for parallelization,
            not relevant if max_workers <= 1
        
    Returns:
        dict: a COCO-formatted dictionary, identical to what's written to [output_file] if [output_file] is not None.
    """
    
    if max_workers > 1:
        assert category_id_to_category_name is not None, \
            'When parallelizing labelme --> COCO conversion, you must supply a category mapping'
            
    if category_id_to_category_name is None:
        category_name_to_id = {}
    else:
        category_name_to_id = {v: k for k, v in category_id_to_category_name.items()}
    for category_name in category_name_to_id:
        try:
            category_name_to_id[category_name] = int(category_name_to_id[category_name])
        except ValueError:
            raise ValueError('Category IDs must be ints or string-formatted ints')
    
    # If the user supplied an explicit empty category ID, and the empty category
    # name is already in category_name_to_id, make sure they match.
    if empty_category_id is not None:
        if empty_category_name in category_name_to_id:
            assert category_name_to_id[empty_category_name] == empty_category_id, \
                'Ambiguous empty category specification'
        if empty_category_id in category_id_to_category_name:
            assert category_id_to_category_name[empty_category_id] == empty_category_name, \
                'Ambiguous empty category specification'
    else:
        if empty_category_name in category_name_to_id:
            empty_category_id = category_name_to_id[empty_category_name]

    del category_id_to_category_name
            
    # Enumerate images
    print('Enumerating images in {}'.format(input_folder))    
    image_filenames_relative = path_utils.find_images(input_folder,recursive=recursive,
                                                      return_relative_paths=True,
                                                      convert_slashes=True)    
    
    # Remove any images we're supposed to skip
    if (relative_paths_to_include is not None) or (relative_paths_to_exclude is not None):
        image_filenames_relative_to_process = []
        for image_fn_relative in image_filenames_relative:
            if relative_paths_to_include is not None and image_fn_relative not in relative_paths_to_include:
                continue
            if relative_paths_to_exclude is not None and image_fn_relative in relative_paths_to_exclude:
                continue
            image_filenames_relative_to_process.append(image_fn_relative)
        print('Processing {} of {} images'.format(
            len(image_filenames_relative_to_process),
            len(image_filenames_relative)))
        image_filenames_relative = image_filenames_relative_to_process
    
    # If the user supplied a category ID to use for empty images...
    if empty_category_id is not None:
        try:
            empty_category_id = int(empty_category_id)
        except ValueError:
            raise ValueError('Category IDs must be ints or string-formatted ints')
        
    if empty_category_id is None:
        empty_category_id = _add_category(empty_category_name,category_name_to_id)
            
    if max_workers <= 1:
        
        image_results = []
        for image_fn_relative in tqdm(image_filenames_relative):
            
            result = _process_labelme_file(image_fn_relative,input_folder,use_folders_as_labels,
                                      no_json_handling,validate_image_sizes,
                                      category_name_to_id,allow_new_categories=True)        
            image_results.append(result)
            
    else:                      
        
        n_workers = min(max_workers,len(image_filenames_relative))
        assert category_name_to_id is not None
        
        if use_threads:
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)
        
        image_results = list(tqdm(pool.imap(
            partial(_process_labelme_file,
                input_folder=input_folder,
                use_folders_as_labels=use_folders_as_labels,
                no_json_handling=no_json_handling,
                validate_image_sizes=validate_image_sizes,
                category_name_to_id=category_name_to_id,
                allow_new_categories=False
                ),image_filenames_relative), total=len(image_filenames_relative)))
        
    images = []
    annotations = []
    
    # Flatten the lists of images and annotations
    for result in image_results:
        im = result['im']
        annotations_this_image = result['annotations_this_image']
        
        if im is None:
            assert annotations_this_image is None
        else:
            images.append(im)
            annotations.extend(annotations_this_image)
            
    output_dict = {}
    output_dict['images'] = images
    output_dict['annotations'] = annotations
    
    if info_struct is None:
        info_struct = {}
    if 'description' not in info_struct:
        info_struct['description'] = \
            'Converted to COCO from labelme annotations in folder {}'.format(input_folder)
    if 'version' not in info_struct:
        info_struct['version'] = 1.0
    
    output_dict['info'] = info_struct
    categories = []
    for category_name in category_name_to_id:
        categories.append({'name':category_name,'id':category_name_to_id[category_name]})
    output_dict['categories'] = categories
    
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(output_dict,f,indent=1)
    
    return output_dict

# ...def labelme_to_coco()


def find_empty_labelme_files(input_folder,recursive=True):
    """
    Returns a list of all image files in in [input_folder] associated with .json files that have 
    no boxes in them.  Also returns a list of images with no associated .json files.  Specifically,
    returns a dict:
    
    .. code-block: none
    
        {
            'images_with_empty_json_files':[list],
            'images_with_no_json_files':[list],
            'images_with_non_empty_json_files':[list]
        }
    
    Args:
        input_folder (str): the folder to search for empty (i.e., box-less) Labelme .json files
        recursive (bool, optional): whether to recurse into [input_folder]
    
    Returns:
        dict: a dict with fields:
            - images_with_empty_json_files: a list of all image files in [input_folder] associated with 
              .json files that have no boxes in them
            - images_with_no_json_files: a list of images in [input_folder] with no associated .json files
            - images_with_non_empty_json_files: a list of images in [input_folder] associated with .json
              files that have at least one box        
    """
    image_filenames_relative = path_utils.find_images(input_folder,recursive=True,
                                                      return_relative_paths=True)
    
    images_with_empty_json_files = []
    images_with_no_json_files = []
    images_with_non_empty_json_files = []
    
    # fn_relative = image_filenames_relative[0]
    for fn_relative in image_filenames_relative:
        
        image_fn_abs = os.path.join(input_folder,fn_relative)
        json_fn_abs = os.path.splitext(image_fn_abs)[0] + '.json'
        
        if not os.path.isfile(json_fn_abs):
            images_with_no_json_files.append(fn_relative)
            continue
        
        else:
           # Read the .json file
           with open(json_fn_abs,'r') as f:
               labelme_data = json.load(f) 
           shapes = labelme_data['shapes']
           if len(shapes) == 0:
               images_with_empty_json_files.append(fn_relative)
           else:
               images_with_non_empty_json_files.append(fn_relative)
               
    # ...for every image
    
    return {'images_with_empty_json_files':images_with_empty_json_files,
            'images_with_no_json_files':images_with_no_json_files,
            'images_with_non_empty_json_files':images_with_non_empty_json_files}

# ...def find_empty_labelme_files(...)


#%% Interactive driver

if False:
    
    pass

    #%% Options
    
    empty_category_name = 'empty'
    empty_category_id = None
    category_id_to_category_name = None
    info_struct = None
    
    input_folder = os.path.expanduser('~/data/md-test')
    output_file = os.path.expanduser('~/data/md-test-labelme-to-coco.json')
    
    
    #%% Programmatic execution
    
    output_dict = labelme_to_coco(input_folder,output_file,
                                  category_id_to_category_name=category_id_to_category_name,
                                  empty_category_name=empty_category_name,
                                  empty_category_id=empty_category_id,
                                  info_struct=None,
                                  use_folders_as_labels=False,
                                  validate_image_sizes=False,
                                  no_json_handling='empty')
    
    
    #%% Validate
    
    from megadetector.data_management.databases import integrity_check_json_db
    
    options = integrity_check_json_db.IntegrityCheckOptions()
        
    options.baseDir = input_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False
    
    sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(output_file,options)    
    

    #%% Preview
    
    from megadetector.visualization import visualize_db
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000

    html_file,_ = visualize_db.visualize_db(output_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                                input_folder,options)
    

    from megadetector.utils import path_utils # noqa
    path_utils.open_file(html_file)
    
    
    #%% Prepare command line

    s = 'python labelme_to_coco.py {} {}'.format(input_folder,output_file)
    print(s)
    import clipboard; clipboard.copy(s)

    
#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(
        description='Convert labelme-formatted data to COCO')
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to images and .json annotation files')
    
    parser.add_argument(
        'output_file',
        type=str,
        help='Output filename (.json)')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    labelme_to_coco(args.input_folder,args.output_file)
    
if __name__ == '__main__':
    main()
