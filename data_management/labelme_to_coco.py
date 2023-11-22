########
#
# labelme_to_coco.py
#
# Converts a folder of labelme-formatted .json files to COCO.
#
########

#%% Constants and imports

import json
import os
import uuid

from md_utils import path_utils
from md_visualization.visualization_utils import open_image

from tqdm import tqdm


#%% Functions

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
                    right_edge_quantization_threshold=None):
    """
    Find all images in [input_folder] that have corresponding .json files, and convert
    to a COCO .json file.
    
    Currently only supports bounding box annotations.
    
    If output_file is None, just returns the resulting dict, does not write to file.    
    
    if use_folders_as_labels is False (default), the output labels come from the labelme
    .json files.  If use_folders_as_labels is True, the lowest-level folder name containing
    each .json file will determine the output label.  E.g., if use_folders_as_labels is True,
    and the folder contains:
        
    images/train/lion/image0001.json
    
    ...all boxes in image0001.json will be given the label "lion", regardless of the labels in the 
    file.  Empty images in the "lion" folder will still be given the label "empty" (or 
    [empty_category_name]).
    
    no_json_handling can be:
        
    * 'skip': ignore image files with no corresponding .json files
    * 'empty': treat image files with no corresponding .json files as empty
    * 'error': throw an error when an image file has no corresponding .json file
    
    right_edge_quantization_threshold is an off-by-default hack to handle cases where 
    boxes that really should be running off the right side of the image only extend like 99%
    of the way there, due to what appears to be a slight bias inherent to MD.  If a box extends
    within [right_edge_quantization_threshold] (a small number, from 0 to 1, but probably around 
    0.02) of the right edge of the image, it will be extended to the far right edge.    
    """
    
    if category_id_to_category_name is None:
        category_name_to_id = {}
    else:
        category_name_to_id = {v: k for k, v in category_id_to_category_name.items()}
        
    for category_name in category_name_to_id:
        try:
            category_name_to_id[category_name] = int(category_name_to_id[category_name])
        except ValueError:
            raise ValueError('Category IDs must be ints or string-formatted ints')

    # Enumerate images
    image_filenames_relative = path_utils.find_images(input_folder,recursive=recursive,
                                                      return_relative_paths=True)    
    
    def add_category(category_name,candidate_category_id=0):
        if category_name in category_name_to_id:
            return category_name_to_id[category_name]
        while candidate_category_id in category_name_to_id.values():
            candidate_category_id += 1
        category_name_to_id[category_name] = candidate_category_id
        return candidate_category_id
    
    if empty_category_id is not None:
        try:
            empty_category_id = int(empty_category_id)
        except ValueError:
            raise ValueError('Category IDs must be ints or string-formatted ints')
        
    if empty_category_id is None:
        empty_category_id = add_category(empty_category_name)
        
    images = []
    annotations = []
    
    n_edges_quantized = 0
    
    # image_fn_relative = image_filenames_relative[0]
    for image_fn_relative in tqdm(image_filenames_relative):
        
        if relative_paths_to_include is not None and image_fn_relative not in relative_paths_to_include:
            continue
        if relative_paths_to_exclude is not None and image_fn_relative in relative_paths_to_exclude:
            continue
        
        image_fn_abs = os.path.join(input_folder,image_fn_relative)
        json_fn_abs = os.path.splitext(image_fn_abs)[0] + '.json'
        
        im = {}
        im['id'] = image_fn_relative
        im['file_name'] = image_fn_relative
        
        # If there's no .json file for this image...
        if not os.path.isfile(json_fn_abs):
            
            # Either skip it...
            if no_json_handling == 'skip':
                continue
            
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
                    continue
                im['width'] = pil_im.width
                im['height'] = pil_im.height
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
                    print('Warning: error opening image {}, skipping'.format(image_fn_abs))
                    continue                
                assert im['width'] == pil_im.width and im['height'] == pil_im.height, \
                    'Image size validation error for file {}'.format(image_fn_relative)                
                
            shapes = labelme_data['shapes']
        
        if len(shapes) == 0:
            
            category_id = add_category('empty')
            ann = {}
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']
            ann['category_id'] = category_id
            ann['sequence_level_annotation'] = False
            annotations.append(ann)
            
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
                
                category_id = add_category(category_name)
            
                points = shape['points']
                assert len(points) == 2, 'Illegal rectangle with {} points'.format(
                    len(points))
                
                p0 = points[0]
                p1 = points[1]
                x0 = min(p0[0],p1[0])
                x1 = max(p0[0],p1[0])
                y0 = min(p0[1],p1[1])
                y1 = max(p0[1],p1[1])
                
                if right_edge_quantization_threshold is not None:                    
                    x1_rel = x1 / (im['width'] - 1)
                    right_edge_distance = 1.0 - x1_rel
                    if right_edge_distance < right_edge_quantization_threshold:
                        n_edges_quantized += 1
                        x1 = im['width'] - 1
                        
                bbox = [x0,y0,abs(x1-x0),abs(y1-y0)]
                ann = {}
                ann['id'] = str(uuid.uuid1())
                ann['image_id'] = im['id']
                ann['category_id'] = category_id
                ann['sequence_level_annotation'] = False
                ann['bbox'] = bbox
                annotations.append(ann)
                
            # ...for each shape
            
        images.append(im)
                  
    # ..for each image                
    
    if n_edges_quantized > 0:
        print('Quantized the right edge in {} of {} images'.format(
            n_edges_quantized,len(image_filenames_relative)))
        
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
        
    {
       'images_with_empty_json_files':[list],
       'images_with_no_json_files':[list],
       'images_with_non_empty_json_files':[list]
    }    
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
    
    from data_management.databases import integrity_check_json_db
    
    options = integrity_check_json_db.IntegrityCheckOptions()
        
    options.baseDir = input_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False
    
    sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(output_file,options)    
    

    #%% Preview
    
    from md_visualization import visualize_db
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000

    html_file,_ = visualize_db.visualize_db(output_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                                input_folder,options)
    

    from md_utils import path_utils # noqa
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
