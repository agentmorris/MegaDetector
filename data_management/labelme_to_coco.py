########
#
# labelme_to_coco.py
#
# Converts a folder of labelme-formatted .json files to COCO.  Typically used after
# filtering MegaDetector boxes in labelme, then moving to COCO for preview/training/etc.
#
# Adds an "empty" annotation (with no boxes) for every image that has a .json file but
# no shapes.
#
########

#%% Constants and imports

from md_utils import path_utils
import json
import os
import uuid


#%% Functions

def labelme_to_coco(input_folder,output_file=None,category_id_to_category_name=None,
                    empty_category_name='empty',empty_category_id=None,info_struct=None,
                    relative_paths_to_include=None,relative_paths_to_exclude=None):
    """
    Find all images in [input_folder] that have corresponding .json files, and convert
    to a COCO .json file.  Ignores images without corresponding .json files; empty images
    need to be accompanied by a .json file with an empty "shapes" field.
    
    Currently only supports bounding box annotations.
    
    If output_file is None, just returns the resulting dict, does not write to file.
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
    image_filenames_relative = path_utils.find_images(input_folder,recursive=True,
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
    
    # image_fn_relative = image_filenames_relative[0]
    for image_fn_relative in image_filenames_relative:
        
        if relative_paths_to_include is not None and image_fn_relative not in relative_paths_to_include:
            continue
        if relative_paths_to_exclude is not None and image_fn_relative in relative_paths_to_exclude:
            continue
        
        image_fn_abs = os.path.join(input_folder,image_fn_relative)
        json_fn_abs = os.path.splitext(image_fn_abs)[0] + '.json'
        if not os.path.isfile(json_fn_abs):
            continue
        
        # Read the .json file
        with open(json_fn_abs,'r') as f:
            labelme_data = json.load(f)
            
        im = {}
        im['width'] = labelme_data['imageWidth']
        im['height'] = labelme_data['imageHeight']
        im['id'] = image_fn_relative
        im['file_name'] = image_fn_relative
        
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


#%% Interactive driver

if False:
    
    pass

    #%% Convert labelme to json
    
    import os, json, uuid # noqa
    from md_utils import path_utils # noqa
    
    from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP
    empty_category_name = 'empty'
    empty_category_id = None
    info_struct = None
    
    category_id_to_category_name = DEFAULT_DETECTOR_LABEL_MAP
    # input_folder = os.path.expanduser('~/data/usgs-kissel-training/train/tegu')
    input_folder = os.path.expanduser('~/data/labelme-json-test')
    output_file = os.path.expanduser('~/tmp/labelme_to_coco_test.json')
    output_dict = labelme_to_coco(input_folder,output_file,
                                  category_id_to_category_name=category_id_to_category_name,
                                  empty_category_name='empty',
                                  empty_category_id=empty_category_id,
                                  info_struct=info_struct)
    
    
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
    options.parallelize_rendering = False

    html_file,_ = visualize_db.process_images(output_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                                input_folder,options)
    

    from md_utils import path_utils # noqa
    path_utils.open_file(html_file)
    
    
#%% Command-line driver

