"""

remap_detection_categories.py

Given a MegaDetector results file, remap the category IDs according to a specified 
dictionary, writing the results to a new file.

Currently only supports remapping detection categories, not classification categories.

"""

#%% Constants and imports

import json
import os

from tqdm import tqdm

from megadetector.utils.ct_utils import invert_dictionary


#%% Main function

def remap_detection_categories(input_file,
                               output_file,
                               target_category_map,
                               extra_category_handling='error',
                               overwrite=False):
    """
    Given a MegaDetector results file [input_file], remap the category IDs according to the dictionary
    [target_category_map], writing the results to [output_file].  The remapped dictionary needs to have 
    the same category names as the input file's detection_categories dictionary.
    
    Typically used to map, e.g., a variety of species to the class "mammal" or the class "animal".
    
    Currently only supports remapping detection categories, not classification categories.
    
    Args:
        input_file (str): the MD .json results file to remap
        output_file (str): the remapped .json file to write
        target_category_map (dict): the category mapping that should be used in the output file.
            This can also be a MD results file, in which case we'll use that file's
            detection_categories dictionary.
        extra_category_handling (str, optional): specifies what we should do if categories are present
            in the source file that are not present in the target mapping:
            
            * 'error' == Error in this case.
            * 'drop_if_unused' == Don't include these in the output file's category mappings if they are 
              unused, error if they are.
            * 'remap' == Remap to unused category IDs.  This is reserved for future use, not currently
              implemented.
        overwrite (bool, optional): whether to overwrite [output_file] if it exists; if this is True and
            [output_file] exists, this function is a no-op
               
    """
    
    if os.path.exists(output_file) and (not overwrite):
        print('File {} exists, bypassing remapping'.format(output_file))        
        return
    
    assert os.path.isfile(input_file), \
        'File {} does not exist'.format(input_file)

    # If "target_category_map" is passed as a filename, load the "detection_categories"
    # dict.
    if isinstance(target_category_map,str):
        target_categories_file = target_category_map
        with open(target_categories_file,'r') as f:
            d = json.load(f)
            target_category_map = d['detection_categories']
    assert isinstance(target_category_map,dict)
    
    with open(input_file,'r') as f:
        input_data = json.load(f)
    
    input_images = input_data['images']
    input_categories = input_data['detection_categories']
    
    # Figure out which categories are actually used
    used_category_ids = set()
    for im in input_images:
        
        if 'detections' not in im or im['detections'] is None:
            continue
        
        for det in im['detections']:
            used_category_ids.add(det['category'])
    used_category_names = [input_categories[cid] for cid in used_category_ids]
        
    input_names_set = set(input_categories.values())
    output_names_set = set(target_category_map.values())
    
    # category_name = list(input_names_set)[0]
    for category_name in input_names_set:
        if category_name in output_names_set:
            continue
        if extra_category_handling == 'error':
            raise ValueError('Category {} present in source but not in target'.format(category_name))
        elif extra_category_handling == 'drop_if_unused':
            if category_name in used_category_names:
                raise ValueError('Category {} present (and used) in source but not in target'.format(
                    category_name))
            else:
                print('Category {} is unused and not present in the target mapping, ignoring'.format(
                    category_name))
                continue
        elif extra_category_handling == 'remap':
            raise NotImplementedError('Remapping of extra category IDs not yet implemented')
        else:
            raise ValueError('Unrecognized extra category handling scheme {}'.format(
                extra_category_handling))
    
    output_category_name_to_output_category_id = invert_dictionary(target_category_map)
    
    input_category_id_to_output_category_id = {}
    for input_category_id in input_categories.keys():
        category_name = input_categories[input_category_id]
        if category_name not in output_category_name_to_output_category_id:
            assert category_name not in used_category_names
        else:
            output_category_id = output_category_name_to_output_category_id[category_name]
            input_category_id_to_output_category_id[input_category_id] = output_category_id
        
    # im = input_images[0]
    for im in tqdm(input_images):
        
        if 'detections' not in im or im['detections'] is None:
            continue
        
        # det = im['detections'][0]
        for det in im['detections']:
            det['category'] = input_category_id_to_output_category_id[det['category']]
    
    input_data['detection_categories'] = target_category_map
    
    with open(output_file,'w') as f:
        json.dump(input_data,f,indent=1)
    
    
    print('Saved remapped results to {}'.format(output_file))
    

#%% Interactive driver

if False:
    
    pass

    #%% 
        
    target_categories_file = '/home/dmorris/tmp/usgs-tegus/model-comparison/all-classes_usgs-only_yolov5x6.json'
    target_category_map = target_categories_file
    input_file = '/home/dmorris/tmp/usgs-tegus/model-comparison/all-classes_usgs-goannas-lilablanks_yolov5x6-20240223.json'    

    output_file = input_file.replace('.json','_remapped.json')
    assert output_file != input_file
    overwrite = True
    
    extra_category_handling = 'drop_if_unused'
    
    remap_detection_categories(input_file=input_file,
                               output_file=output_file,
                               target_category_map=target_category_map,
                               extra_category_handling=extra_category_handling,                               
                               overwrite=overwrite)


#%% Command-line driver

# TODO
