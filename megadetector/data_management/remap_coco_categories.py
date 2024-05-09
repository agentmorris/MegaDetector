"""

remap_coco_categories.py

Given a COCO-formatted dataset, remap the categories to a new mapping.

"""

#%% Imports and constants

import os
import json

from copy import deepcopy


#%% Main function

def remap_coco_categories(input_data,
                          output_category_name_to_id,
                          input_category_name_to_output_category_name,
                          output_file=None):
    """
    Given a COCO-formatted dataset, remap the categories to a new categories mapping, optionally
    writing the results to a new file.
    
    output_category_name_to_id is a dict mapping strings to ints.
    
    input_category_name_to_output_category_name is a dict mapping strings to strings.
    
    [input_data] can be a COCO-formatted dict or a filename.  If it's a dict, it will be copied,
    not modified in place.
    """
    
    if isinstance(input_data,str):
        assert os.path.isfile(input_data), "Can't find file {}".format(input_data)
        with open(input_data,'r') as f:
            input_data = json.load(f)
        assert isinstance(input_data,dict), 'Illegal COCO input data'
    else:
        assert isinstance(input_data,dict), 'Illegal COCO input data'
        input_data = deepcopy(input_data)
    
    # It's safe to modify in-place now
    output_data = input_data
    
    # Read input name --> ID mapping
    input_category_name_to_input_category_id = {}
    for c in input_data['categories']:
        input_category_name_to_input_category_id[c['name']] = c['id']
    
    # Map input IDs --> output IDs
    input_category_id_to_output_category_id = {}
    for input_name in input_category_name_to_output_category_name.keys():
        output_name = input_category_name_to_output_category_name[input_name]
        assert output_name in output_category_name_to_id, \
            'No output ID for {} --> {}'.format(input_name,output_name)
        input_id = input_category_name_to_input_category_id[input_name]
        output_id = output_category_name_to_id[output_name]
        input_category_id_to_output_category_id[input_id] = output_id
        
    # Map annotations
    for ann in output_data['annotations']:
        assert ann['category_id'] in input_category_id_to_output_category_id, \
            'Unrecognized category ID {}'.format(ann['category_id'])
        ann['category_id'] = input_category_id_to_output_category_id[ann['category_id']]
        
    # Update the category list
    output_categories = []
    for output_name in output_category_name_to_id:
        category = {'name':output_name,'id':output_category_name_to_id[output_name]}
        output_categories.append(category)
    output_data['categories'] = output_categories
        
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(output_data,f,indent=1)
            
    return input_data


#%% Command-line driver

# TODO
