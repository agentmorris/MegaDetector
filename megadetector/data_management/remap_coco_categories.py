"""

remap_coco_categories.py

Given a COCO-formatted dataset, remap the categories to a new mapping.  A common use
case is to take a fine-grained dataset (e.g. with species-level categories) and
map them to coarse categories (typically MD categories).

"""

#%% Imports and constants

import os
import json
import argparse

from copy import deepcopy
from megadetector.utils.ct_utils import invert_dictionary


#%% Main function

def remap_coco_categories(input_data,
                          output_category_name_to_id,
                          input_category_name_to_output_category_name,
                          output_file=None,
                          allow_unused_categories=False):
    """
    Given a COCO-formatted dataset, remap the categories to a new categories mapping, optionally
    writing the results to a new file.

    Args:
        input_data (str or dict): a COCO-formatted dict or a filename.  If it's a dict, it will
            be copied, not modified in place.
        output_category_name_to_id (dict): a dict mapping strings to ints.  Categories not in
            this dict will be ignored or will result in errors, depending on allow_unused_categories.
        input_category_name_to_output_category_name (dict): a dict mapping strings to strings.
            Annotations using categories not in this dict will be omitted or will result in
            errors, depending on allow_unused_categories.
        output_file (str, optional): output file to which we should write remapped COCO data
        allow_unused_categories (bool, optional): should we ignore categories not present in the
            input/output mappings?  If this is False and we encounter an unmapped category, we'll
            error.

    Returns:
        dict: COCO-formatted dict
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
    input_category_id_to_input_category_name = \
        invert_dictionary(input_category_name_to_input_category_id)

    # Map input IDs --> output IDs
    input_category_id_to_output_category_id = {}
    input_category_names = list(input_category_name_to_output_category_name.keys())

    # input_name = input_category_names[0]
    for input_name in input_category_names:

        output_name = input_category_name_to_output_category_name[input_name]
        assert output_name in output_category_name_to_id, \
            'No output ID for {} --> {}'.format(input_name,output_name)
        input_id = input_category_name_to_input_category_id[input_name]
        output_id = output_category_name_to_id[output_name]
        input_category_id_to_output_category_id[input_id] = output_id

    # ...for each category we want to keep

    printed_unused_category_warnings = set()

    valid_annotations = []

    # Map annotations
    for ann in output_data['annotations']:

        input_category_id = ann['category_id']
        if input_category_id not in input_category_id_to_output_category_id:
            if allow_unused_categories:
                if input_category_id not in printed_unused_category_warnings:
                    printed_unused_category_warnings.add(input_category_id)
                    input_category_name = \
                        input_category_id_to_input_category_name[input_category_id]
                    s = 'Skipping unmapped category ID {} ({})'.format(
                        input_category_id,input_category_name)
                    print(s)
                continue
            else:
                s = 'Unmapped category ID {}'.format(input_category_id)
                raise ValueError(s)
        output_category_id = input_category_id_to_output_category_id[input_category_id]
        ann['category_id'] = output_category_id
        valid_annotations.append(ann)

    # ...for each annotation

    # The only reason annotations should get excluded is the case where we allow
    # unused categories
    if not allow_unused_categories:
        assert len(valid_annotations) == len(output_data['annotations'])

    output_data['annotations'] = valid_annotations

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

# ...def remap_coco_categories(...)


#%% Command-line driver

def main():
    """
    Command-line interface to remap COCO categories.
    """

    parser = argparse.ArgumentParser(
        description='Remap categories in a COCO-formatted dataset'
    )
    parser.add_argument(
        'input_coco_file',
        type=str,
        help='Path to the input COCO .json file'
    )
    parser.add_argument(
        'output_category_map_file',
        type=str,
        help="Path to a .json file mapping output category names to integer IDs (e.g., {'cat':0, 'dog':1})"
    )
    parser.add_argument(
        'input_to_output_category_map_file',
        type=str,
        help="Path to a .json file mapping input category names to output category names" + \
            " (e.g., {'old_cat_name':'cat', 'old_dog_name':'dog'})"
    )
    parser.add_argument(
        'output_coco_file',
        type=str,
        help='Path to save the remapped COCO .json file'
    )
    parser.add_argument(
        '--allow_unused_categories',
        action='store_true',
        help='Allow unmapped categories (by default, errors on unmapped categories)'
    )

    args = parser.parse_args()

    # Load category mappings
    with open(args.output_category_map_file, 'r') as f:
        output_category_name_to_id = json.load(f)

    with open(args.input_to_output_category_map_file, 'r') as f:
        input_category_name_to_output_category_name = json.load(f)

    # Load COCO data
    with open(args.input_coco_file, 'r') as f:
        input_data = json.load(f)

    remap_coco_categories(
        input_data=input_data,
        output_category_name_to_id=output_category_name_to_id,
        input_category_name_to_output_category_name=input_category_name_to_output_category_name,
        output_file=args.output_coco_file,
        allow_unused_categories=args.allow_unused_categories
    )

    print(f'Successfully remapped categories and saved to {args.output_coco_file}')

if __name__ == '__main__':
    main()
