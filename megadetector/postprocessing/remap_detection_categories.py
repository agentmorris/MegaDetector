"""

remap_detection_categories.py

Given a MegaDetector results file, remap the category IDs according to a specified
dictionary, writing the results to a new file.

Currently only supports remapping detection categories, not classification categories.

"""

#%% Constants and imports

import os
import json

from tqdm import tqdm

from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import write_json


#%% Main function

def remap_detection_categories(input_file,
                               output_file,
                               target_category_map,
                               input_category_name_to_output_category_name,
                               overwrite=False,
                               invalid_category_handling='unknown'):
    """
    Given a MegaDetector results file [input_file], remap the category IDs according to the dictionary
    [target_category_map], writing the results to [output_file].  The remapped dictionary needs to have
    the same category names as the input file's detection_categories dictionary.

    Typically used to map, e.g., a variety of species to the class "mammal" or the class "animal".

    Currently only supports remapping detection categories, not classification categories.

    Args:
        input_file (str): the MD .json results file to remap
        output_file (str): the remapped .json file to write
        target_category_map (dict): the category mapping that should be used in the output file,
            mapping string-ints to class names. This can also be a MD results file, in which case
            we'll use that file's detection_categories dictionary.
        input_category_name_to_output_category_name (dict): str->str, the specific category mapping
            that should be used, otherwise will determine from target class names
        overwrite (bool, optional): whether to overwrite [output_file] if it exists; if this is True and
            [output_file] exists, this function is a no-op
        invalid_category_handling (str, optional): what to do about categories that are not in
            the input file's category list ('error' or 'unknown'), if 'uknown', creates a new
            "unknown" category

    """

    if os.path.exists(output_file) and (not overwrite):
        print('File {} exists, bypassing remapping'.format(output_file))
        return

    assert invalid_category_handling in ('error','unknown'), \
        'Invalid value for invalid_category_handling: {}'.format(invalid_category_handling)

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

    output_category_name_to_output_category_id = invert_dictionary(target_category_map)

    # Map input category IDs to output category IDs
    input_category_id_to_output_category_id = {}
    for input_category_id in input_categories.keys():
        input_category_name = input_categories[input_category_id]
        output_category_name = \
            input_category_name_to_output_category_name[input_category_name]
        output_category_id = \
            output_category_name_to_output_category_id[output_category_name]
        input_category_id_to_output_category_id[input_category_id] = output_category_id

    # im = input_images[0]
    for im in tqdm(input_images):

        if 'detections' not in im or im['detections'] is None:
            continue

        # det = im['detections'][0]
        for det in im['detections']:
            input_category_id = det['category']
            if input_category_id not in input_category_id_to_output_category_id:
                input_category_name = '[unknown]'
                if input_category_id in input_categories:
                    input_category_name = input_categories[input_category_id]
                s = 'Detection category {} ({}) not mapped'.format(
                    input_category_id,input_category_name)
                if invalid_category_handling == 'error':
                    raise ValueError(s)
                else:
                    assert invalid_category_handling == 'unknown'
                    print('Warning: {}'.format(s))
                    if 'unknown' in output_category_name_to_output_category_id:
                        output_category_id = output_category_name_to_output_category_id['unknown']
                    else:
                        category_id_values = [int(x) for x in output_category_name_to_output_category_id.values()]
                        unknown_category_id = max(category_id_values) + 1
                        output_category_name_to_output_category_id['unknown'] = unknown_category_id
                        output_category_id = unknown_category_id
            else:
                output_category_id = input_category_id_to_output_category_id[input_category_id]
            det['category'] = output_category_id

    # ...for each image

    input_data['detection_categories'] = target_category_map

    write_json(output_file,input_data)

    print('Saved remapped results to {}'.format(output_file))

# ...def remap_detection_categories(...)
