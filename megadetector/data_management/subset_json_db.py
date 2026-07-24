"""

subset_json_db.py

Select a subset of images (and associated annotations) from a .json file in COCO
Camera Traps format based on a string query.

To subset .json files in the MegaDetector output format, see
subset_json_detector_output.py.

"""

#%% Constants and imports

import sys
import json
import argparse

from tqdm import tqdm
from copy import copy

from megadetector.utils import ct_utils
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key


#%% Functions

def subset_json_db(input_json,
                   query,
                   output_json=None,
                   ignore_case=False,
                   remap_categories=True,
                   verbose=False):
    """
    Given a json file (or dictionary already loaded from a json file), produce a new
    database containing only the images whose filenames contain the string 'query',
    optionally writing that DB output to a new json file.

    Args:
        input_json (str): COCO Camera Traps .json file to load, or an already-loaded dict
        query (str or list): string to query for, only include images in the output whose filenames
            contain this string.  If this is a list, test for exact matches.
        output_json (str, optional): file to write the resulting .json file to
        ignore_case (bool, optional): whether to perform a case-insensitive search for [query]
        remap_categories (bool, optional): trim the category list to only the categores used
            in the subset
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: CCT dictionary containing a subset of the images and annotations in the input dict
    """

    # Load the input file if necessary
    if isinstance(input_json,str):
        print('Loading input .json...')
        with open(input_json, 'r') as f:
            input_data = json.load(f)
    else:
        input_data = input_json

    # Find images matching the query
    images = []

    if isinstance(query,str):

        if ignore_case:
            query = query.lower()

        for im in tqdm(input_data['images']):
            fn = im['file_name']
            if ignore_case:
                fn = fn.lower()
            if query in fn:
                images.append(im)

    else:

        query = set(query)

        if ignore_case:
            query = set([s.lower() for s in query])

        for im in input_data['images']:
            fn = im['file_name']
            if ignore_case:
                fn = fn.lower()
            if fn in query:
                images.append(im)

    image_ids = set([im['id'] for im in images])

    # Find annotations referring to those images
    annotations = []

    for ann in input_data['annotations']:
        if ann['image_id'] in image_ids:
            annotations.append(ann)

    output_data = copy(input_data)
    output_data['images'] = images
    output_data['annotations'] = annotations

    # Remap categories if necessary
    if remap_categories:

        category_ids_used = set()
        for ann in annotations:
            category_ids_used.add(ann['category_id'])

        if verbose:
            print('Keeping {} of {} categories'.format(
                len(category_ids_used),len(input_data['categories'])))

        input_category_id_to_output_category_id = {}

        next_category_id = 0

        # Build mappings from old to new category IDs
        for input_category_id in category_ids_used:
            assert isinstance(input_category_id,int), \
                'Illegal category ID {}'.format(input_category_id)
            output_category_id = next_category_id
            next_category_id = next_category_id + 1
            input_category_id_to_output_category_id[input_category_id] = output_category_id

        # Modify the annotations
        for ann in annotations:
            assert ann['category_id'] in input_category_id_to_output_category_id
            ann['category_id'] = input_category_id_to_output_category_id[ann['category_id']]

        output_categories = []

        # Re-write the category table
        for cat in input_data['categories']:

            if cat['id'] in input_category_id_to_output_category_id:

                # There may be non-required fields, so don't just create an empty dict
                # and copy the name/id field, keep the original dict other than "id"
                output_category = copy(cat)
                output_category['id'] = input_category_id_to_output_category_id[cat['id']]
                output_categories.append(output_category)

        output_categories = sort_list_of_dicts_by_key(output_categories,'id')
        output_data['categories'] = output_categories

    # ...if we need to remap categories

    # Write the output file if requested
    if output_json is not None:
        if verbose:
            print('Writing output .json to {}'.format(output_json))
        ct_utils.write_json(output_json, output_data)

    if verbose:
        print('Keeping {} of {} images, {} of {} annotations'.format(
            len(output_data['images']),len(input_data['images']),
            len(output_data['annotations']),len(input_data['annotations'])))

    return output_data


#%% Interactive driver

if False:

    #%%

    input_json = r"e:\Statewide_wolf_container\idfg_20190409.json"
    output_json = r"e:\Statewide_wolf_container\idfg_20190409_clearcreek.json"
    query = 'clearcreek'
    ignore_case = True
    db = subset_json_db(input_json, query, output_json, ignore_case)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str, help='Input file (a COCO Camera Traps .json file)')
    parser.add_argument('output_json', type=str, help='Output file')
    parser.add_argument('query', type=str, help='Filename query')
    parser.add_argument('--ignore_case', action='store_true')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    subset_json_db(args.input_json,args.query,args.output_json,args.ignore_case)

if __name__ == '__main__':
    main()
