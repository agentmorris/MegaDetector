"""

subset_json_db.py

Select a subset of images (and associated annotations) from a .json file in COCO
Camera Traps format based on a string query.

To subset .json files in the MegaDetector output format, see
subset_json_detector_output.py.

"""

#%% Constants and imports

import os
import sys
import json
import argparse

from tqdm import tqdm
from megadetector.utils import ct_utils
from copy import copy


#%% Functions

def subset_json_db(input_json, query, output_json=None, ignore_case=False, verbose=False):
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

    # Write the output file if requested
    if output_json is not None:
        if verbose:
            print('Writing output .json to {}'.format(output_json))
        output_dir = os.path.dirname(output_json)
        os.makedirs(output_dir,exist_ok=True)
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
