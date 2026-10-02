"""

combine_batch_outputs.py

Merges two or more .json files in MD output format, optionally
writing the results to another .json file.

* Concatenates image lists, optionally erroring if images are not unique.
* Errors if detection class lists are not compatible (does not try to remap detection categories).
* Merges classification categories by name.  Files with and without classification
  categories can be merged; an empty classification category dict is treated the same as
  no classification categories.
* For "info" and any non-standard fields, arbitrarily uses the value from the first dict.

File format:

https://lila.science/megadetector-output-format

Command-line use:

combine_batch_outputs input1.json input2.json ... inputN.json output.json

This script does not merge detections within an image; if you are looking to ensemble the results
of multiple model versions, see merge_detections.py.

"""

#%% Constants and imports

import argparse
import sys
import json
import copy

from megadetector.utils.ct_utils import write_json
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.postprocessing.classification_postprocessing import merge_classification_categories


#%% Merge functions

def combine_batch_output_files(input_files,
                               output_file=None,
                               require_uniqueness=True,
                               verbose=True):
    """
    Merges the list of MD results files [input_files] into a single
    dictionary, optionally writing the result to [output_file].

    Always overwrites [output_file] if it exists.

    Args:
        input_files (list of str): paths to JSON detection files
        output_file (str, optional): path to write merged JSON
        require_uniqueness (bool, optional): whether to require that the images in
            each list of images be unique
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: merged dictionaries loaded from [input_files], identical to what's
        written to [output_file] if [output_file] is not None
    """

    def print_if_verbose(s):
        if verbose:
            print(s)

    input_dicts = []
    for fn in input_files:
        print_if_verbose('Loading results from {}'.format(fn))
        with open(fn, 'r', encoding='utf-8') as f:
            input_dicts.append(json.load(f))

    print_if_verbose('Merging results')
    merged_dict = combine_batch_output_dictionaries(
        input_dicts, require_uniqueness=require_uniqueness)

    if output_file is not None:
        print_if_verbose('Writing output to {}'.format(output_file))
        write_json(output_file, merged_dict)

    return merged_dict

# ...def combine_batch_output_files(...)


def combine_batch_output_dictionaries(input_dicts, require_uniqueness=True):
    """
    Merges the list of MD results dictionaries [input_dicts] into a single dict.
    See module header comment for details on merge rules.

    Args:
        input_dicts (list of dicts): list of dicts in which each dict represents the
            contents of a MD output file
        require_uniqueness (bool, optional): whether to require that the images in
            each input dict be unique; if this is True and image filenames are
            not unique, an error is raised.

    Returns:
        dict: merged MD results
    """

    classification_fields = ['classification_categories',
                             'classification_category_descriptions']

    known_fields = ['info',
                    'detection_categories',
                    'images'] + classification_fields

    def _has_classification_categories(d):
        # An empty classification category dict is equivalent to no classification categories
        return ('classification_categories' in d) and (len(d['classification_categories']) > 0)

    output_dict = None

    # Map image filenames to image dicts, we'll convert to a list later
    images = {}

    n_redundant_images = 0
    n_images = 0

    for input_dict in input_dicts:

        # Initialize the output to the first input dict
        if output_dict is None:

            output_dict = {}
            for k in input_dict:
                if k == 'images':
                    continue
                if (k in classification_fields) and (not _has_classification_categories(input_dict)):
                    continue
                output_dict[k] = copy.deepcopy(input_dict[k])

            input_images = copy.deepcopy(input_dict['images'])

        else:

            # For fields we don't know how to handle, use the first occurrence
            for k in input_dict:
                if k in known_fields:
                    continue
                if k in output_dict:
                    print('Warning: not merging unrecognized field {}'.format(k))
                else:
                    output_dict[k] = copy.deepcopy(input_dict[k])

            # Check compatibility of detection categories
            for cat_id in input_dict['detection_categories']:
                cat_name = input_dict['detection_categories'][cat_id]
                if cat_id in output_dict['detection_categories']:
                    assert output_dict['detection_categories'][cat_id] == cat_name, \
                        'Detection category mismatch'
                else:
                    output_dict['detection_categories'][cat_id] = cat_name

            # If both the output and this input have classification categories, map this input's
            # classification categories into the output category space
            if _has_classification_categories(input_dict) and \
               _has_classification_categories(output_dict):

                # merge_classification_categories only needs the category fields from the
                # target dict, so don't pass (and make it copy) the accumulated image list
                target_dict = {'images': []}
                for k in classification_fields:
                    if k in output_dict:
                        target_dict[k] = output_dict[k]

                # This returns a copy of [input_dict] in which classification_categories (and
                # classification_category_descriptions, if present in either dict) are a superset
                # of the output dict's categories, and images refer to those categories.
                merged_input_dict = merge_classification_categories(
                    target_file=target_dict,
                    source_file=input_dict,
                    output_file=None,
                    verbose=False)

                for k in classification_fields:
                    if k in merged_input_dict:
                        output_dict[k] = merged_input_dict[k]

                input_images = merged_input_dict['images']

            else:

                # If only this input has classification categories, the output inherits them
                # as-is; if only the output has classification categories, there is nothing
                # to remap.
                if _has_classification_categories(input_dict):
                    for k in classification_fields:
                        if k in input_dict:
                            output_dict[k] = copy.deepcopy(input_dict[k])

                input_images = copy.deepcopy(input_dict['images'])

            # ...if we do/don't need to merge classification categories

        # ...if this is/isn't the first input dict

        # Merge image lists, checking uniqueness
        for im in input_images:

            # Always use forward slashes in output
            im['file'] = im['file'].replace('\\','/')
            im_file = im['file']
            if require_uniqueness:
                assert im_file not in images, 'Duplicate image: {}'.format(im_file)
                images[im_file] = im
                n_images += 1
            else:
                if im_file in images:
                    n_redundant_images += 1
                    previous_im = images[im_file]
                    # Replace a previous failure with a success, otherwise keep the original
                    # record.  Failed images have "detections" set to None or omitted.
                    im_succeeded = (im.get('detections') is not None)
                    previous_im_succeeded = (previous_im.get('detections') is not None)
                    if im_succeeded and (not previous_im_succeeded):
                        images[im_file] = im
                        print('Replacing previous failure for image: {}'.format(im_file))
                else:
                    images[im_file] = im
                    n_images += 1

        # ...for each image

    # ...for each dictionary

    if (n_redundant_images > 0):
        print('Warning: found {} redundant images (out of {} total) during merge'.format(
            n_redundant_images,n_images))

    # Convert merged image dictionaries to a sorted list
    output_dict['images'] = sort_list_of_dicts_by_key(list(images.values()),'file')

    return output_dict

# ...def combine_batch_output_dictionaries(...)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_paths', nargs='+',
        help='List of input .json files')
    parser.add_argument(
        'output_path',
        help='Output .json file')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    combine_batch_output_files(args.input_paths, args.output_path)

if __name__ == '__main__':
    main()
