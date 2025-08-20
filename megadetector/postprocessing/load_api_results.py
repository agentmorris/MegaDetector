"""

load_api_results.py

DEPRECATED

As of 2023.12, this module is still used in postprocessing and RDE, but it's not recommended
for new code.

Loads the output of the batch processing API (json) into a Pandas dataframe.

Includes functions to read/write the (very very old) .csv results format.

"""

#%% Imports

import json
import os

from typing import Optional
from collections.abc import Mapping

import pandas as pd

from megadetector.utils import ct_utils
from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file


#%% Functions for loading .json results into a Pandas DataFrame, and writing back to .json

def load_api_results(api_output_path: str, normalize_paths: bool = True,
                     filename_replacements: Optional[Mapping[str, str]] = None,
                     force_forward_slashes: bool = True
                     ) -> tuple[pd.DataFrame, dict]:
    r"""
    Loads json-formatted MegaDetector results to a Pandas DataFrame.

    Args:
        api_output_path (str): path to the output json file
        normalize_paths (bool, optional): whether to apply os.path.normpath to the 'file'
            field in each image entry in the output file
        filename_replacements (dict, optional): replace some path tokens to match local paths
            to the original file structure
        force_forward_slashes (bool, optional): whether to convert backslashes to forward
            slashes in filenames

    Returns:
        detection_results: pd.DataFrame, contains at least the columns ['file', 'detections','failure']
        other_fields: a dict containing fields in the results other than 'images'
    """

    print('Loading results from {}'.format(api_output_path))

    detection_results = load_md_or_speciesnet_file(api_output_path)

    # Validate that this is really a detector output file
    for s in ['info', 'detection_categories', 'images']:
        assert s in detection_results, 'Missing field {} in detection results'.format(s)

    # Fields in the output json other than 'images'
    other_fields = {}
    for k, v in detection_results.items():
        if k != 'images':
            other_fields[k] = v

    if normalize_paths:
        for image in detection_results['images']:
            image['file'] = os.path.normpath(image['file'])

    if force_forward_slashes:
        for image in detection_results['images']:
            image['file'] = image['file'].replace('\\','/')

    # Replace some path tokens to match local paths to original blob structure
    if filename_replacements is not None:
        for string_to_replace in filename_replacements.keys():
            replacement_string = filename_replacements[string_to_replace]
            for im in detection_results['images']:
                im['file'] = im['file'].replace(string_to_replace,replacement_string)

    print('Converting results to dataframe')

    # If this is a newer file that doesn't include maximum detection confidence values,
    # add them, because our unofficial internal dataframe format includes this.
    for im in detection_results['images']:
        if 'max_detection_conf' not in im:
            im['max_detection_conf'] = ct_utils.get_max_conf(im)

    # Pack the json output into a Pandas DataFrame
    detection_results = pd.DataFrame(detection_results['images'])

    print('Finished loading MegaDetector results for {} images from {}'.format(
            len(detection_results),api_output_path))

    return detection_results, other_fields


def write_api_results(detection_results_table, other_fields, out_path):
    """
    Writes a Pandas DataFrame to the MegaDetector .json format.

    Args:
        detection_results_table (DataFrame): data to write
        other_fields (dict): additional fields to include in the output .json
        out_path (str): output .json filename
    """

    print('Writing detection results to {}'.format(out_path))

    fields = other_fields

    images = detection_results_table.to_json(orient='records',
                                             double_precision=3)
    images = json.loads(images)
    for im in images:
        if 'failure' in im and im['failure'] is None:
            del im['failure']
    fields['images'] = images

    # Convert the 'version' field back to a string as per format convention
    try:
        version = other_fields['info']['format_version']
        if not isinstance(version,str):
            other_fields['info']['format_version'] = str(version)
    except Exception:
        print('Warning: error determining format version')
        pass

    # Remove 'max_detection_conf' as per newer file convention (format >= v1.3)
    try:
        version = other_fields['info']['format_version']
        version = float(version)
        if version >= 1.3:
            for im in images:
                if 'max_detection_conf' in im:
                    del im['max_detection_conf']
    except Exception:
        print('Warning: error removing max_detection_conf from output')
        pass

    with open(out_path, 'w') as f:
        json.dump(fields, f, indent=1)

    print('Finished writing detection results to {}'.format(out_path))


def load_api_results_csv(filename, normalize_paths=True, filename_replacements=None, nrows=None):
    """
    [DEPRECATED]

    Loads .csv-formatted MegaDetector results to a pandas table

    Args:
        filename (str): path to the csv file to read
        normalize_paths (bool, optional): whether to apply os.path.normpath to the 'file'
            field in each image entry in the output file
        filename_replacements (dict, optional): replace some path tokens to match local paths
            to the original file structure
        nrows (int, optional): read only the first N rows of [filename]
    """

    if filename_replacements is None:
        filename_replacements = {}

    print('Loading MegaDetector results from {}'.format(filename))

    detection_results = pd.read_csv(filename,nrows=nrows)

    print('De-serializing MegaDetector results from {}'.format(filename))

    # Confirm that this is really a detector output file
    for s in ['image_path','max_confidence','detections']:
        assert s in detection_results.columns

    # Normalize paths to simplify comparisons later
    if normalize_paths:
        detection_results['image_path'] = detection_results['image_path'].apply(os.path.normpath)

    # De-serialize detections
    detection_results['detections'] = detection_results['detections'].apply(json.loads)

    # Optionally replace some path tokens to match local paths to the original blob structure
    # string_to_replace = list(options.detector_output_filename_replacements.keys())[0]
    for string_to_replace in filename_replacements:

        replacement_string = filename_replacements[string_to_replace]

        # i_row = 0
        for i_row in range(0,len(detection_results)):
            row = detection_results.iloc[i_row]
            fn = row['image_path']
            fn = fn.replace(string_to_replace,replacement_string)
            detection_results.at[i_row,'image_path'] = fn

    print('Finished loading and de-serializing MD results for {} images from {}'.format(
        len(detection_results),filename))

    return detection_results


def write_api_results_csv(detection_results, filename):
    """
    [DEPRECATED]

    Writes a Pandas table to csv in a way that's compatible with the .csv output
    format.  Currently just a wrapper around to_csv that forces output writing
    to go through a common code path.

    Args:
        detection_results (DataFrame): dataframe to write to [filename]
        filename (str): .csv filename to write
    """

    print('Writing detection results to {}'.format(filename))

    detection_results.to_csv(filename, index=False)

    print('Finished writing detection results to {}'.format(filename))
