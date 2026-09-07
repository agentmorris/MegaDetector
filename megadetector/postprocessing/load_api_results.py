"""

load_api_results.py

DEPRECATED

As of 2023.12, this module is still used in postprocessing and RDE, but it's not recommended
for new code.

Loads the output of the batch processing API (json) into a Pandas dataframe.

Includes functions to read/write the (very very old) .csv results format.

"""

#%% Imports

import os
import json
import math
import shutil

import pandas as pd

from megadetector.utils.ct_utils import get_max_conf
from megadetector.utils.ct_utils import make_test_folder
from megadetector.utils.ct_utils import write_json
from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file


#%% Constants

#: Value used in the dataframes returned by load_api_results() to indicate that a field
#: was absent for a particular image in the source file.
#:
#: Fields that are not part of the MegaDetector output format - and even a few that are,
#: e.g. 'failure' - may be present for only some of the images in a results file.  Pandas
#: fills the corresponding cells with NaN when a .json file is read into a dataframe, which
#: is indistinguishable from an explicit null in the .json file, and which forces columns
#: that contain only integers into floating-point representation.  Instead, we fil those
#: cells with this sentinel value, which allows write_api_results() to omit those fields
#: and preserves the types of the values that *are* present.
#:
#: Use is_missing_field_value() rather than comparing to this value directly.
MISSING_FIELD_VALUE = '##megadetector-missing-field-value##'


#%% Functions for loading .json results into a Pandas DataFrame, and writing back to .json

def is_missing_field_value(v):
    """
    Determines whether [v] - typically a value read from a dataframe returned by
    load_api_results() - represents a field that had no value for a particular image.

    This is True for MISSING_FIELD_VALUE (used by load_api_results() for fields that were
    absent for a particular image), for None (used for fields that were explicitly null in
    the source file), and for NaN (which is what Pandas uses for absent fields in dataframes
    that were not loaded by load_api_results()).

    Args:
        v (object): the value to test

    Returns:
        bool: whether [v] represents a missing field value
    """

    return (v is None) or \
           (isinstance(v,str) and (v == MISSING_FIELD_VALUE)) or \
           (isinstance(v,float) and math.isnan(v))

# ...def is_missing_field_value(...)


def load_api_results(api_output_path,
                     normalize_paths=True,
                     filename_replacements=None,
                     force_forward_slashes=True
                     ):
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
        detection_results: pd.DataFrame, contains at least the columns ['file', 'detections','failure'].
        Cells corresponding to fields that were absent for a particular image in [api_output_path]
        are populated with MISSING_FIELD_VALUE, rather than the NaN that Pandas would use by
        default; see is_missing_field_value().
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
            im['max_detection_conf'] = get_max_conf(im)

    # Populate fields that are absent for some images with a sentinel value, so we can tell
    # them apart from fields that are explicitly null, and so that columns containing only
    # integers don't get converted to floating-point.  See MISSING_FIELD_VALUE.
    image_field_names = {}
    for im in detection_results['images']:
        for field_name in im.keys():
            image_field_names[field_name] = True

    for im in detection_results['images']:
        for field_name in image_field_names.keys():
            if field_name not in im:
                im[field_name] = MISSING_FIELD_VALUE

    # Pack the json output into a Pandas DataFrame
    detection_results = pd.DataFrame(detection_results['images'])

    print('Finished loading MegaDetector results for {} images from {}'.format(
            len(detection_results),api_output_path))

    return detection_results, other_fields

# ...def load_api_results(...)


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

        # Remove fields that weren't present for this image in the file this table was
        # loaded from; see MISSING_FIELD_VALUE.  Fields that were explicitly null are
        # left alone.
        field_names_to_remove = []
        for field_name in im.keys():
            if isinstance(im[field_name],str) and (im[field_name] == MISSING_FIELD_VALUE):
                field_names_to_remove.append(field_name)
        for field_name in field_names_to_remove:
            del im[field_name]

        # An explicitly-null failure indicator is meaningless, remove it
        if ('failure' in im) and (im['failure'] is None):
            del im['failure']

    # ...for each image

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

    write_json(out_path,fields)

    print('Finished writing detection results to {}'.format(out_path))

# ...def write_api_results(...)


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

# ...def load_api_results_csv(...)


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

    output_dir = os.path.dirname(filename)
    if len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)

    detection_results.to_csv(filename, index=False)

    print('Finished writing detection results to {}'.format(filename))

# ...def write_api_results_csv(...)


#%% Tests

def test_load_api_results():
    """
    Test that a .json results file survives a load_api_results()/write_api_results()
    round trip, particularly fields that are present for only some images.
    """

    test_folder = make_test_folder(subfolder='load_api_results_tests')

    try:

        input_file = os.path.join(test_folder,'test_results.json')
        output_file = os.path.join(test_folder,'test_results_filtered.json')

        images = [
            {
                'file':'test_folder/failure.jpg',
                'detections':None,
                'failure':'synthetic failure'
            },
            {
                'file':'test_folder/string_field.jpg',
                'detections':[{'category':'1','conf':0.797,
                               'bbox':[0.591,0.077,0.047,0.047]}],
                'synthetic_string_field':'synthetic value'
            },
            {
                'file':'test_folder/int_field.jpg',
                'detections':[{'category':'1','conf':0.254,
                               'bbox':[0.622,0.077,0.012,0.032]}],
                'synthetic_int_field':10
            },
            {
                'file':'test_folder/null_field.jpg',
                'detections':[],
                'synthetic_string_field':None
            },
            {
                'file':'test_folder/no_extra_fields.jpg',
                'detections':[]
            }
        ]

        input_data = {
            'info':{'format_version':'1.3','detector':'synthetic_detector'},
            'detection_categories':{'1':'animal'},
            'synthetic_file_level_field':{'test_key':'test value'},
            'images':images
        }

        write_json(input_file,input_data)

        detection_results_table, other_fields = load_api_results(input_file)

        # Fields that are absent for an image should be represented with a sentinel value,
        # rather than with the NaN Pandas would use by default
        absent_field_value = detection_results_table['synthetic_string_field'].iloc[4]
        assert absent_field_value == MISSING_FIELD_VALUE, \
            'Absent field represented as {}, expected a sentinel value'.format(absent_field_value)
        assert is_missing_field_value(absent_field_value), \
            'Sentinel value not recognized as a missing field value'

        write_api_results(detection_results_table,other_fields,output_file)

        with open(output_file,'r') as f:
            output_data = json.load(f)

        # Fields that were absent should still be absent, fields that were explicitly null
        # should still be null, and integers should not have become floating-point values
        assert output_data['images'] == images, \
            'Image fields did not survive a load/write round trip'
        assert output_data['synthetic_file_level_field'] == \
            input_data['synthetic_file_level_field'], \
            'File-level fields did not survive a load/write round trip'

    finally:

        shutil.rmtree(test_folder,ignore_errors=True)

# ...def test_load_api_results(...)
