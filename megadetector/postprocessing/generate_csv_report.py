"""

generate_csv_report.py

Generates a .csv report from a MD-formatted .json file with the following columns:

* filename
* datetime (if images or EXIF information is supplied)
* detection_category
* max_detection_confidence
* classification_category
* max_classification_confidence
* count

One row is generated per category pair per image.  For example, these would be unique rows:

image0001.jpg,animal,deer,4
image0001.jpg,animal,lion,4
image0001.jpg,animal,[none],4
image0001.jpg,person,[none],2

Images with no above-threshold detections will have a single row:

image0001.jpg,empty,[none],-1

Images with processing errors will have a single row:

image0001.jpg,error,error_string,-1

"""

#%% Constants and imports

import os
import json
import tempfile
import sys
import argparse
import uuid

import pandas as pd

from copy import deepcopy

from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file
from megadetector.utils.ct_utils import get_max_conf
from megadetector.utils.ct_utils import is_list_sorted
from megadetector.detection.run_detector import \
    get_typical_confidence_threshold_from_results
from megadetector.data_management.read_exif import \
    read_exif_from_folder, ReadExifOptions, minimal_exif_tags

default_classification_threshold = 0.3
unknown_datetime_tag = ''


#%% Functions

def generate_csv_report(md_results_file,
                        output_file=None,
                        datetime_source=None,
                        folder_level_columns=None,
                        detection_confidence_threshold=None,
                        classification_confidence_threshold=None,
                        verbose=True):
    """
    Generates a .csv report from a MD-formatted .json file

    Args:
        md_results_file (str): MD results .json file for which we should generate a report
        output_file (str, optional): .csv file to write; if this is None, we'll use md_results_file.csv
        datetime_source (str, optional): if datetime information is required, this should point to
            a folder of images, a MD results .json file (can be the same as the input file), or
            an exif_info.json file created with read_exif().
        folder_level_columns (list of int, optional): list of folder levels (where zero is the top-level
            folder in a path name) for which we should create separate columns.  Should be zero-indexed ints,
            or a comma-delimited list of zero-indexed int-strings.
        detection_confidence_threshold (float, optional): detections below this confidence threshold will not
            be included in the output data.  Defaults to the recommended value based on the .json file.
        classification_confidence_threshold (float, optional): classifications below this confidence threshold will
            not be included in the output data (i.e., detections will be considered "animal").
        verbose (bool, optional): enable debug output, including the progress bar,

    Returns:
        str: the output .csv filename
    """

    ##%% Load results file

    results = load_md_or_speciesnet_file(md_results_file)

    print('Loaded results for {} images'.format(len(results['images'])))

    detection_category_id_to_name = results['detection_categories']
    classification_category_id_to_name = None
    if 'classification_categories' in results:
        classification_category_id_to_name = results['classification_categories']

    if output_file is None:
        output_file = md_results_file + '.csv'

    ##%% Read datetime information if necessary

    filename_to_datetime_string = None

    if datetime_source is not None:

        all_exif_results = None

        if os.path.isdir(datetime_source):

            # Read EXIF info from images
            read_exif_options = ReadExifOptions()
            read_exif_options.tags_to_include = minimal_exif_tags
            read_exif_options.byte_handling = 'delete'
            exif_cache_file = os.path.join(tempfile.gettempdir(),
                                           'md-exif-data',
                                           str(uuid.uuid1())+'.json')
            print('Reading EXIF datetime info from {}, writing to {}'.format(
                datetime_source,exif_cache_file))
            os.makedirs(os.path.dirname(exif_cache_file),exist_ok=True)

            all_exif_results = read_exif_from_folder(input_folder=datetime_source,
                                                     output_file=exif_cache_file,
                                                     options=read_exif_options,
                                                     recursive=True)

        else:

            assert os.path.isfile(datetime_source), \
                'datetime source {} is neither a folder nor a file'.format(datetime_source)

            # Is this the same file we've already read?

            # Load this, decide whether it's a MD file or an exif_info file
            with open(datetime_source,'r') as f:
                d = json.load(f)

            if isinstance(d,list):
                all_exif_results = d
            else:
                assert isinstance(d,dict), 'Unrecognized file format supplied as datetime source'
                assert 'images' in d,\
                    'The datetime source you provided doesn\'t look like a valid source .json file'
                all_exif_results = []
                found_datetime = False
                for im in d['images']:
                    exif_result = {'file_name':im['file']}
                    if 'datetime' in im:
                        found_datetime = True
                        exif_result['exif_tags'] = {'DateTimeOriginal':im['datetime']}
                    all_exif_results.append(exif_result)
                if not found_datetime:
                    print('Warning: a MD results file was supplied as the datetime source, but it does not appear '
                          'to contain datetime information.')

        # ...if datetime_source is a folder/file

        assert all_exif_results is not None

        filename_to_datetime_string = {}

        for exif_result in all_exif_results:

            datetime_string = unknown_datetime_tag
            if ('exif_tags' in exif_result) and \
               (exif_result['exif_tags'] is not None) and \
               ('DateTimeOriginal' in exif_result['exif_tags']):
                datetime_string = exif_result['exif_tags']['DateTimeOriginal']
                if datetime_string is None:
                    datetime_string = ''
                else:
                    assert isinstance(datetime_string,str), 'Unrecognized datetime format'
            filename_to_datetime_string[exif_result['file_name']] = datetime_string

        # ...for each exif result

        image_files = [im['file'] for im in results['images']]
        image_files_set = set(image_files)

        files_in_exif_but_not_in_results = []
        files_in_results_but_not_in_exif = []
        files_with_no_datetime_info = []

        for fn in filename_to_datetime_string:
            dts = filename_to_datetime_string[fn]
            if (dts is None) or (dts == unknown_datetime_tag) or (len(dts) == 0):
                files_with_no_datetime_info.append(fn)
            if fn not in image_files_set:
                files_in_exif_but_not_in_results.append(fn)

        for fn in image_files_set:
            if fn not in filename_to_datetime_string:
                files_in_results_but_not_in_exif.append(fn)

        print('{} files (of {}) in EXIF info not found in MD results'.format(
            len(files_in_exif_but_not_in_results),len(filename_to_datetime_string)
        ))

        print('{} files (of {}) in MD results not found in MD EXIF info'.format(
            len(files_in_results_but_not_in_exif),len(image_files_set)
        ))

        print('Failed to read datetime information for {} files (of {}) in EXIF info'.format(
            len(files_with_no_datetime_info),len(filename_to_datetime_string)
        ))

    # ...if we need to deal with datetimes


    ##%% Parse folder level column specifier

    if folder_level_columns is not None:

        if isinstance(folder_level_columns,str):
            tokens = folder_level_columns.split(',')
            folder_level_columns = [int(s) for s in tokens]
        for folder_level in folder_level_columns:
            if (not isinstance(folder_level,int)) or (folder_level < 0):
                raise ValueError('Illegal folder level specifier {}'.format(
                    str(folder_level_columns)))


    ##%% Fill in default thresholds

    if classification_confidence_threshold is None:
        classification_confidence_threshold = default_classification_threshold
    if detection_confidence_threshold is None:
        detection_confidence_threshold = \
            get_typical_confidence_threshold_from_results(results)

    assert detection_confidence_threshold is not None


    ##%% Fill in output records

    output_records = []

    # For each image
    #
    # im = results['images'][0]
    for im in results['images']:

        """
        * filename
        * datetime (if images or EXIF information is supplied)
        * detection_category
        * max_detection_confidence
        * classification_category
        * max_classification_confidence
        * count
        """

        base_record = {}

        base_record['filename'] = im['file'].replace('\\','/')

        # Datetime (if necessary)
        datetime_string = ''
        if filename_to_datetime_string is not None:
            if im['file'] in filename_to_datetime_string:
                datetime_string = filename_to_datetime_string[im['file']]
        base_record['datetime'] = datetime_string

        for s in ['detection_category','max_detection_confidence',
                  'classification_category','max_classification_confidence',
                  'count']:
            base_record[s] = ''

        # Folder level columns
        tokens = im['file'].split('/')

        if folder_level_columns is not None:

            for folder_level in folder_level_columns:
                folder_level_column_name = 'folder_level_' + str(folder_level).zfill(2)
                if folder_level >= len(tokens):
                    folder_level_value = ''
                else:
                    folder_level_value = tokens[folder_level]
                base_record[folder_level_column_name] = folder_level_value

        records_this_image = []

        # Create one output row if this image failed
        if 'failure' in im and im['failure'] is not None and len(im['failure']) > 0:

            record = deepcopy(base_record)
            record['detection_category'] = 'error'
            record['classification_category'] = im['failure']
            records_this_image.append(record)
            assert 'detections' not in im or im['detections'] is None

        else:

            assert 'detections' in im and im['detections'] is not None

            # Count above-threshold detections
            detections_above_threshold = []
            for det in im['detections']:
                if det['conf'] >= detection_confidence_threshold:
                    detections_above_threshold.append(det)
            max_detection_conf = get_max_conf(im)

            # Create one output row if this image is empty (i.e., has no
            # above-threshold detections)
            if len(detections_above_threshold) == 0:

                record = deepcopy(base_record)
                record['detection_category'] = 'empty'
                record['max_detection_confidence'] = max_detection_conf
                records_this_image.append(record)

            # ...if this image is empty

            else:

                # Maps a string of the form:
                #
                # detection_category:classification_category
                #
                # ...to a dict with fields ['max_detection_conf','max_classification_conf','count']
                category_info_string_to_record = {}

                for det in detections_above_threshold:

                    assert det['conf'] >= detection_confidence_threshold

                    detection_category_name = detection_category_id_to_name[det['category']]
                    detection_confidence = det['conf']
                    classification_category_name = ''
                    classification_confidence = 0.0

                    if ('classifications' in det) and (len(det['classifications']) > 0):

                        # Classifications should always be sorted by confidence.  Not
                        # technically required, but always true in practice.
                        assert is_list_sorted([c[1] for c in det['classifications']]), \
                            'This script does not yet support unsorted classifications'
                        assert classification_category_id_to_name is not None, \
                            'If classifications are present, category mappings should be present'

                        # Only use the first classification
                        classification = det['classifications'][0]
                        if classification[1] >= classification_confidence_threshold:
                            classification_category_name = \
                                classification_category_id_to_name[classification[0]]
                            classification_confidence = classification[1]

                    # ...if classifications are present

                    # E.g. "animal:rodent", or "vehicle:"
                    category_info_string = detection_category_name + ':' + classification_category_name

                    if category_info_string not in category_info_string_to_record:
                        category_info_string_to_record[category_info_string] = {
                            'max_detection_confidence':0.0,
                            'max_classification_confidence':0.0,
                            'count':0,
                            'detection_category':detection_category_name,
                            'classification_category':classification_category_name
                        }

                    record = category_info_string_to_record[category_info_string]
                    record['count'] += 1
                    if detection_confidence > record['max_detection_confidence']:
                        record['max_detection_confidence'] = detection_confidence
                    if classification_confidence > record['max_classification_confidence']:
                        record['max_classification_confidence'] = classification_confidence

                # ...for each detection

                for record_in in category_info_string_to_record.values():
                    assert record_in['count'] > 0
                    record_out = deepcopy(base_record)
                    for k in record_in.keys():
                        assert k in record_out.keys()
                        record_out[k] = record_in[k]
                    records_this_image.append(record_out)

            # ...is this empty/non-empty?

        # ...if this image failed/didn't fail

        # Add to [records]
        output_records.extend(records_this_image)

    # ...for each image

    # Make sure every record has the same columns

    if len(output_records) == 0:
        print('Warning: no output records generated')
    else:
        column_names = output_records[0].keys()
        for record in output_records:
            assert record.keys() == column_names

        # Create folder for output file if necessary
        output_dir = os.path.dirname(output_file)
        if len(output_dir) > 0:
            os.makedirs(output_dir, exist_ok=True)

        # Write to .csv
        df = pd.DataFrame(output_records)
        df.to_csv(output_file,header=True,index=False)

    # from megadetector.utils.path_utils import open_file; open_file(output_file)

# ...generate_csv_report(...)


# %%

#%% Interactive driver

if False:

    pass

    #%% Configure options

    r"""
    python run_detector_batch.py MDV5A "g:\temp\md-test-images"
    "g:\temp\md-test-images\md_results_with_datetime.json"
    --recursive --output_relative_filenames --include_image_timestamp --include_exif_data
    """

    md_results_file = 'g:/temp/csv-report-test/md-results.json'
    datetime_source = 'g:/temp/csv-report-test/exif_data.json'

    # datetime_source = 'g:/temp/md-test-images'
    # datetime_source = 'g:/temp/md-test-images/md_results_with_datetime.json'
    # md_results_file = 'g:/temp/md-test-images/md_results_with_datetime.json'
    # md_results_file = 'g:/temp/md-test-images/speciesnet_results_md_format.json'

    output_file = None
    folder_level_columns = [0,1,2,3]
    detection_confidence_threshold = None
    classification_confidence_threshold = None
    verbose = True


    #%% Programmatic execution

    generate_csv_report(md_results_file=md_results_file,
                        output_file=output_file,
                        datetime_source=datetime_source,
                        folder_level_columns=folder_level_columns,
                        detection_confidence_threshold=detection_confidence_threshold,
                        classification_confidence_threshold=classification_confidence_threshold,
                        verbose=verbose)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser(
        description='Generates a .csv report from a MD-formatted .json file')

    parser.add_argument(
        'md_results_file',
        type=str,
        help='Path to MD results file (.json)')

    parser.add_argument(
        '--output_file',
        type=str,
        help='Output filename (.csv) (if omitted, will append .csv to the input file)')

    parser.add_argument(
        '--datetime_source',
        type=str,
        default=None,
        help='Image folder, exif_info.json file, or MD results file from which we should read datetime information'
        )

    parser.add_argument(
        '--folder_level_columns',
        type=str,
        default=None,
        help='Comma-separated list of zero-indexed folder levels that should become columns in the output file'
        )

    parser.add_argument(
        '--detection_confidence_threshold',
        type=float,
        default=None,
        help='Detection threshold (if omitted, chooses a reasonable default based on the .json file)'
        )

    parser.add_argument(
        '--classification_confidence_threshold',
        type=float,
        default=None,
        help='Classification threshold (default {})'.format(default_classification_threshold)
        )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable additional debug output'
        )


    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    generate_csv_report(md_results_file=args.md_results_file,
                        output_file=args.output_file,
                        datetime_source=args.datetime_source,
                        folder_level_columns=args.folder_level_columns,
                        detection_confidence_threshold=args.detection_confidence_threshold,
                        classification_confidence_threshold=args.classification_confidence_threshold,
                        verbose=args.verbose)

if __name__ == '__main__':
    main()
