"""

convert_output_format.py

Converts between file .json and .csv representations of MD output.  The .csv format is
largely obsolete, don't use it unless you're super-duper sure you need it.

"""

#%% Constants and imports

import argparse
import json
import sys
import os

from tqdm import tqdm
from collections import defaultdict

import pandas as pd

from megadetector.postprocessing.load_api_results import load_api_results_csv
from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file
from megadetector.data_management.annotations import annotation_constants
from megadetector.utils import ct_utils

CONF_DIGITS = 3


#%% Conversion functions

def convert_json_to_csv(input_path,
                        output_path=None,
                        min_confidence=None,
                        omit_bounding_boxes=False,
                        output_encoding=None,
                        overwrite=True,
                        verbose=False):
    """
    Converts a MD results .json file to a totally non-standard .csv format.

    If [output_path] is None, will convert x.json to x.csv.

    Args:
        input_path (str): the input .json file to convert
        output_path (str, optional): the output .csv file to generate; if this is None, uses
            [input_path].csv
        min_confidence (float, optional): the minimum-confidence detection we should include
            in the "detections" column; has no impact on the other columns
        omit_bounding_boxes (bool, optional): whether to leave out the json-formatted bounding
            boxes that make up the "detections" column, which are not generally useful for someone
            who wants to consume this data as a .csv file
        output_encoding (str, optional): encoding to use for the .csv file
        overwrite (bool, optional): whether to overwrite an existing .csv file; if this is False and
            the output file exists, no-ops and returns
        verbose (bool, optional): enable additional debug output
    """

    if output_path is None:
        output_path = os.path.splitext(input_path)[0]+'.csv'

    if os.path.isfile(output_path) and (not overwrite):
        print('File {} exists, skipping json --> csv conversion'.format(output_path))
        return

    print('Loading json results from {}...'.format(input_path))
    json_output = load_md_or_speciesnet_file(input_path,
                                             verbose=verbose)

    def clean_category_name(s):
        return s.replace(',','_').replace(' ','_').lower()

    # Create column names for max detection confidences
    detection_category_id_to_max_conf_column_name = {}
    for category_id in json_output['detection_categories'].keys():
        category_name = clean_category_name(json_output['detection_categories'][category_id])
        detection_category_id_to_max_conf_column_name[category_id] = \
            'max_conf_' + category_name

    classification_category_id_to_max_conf_column_name = {}

    # Create column names for max classification confidences (if necessary)
    if 'classification_categories' in json_output.keys():

        for category_id in json_output['classification_categories'].keys():
            category_name = clean_category_name(json_output['classification_categories'][category_id])
            classification_category_id_to_max_conf_column_name[category_id] = \
                'max_classification_conf_' + category_name

    # There are several .json fields for which we add .csv columns; other random bespoke fields
    # will be ignored.
    optional_fields = ['width','height','datetime','exif_metadata']
    optional_fields_present = set()

    # Iterate once over the data to check for optional fields
    print('Looking for optional fields...')

    for im in tqdm(json_output['images']):
        # Which optional fields are present for this image?
        for k in im.keys():
            if k in optional_fields:
                optional_fields_present.add(k)

    optional_fields_present = sorted(list(optional_fields_present))
    if len(optional_fields_present) > 0:
        print('Found {} optional fields'.format(len(optional_fields_present)))

    print('Formatting results...')

    output_records = []

    # i_image = 0; im = json_output['images'][i_image]
    for im in tqdm(json_output['images']):

        output_record = {}
        output_records.append(output_record)

        output_record['image_path'] = im['file']
        output_record['max_confidence'] = ''
        output_record['detections'] = ''

        for field_name in optional_fields_present:
            output_record[field_name] = ''
            if field_name in im:
                output_record[field_name] = im[field_name]

        for detection_category_id in detection_category_id_to_max_conf_column_name:
            column_name = detection_category_id_to_max_conf_column_name[detection_category_id]
            output_record[column_name] = 0

        for classification_category_id in classification_category_id_to_max_conf_column_name:
            column_name = classification_category_id_to_max_conf_column_name[classification_category_id]
            output_record[column_name] = 0

        if 'failure' in im and im['failure'] is not None:
            output_record['max_confidence'] = 'failure'
            output_record['detections'] = im['failure']
            # print('Skipping failed image {} ({})'.format(im['file'],im['failure']))
            continue

        max_conf = ct_utils.get_max_conf(im)
        detection_category_id_to_max_conf = defaultdict(float)
        classification_category_id_to_max_conf = defaultdict(float)
        detections = []

        # d = im['detections'][0]
        for d in im['detections']:

            # Skip sub-threshold detections
            if (min_confidence is not None) and (d['conf'] < min_confidence):
                continue

            input_bbox = d['bbox']

            # Our .json format is xmin/ymin/w/h
            #
            # Our .csv format was ymin/xmin/ymax/xmax
            xmin = input_bbox[0]
            ymin = input_bbox[1]
            xmax = input_bbox[0] + input_bbox[2]
            ymax = input_bbox[1] + input_bbox[3]
            output_detection = [ymin, xmin, ymax, xmax]
            output_detection.append(d['conf'])
            output_detection.append(int(d['category']))
            detections.append(output_detection)

            detection_category_id = d['category']
            detection_category_max = detection_category_id_to_max_conf[detection_category_id]
            if d['conf'] > detection_category_max:
                detection_category_id_to_max_conf[detection_category_id] = d['conf']

            if 'classifications' in d:

                for c in d['classifications']:
                    classification_category_id = c[0]
                    classification_conf = c[1]
                    classification_category_max = \
                        classification_category_id_to_max_conf[classification_category_id]
                    if classification_conf > classification_category_max:
                        classification_category_id_to_max_conf[classification_category_id] = d['conf']

                # ...for each classification

            # ...if we have classification results for this detection

        # ...for each detection

        detection_string = ''
        if not omit_bounding_boxes:
            detection_string = json.dumps(detections)

        output_record['detections'] = detection_string
        output_record['max_confidence'] = max_conf

        for detection_category_id in detection_category_id_to_max_conf_column_name:
            column_name = detection_category_id_to_max_conf_column_name[detection_category_id]
            output_record[column_name] = \
                detection_category_id_to_max_conf[detection_category_id]

        for classification_category_id in classification_category_id_to_max_conf_column_name:
            column_name = classification_category_id_to_max_conf_column_name[classification_category_id]
            output_record[column_name] = \
                classification_category_id_to_max_conf[classification_category_id]

    # ...for each image

    print('Writing to csv...')

    df = pd.DataFrame(output_records)

    if omit_bounding_boxes:
        df = df.drop('detections',axis=1)
    df.to_csv(output_path,index=False,header=True)

# ...def convert_json_to_csv(...)


def convert_csv_to_json(input_path,output_path=None,overwrite=True):
    """
    Convert .csv to .json.  If output_path is None, will convert x.csv to x.json.  This
    supports a largely obsolete .csv format, there's almost no reason you want to do this.

    Args:
        input_path (str): .csv filename to convert to .json
        output_path (str, optional): the output .json file to generate; if this is None, uses
            [input_path].json
        overwrite (bool, optional): whether to overwrite an existing .json file; if this is
            False and the output file exists, no-ops and returns

    """

    if output_path is None:
        output_path = os.path.splitext(input_path)[0]+'.json'

    if os.path.isfile(output_path) and (not overwrite):
        print('File {} exists, skipping csv --> json conversion'.format(output_path))
        return

    # Format spec:
    #
    # https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing

    print('Loading csv results...')
    df = load_api_results_csv(input_path)

    info = {
        "format_version":"1.2",
        "detector": "unknown",
        "detection_completion_time" : "unknown",
        "classifier": "unknown",
        "classification_completion_time": "unknown"
    }

    classification_categories = {}
    detection_categories = annotation_constants.detector_bbox_categories

    images = []

    # i_file = 0; row = df.iloc[i_file]
    for i_file,row in df.iterrows():

        image = {}
        image['file'] = row['image_path']
        image['max_detection_conf'] = round(row['max_confidence'], CONF_DIGITS)
        src_detections = row['detections']
        out_detections = []

        for i_detection,detection in enumerate(src_detections):

            # Our .csv format was ymin/xmin/ymax/xmax
            #
            # Our .json format is xmin/ymin/w/h
            ymin = detection[0]
            xmin = detection[1]
            ymax = detection[2]
            xmax = detection[3]
            bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
            conf = detection[4]
            i_class = detection[5]
            out_detection = {}
            out_detection['category'] = str(i_class)
            out_detection['conf'] = conf
            out_detection['bbox'] = bbox
            out_detections.append(out_detection)

        # ...for each detection

        image['detections'] = out_detections
        images.append(image)

    # ...for each image
    json_out = {}
    json_out['info'] = info
    json_out['detection_categories'] = detection_categories
    json_out['classification_categories'] = classification_categories
    json_out['images'] = images

    json.dump(json_out,open(output_path,'w'),indent=1)

# ...def convert_csv_to_json(...)


#%% Interactive driver

if False:

    #%%

    input_path = r'c:\temp\test.json'
    min_confidence = None
    output_path = input_path + '.csv'
    convert_json_to_csv(input_path,output_path,min_confidence=min_confidence,
                        omit_bounding_boxes=False)

    #%%

    base_path = r'c:\temp\json'
    input_paths = os.listdir(base_path)
    input_paths = [os.path.join(base_path,s) for s in input_paths]

    min_confidence = None
    for input_path in input_paths:
        output_path = input_path + '.csv'
        convert_json_to_csv(input_path,output_path,min_confidence=min_confidence,
                            omit_bounding_boxes=True)

    #%% Concatenate .csv files from a folder

    import glob
    csv_files = glob.glob(os.path.join(base_path,'*.json.csv' ))
    master_csv = os.path.join(base_path,'all.csv')

    print('Concatenating {} files to {}'.format(len(csv_files),master_csv))

    header = None
    with open(master_csv, 'w') as fout:

        for filename in tqdm(csv_files):

            with open(filename) as fin:

                lines = fin.readlines()

                if header is not None:
                    assert lines[0] == header
                else:
                    header = lines[0]
                    fout.write(header)

                for line in lines[1:]:
                    if len(line.strip()) == 0:
                        continue
                    fout.write(line)

        # ...for each .csv file

    # with open(master_csv)


#%% Command-line driver

def main():
    """
    Command-line driver for convert_output_format(), which converts
    json <--> csv.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str,
                        help='Input filename ending in .json or .csv')
    parser.add_argument('--output_path',type=str,default=None,
                        help='Output filename ending in .json or .csv (defaults to ' + \
                             'input file, with .json/.csv replaced by .csv/.json)')
    parser.add_argument('--omit_bounding_boxes',action='store_true',
                        help='Output bounding box text from .csv output (large and usually not useful)')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.output_path is None:
        if args.input_path.endswith('.csv'):
            args.output_path = args.input_path[:-4] + '.json'
        elif args.input_path.endswith('.json'):
            args.output_path = args.input_path[:-5] + '.csv'
        else:
            raise ValueError('Illegal input file extension')

    if args.input_path.endswith('.csv') and args.output_path.endswith('.json'):
        assert not args.omit_bounding_boxes, \
            '--omit_bounding_boxes does not apply to csv --> json conversion'
        convert_csv_to_json(args.input_path,args.output_path)
    elif args.input_path.endswith('.json') and args.output_path.endswith('.csv'):
        convert_json_to_csv(args.input_path,args.output_path,omit_bounding_boxes=args.omit_bounding_boxes)
    else:
        raise ValueError('Illegal format combination')

if __name__ == '__main__':
    main()
