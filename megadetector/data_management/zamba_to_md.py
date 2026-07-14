"""

zamba_to_md.py

Convert a labels.csv file produced by Zamba Cloud to a MD-formatted results file.

For video files, columns are expected to be:

    video_uuid (not used)
    original_filename (assumed to be a relative path name)
    top_k_label,top_k_probability, for k = 1..N
    [category name 1],[category name 2],...
    corrected_label

For image files, columns are expected to be:

    filepath (assumed to have the same stem as an original file)
    detection_category (MD category ID)
    detection_conf
    x1, y1, x2, y2 (in absolute coordinates)
    class1 (confidence of classification category 1)
    class2 (confidence of classification category 2)
    ...

"""

#%% Imports and constants

import os
import sys
import argparse

import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from megadetector.utils.ct_utils import write_json
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.wi_taxonomy_utils import get_common_name_from_prediction_string
from megadetector.utils.wi_taxonomy_utils import is_valid_prediction_string
from megadetector.detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP
from megadetector.utils.path_utils import find_images
from megadetector.visualization.visualization_utils import get_image_size

expected_video_columns = ('video_uuid','corrected_label','original_filename')
expected_image_columns = ('filepath', 'detection_category', 'detection_conf', 'x1', 'y1', 'x2', 'y2')

assert expected_image_columns[0] != expected_video_columns[0], \
    'Invalid image/video column names'

# How many classifications should we include per detection?
#
# Only impacts image (not video) files
top_n_classifications = 3

# Min classification confidence to include in ouutput
#
# Only impacts image (not video) files
min_classification_confidence = 0.05


#%% Main function

def zamba_results_to_md_results(input_file,output_file=None,image_folder=None):
    """
    Converts the .csv file [input_file] to the MD-formatted .json file [output_file].

    If [output_file] is None, '.json' will be appended to the input file.

    Args:
        input_file (str): the .csv file to convert
        output_file (str, optional): the output .json file (defaults to
            [input_file].json)
        image_folder (str, optional): folder name containing images, file name stems
            are assumed to be unique within this folder (only required  for image results, not
            video)

    Returns:
        str: the file to which results are written (the same as output_file if it was supplied)
    """

    if output_file is None:
        output_file = input_file + '.json'

    df = pd.read_csv(input_file)

    if expected_video_columns[0] in df.columns:
        _zamba_video_results_to_md_results(df=df,output_file=output_file)
    elif expected_image_columns[0] in df.columns:
        _zamba_image_results_to_md_results(df=df,
                                           output_file=output_file,
                                           image_folder=image_folder)
    else:
        raise ValueError('Could not determine Zamba file type')

    return output_file

# ...def zamba_results_to_md_results(...)


#%% Media-specific functions

def _zamba_image_results_to_md_results(df,output_file,image_folder):

    assert image_folder is not None, 'image_folder is required for image files'
    assert os.path.isdir(image_folder), 'Image folder {} not found'.format(image_folder)

    image_filenames_relative = find_images(image_folder,recursive=True,return_relative_paths=True)
    image_basename_to_relative_paths = defaultdict(list)
    for fn_relative in image_filenames_relative:
        bn = os.path.basename(fn_relative)
        image_basename_to_relative_paths[bn].append(fn_relative)

    print('Found {} images in {}'.format(len(image_filenames_relative),
                                         image_folder))

    for s in expected_image_columns:
        assert s in df.columns,\
            'Expected column {} not found, are you sure this is a Zamba results .csv file?'.format(
                s)

    # Non-required columns are assumed to be category scores
    #
    # If someone has a species called, e.g., "x1", they're out of luck
    category_columns = []
    for s in df.columns:
        if s not in expected_image_columns:
            category_columns.append(s)

    print('Found columns for {} categories'.format(len(category_columns)))

    # To handle the SpeciesNet special case, allow category names to be extracted from
    # column names.
    column_name_to_category_name = {}
    classification_category_name_to_id = {}
    classification_category_id_to_description = {}
    for column_name in category_columns:

        if is_valid_prediction_string(s):
            category_name = get_common_name_from_prediction_string(column_name)
        else:
            category_name = column_name
        column_name_to_category_name[column_name] = category_name
        category_id = str(len(classification_category_name_to_id))
        classification_category_name_to_id[category_name] = category_id
        classification_category_id_to_description[category_id] = column_name

    # ...for each column

    classification_category_id_to_name = invert_dictionary(classification_category_name_to_id)

    info = {}
    info['format_version'] = '1.6'
    info['detector'] = 'Zamba Cloud'
    info['classifier'] = 'Zamba Cloud'

    detection_category_id_to_name = {}

    records = df.to_dict(orient='records')

    # If we only ever see categories 1, 2, 3, assume MD categories
    #
    # record = records[0]
    detection_category_ids = set()
    for record in records:
        detection_category_id = str(int(record['detection_category']))
        detection_category_ids.add(detection_category_id)

    found_non_md_id = False
    for detection_category_id in detection_category_ids:
        if detection_category_id not in ('1','2','3'):
            found_non_md_id = True
            break

    if found_non_md_id:
        print('Warning: found non-MD category ID')
        # Use dummy category names
        for detection_category_id in detection_category_ids:
            detection_category_id_to_name[detection_category_id] = detection_category_id
    else:
        detection_category_id_to_name = DEFAULT_DETECTOR_LABEL_MAP

    filename_to_image = {}

    skipped_files = []

    images = []

    # record = records[0]
    for record in tqdm(records):

        file_identifier = record['filepath']

        image_basename = os.path.basename(file_identifier)
        if image_basename not in image_basename_to_relative_paths:
            print('Warning: file {} not found on disk, skipping'.format(file_identifier))
            skipped_files.append(file_identifier)
            continue

        fn_relative = image_basename_to_relative_paths[image_basename]
        assert len(fn_relative) > 0
        if len(fn_relative) > 1:
            print('Warning: file {} matches multiple files on disk, skipping'.format(file_identifier))
            skipped_files.append(file_identifier)
            continue
        fn_relative = fn_relative[0]
        assert isinstance(fn_relative,str)

        fn_abs = os.path.join(image_folder,fn_relative)
        assert os.path.isfile(fn_abs), 'File {} not found'.format(fn_abs)

        if fn_relative in filename_to_image:
            im = filename_to_image[fn_relative]
        else:
            im = {}
            try:
                w,h = get_image_size(fn_abs)
            except Exception as e:
                print('Warning: failed to read size from image {}: {}'.format(
                    fn_relative, str(e)))
                skipped_files.append(file_identifier)
                continue

            im['file'] = fn_relative
            im['detections'] = []
            filename_to_image[fn_relative] = im

        det = {}
        det['category'] = str(int(record['detection_category']))
        det['conf'] = float(record['detection_conf'])
        x1_norm = record['x1'] / w
        y1_norm = record['y1'] / h
        x2_norm = record['x2'] / w
        y2_norm = record['y2'] / h
        box_w = x2_norm - x1_norm
        box_h = y2_norm - y1_norm
        det['bbox'] = [x1_norm, y1_norm, box_w, box_h]
        im['detections'].append(det)

        classification_category_id_to_confidence = {}

        for column_name in category_columns:
            classification_conf = float(record[column_name])
            category_name = column_name_to_category_name[column_name]
            classification_category_id = classification_category_name_to_id[category_name]
            classification_category_id_to_confidence[classification_category_id] = \
                classification_conf

        classification_category_id_to_confidence = \
            sort_dictionary_by_value(classification_category_id_to_confidence,reverse=True)

        for i_category,classification_category_id in \
            enumerate(classification_category_id_to_confidence.keys()):

            classification_conf = \
                classification_category_id_to_confidence[classification_category_id]
            if (i_category > top_n_classifications) or \
                (classification_conf < min_classification_confidence):
                break
            if 'classifications' not in det:
                det['classifications'] = []
            det['classifications'].append([classification_category_id,classification_conf])

        # ...for each classification category

        images.append(im)

    # ...for each record

    if len(skipped_files) > 0:
        print('Warning: skipped {} of {} files'.format(
            len(skipped_files), len(records)))

    results = {}
    results['info'] = info
    results['detection_categories'] = detection_category_id_to_name
    results['classification_categories'] = classification_category_id_to_name
    results['classification_category_descriptions'] = classification_category_id_to_description
    results['images'] = images

    write_json(output_file,results)

    print('Wrote results for {} images to {}'.format(len(images),output_file))

# ...def _zamba_image_results_to_md_results(df,output_file)


def _zamba_video_results_to_md_results(df,output_file):

    for s in expected_video_columns:
        assert s in df.columns,\
            'Expected column {} not found, are you sure this is a Zamba results .csv file?'.format(
                s)

    # How many results are included per file?
    assert 'top_1_probability' in df.columns and 'top_1_label' in df.columns
    top_k = 2
    while(True):
        p_string = 'top_' + str(top_k) + '_probability'
        label_string = 'top_' + str(top_k) + '_label'

        if p_string in df.columns:
            assert label_string in df.columns,\
                'Oops, {} is a column but {} is not'.format(
                    p_string,label_string)
            top_k += 1
            continue
        else:
            assert label_string not in df.columns,\
                'Oops, {} is a column but {} is not'.format(
                    label_string,p_string)
            top_k -= 1
            break

    print('Found {} probability column pairs'.format(top_k))

    # Category names start after the fixed columns and the probability columns
    category_names = []
    column_names = list(df.columns)
    first_category_name_index = 0
    while('top_' in column_names[first_category_name_index] or \
          column_names[first_category_name_index] in expected_video_columns):
        first_category_name_index += 1

    i_column = first_category_name_index
    while( (i_column < len(column_names)) and (column_names[i_column] != 'corrected_label') ):
        category_names.append(column_names[i_column])
        i_column += 1

    print('Found {} categories:\n'.format(len(category_names)))

    for s in category_names:
        print(s)

    info = {}
    info['format_version'] = '1.6'
    info['detector'] = 'Zamba Cloud'
    info['classifier'] = 'Zamba Cloud'

    detection_category_id_to_name = {}
    for category_id,category_name in enumerate(category_names):
        detection_category_id_to_name[str(category_id)] = category_name
    detection_category_name_to_id = {v: k for k, v in detection_category_id_to_name.items()}

    images = []

    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():

        im = {}
        images.append(im)
        im['file'] = row['original_filename']

        detections = []

        # k = 1
        for k in range(1,top_k+1):
            label = row['top_{}_label'.format(k)]
            confidence = row['top_{}_probability'.format(k)]
            det = {}
            det['category'] = detection_category_name_to_id[label]
            det['conf'] = confidence
            det['bbox'] = [0,0,1.0,1.0]
            detections.append(det)

        im['detections'] = detections

    # ...for each row

    results = {}
    results['info'] = info
    results['detection_categories'] = detection_category_id_to_name
    results['images'] = images

    write_json(output_file,results)

# ..._zamba_video_results_to_md_results(...)


#%% Interactive driver

if False:

    pass

    #%%

    input_file = r"G:\temp\labels-job-b95a4b76-e332-4e17-ab40-03469392d36a-2023-11-04_16-28-50.060130.csv"
    output_file = None
    zamba_results_to_md_results(input_file,output_file)


#%% Command-line driver

def main():
    """
    Command-line driver for zamba_to_md
    """

    parser = argparse.ArgumentParser(
        description='Convert a Zamba-formatted .csv results file to a MD-formatted .json results file')

    parser.add_argument(
        'input_file',
        type=str,
        help='input .csv file')

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='output .json file (defaults to input file appended with ".json")')

    parser.add_argument(
        '--image_folder',
        type=str,
        default=None,
        help='Folder name containing images (only required for image results, not video)')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    zamba_results_to_md_results(args.input_file,args.output_file,args.image_folder)

if __name__ == '__main__':
    main()
