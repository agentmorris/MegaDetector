"""

mewc_to_md.py

Converts the output of the MEWC inference scripts to the MD output format.

"""

#%% Imports and constants

import os
import json
import pandas as pd
import sys
import argparse

from copy import deepcopy
from collections import defaultdict
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key, invert_dictionary # noqa
from megadetector.utils.path_utils import recursive_file_list

from megadetector.postprocessing.validate_batch_results import \
    ValidateBatchResultsOptions, validate_batch_results

default_mewc_mount_prefix = '/images/'
default_mewc_category_name_column = 'class_id'


#%% Functions

def mewc_to_md(mewc_input_folder,
               output_file=None,
               mount_prefix=default_mewc_mount_prefix,
               category_name_column=default_mewc_category_name_column,
               mewc_out_filename='mewc_out.csv',
               md_out_filename='md_out.json'):
    """
    Converts the output of the MEWC inference scripts to the MD output format.

    Args:
        mewc_input_folder (str): the folder we'll search for MEWC output files
        output_file (str, optional): .json file to write with class information
        mount_prefix (str, optional): string to remove from all filenames in the MD
            .json file, typically the prefix used to mount the image folder.
        category_name_column (str, optional): column in the MEWC results .csv to use for
            category naming.
        mewc_out_filename (str, optional): MEWC-formatted .csv file that should be
            in [mewc_input_folder]
        md_out_filename (str, optional): MD-formatted .json file (without classification
            information) that should be in [mewc_input_folder]

    Returns:
        dict: an MD-formatted dict, the same as what's written to [output_file]
    """

    ##%% Read input files

    assert os.path.isdir(mewc_input_folder), \
        'Could not find folder {}'.format(mewc_input_folder)


    ##%% Find MEWC output files

    relative_path_to_mewc_info = {}

    print('Listing files in folder {}'.format(mewc_input_folder))
    all_files_relative = set(recursive_file_list(mewc_input_folder,return_relative_paths=True))

    for fn_relative in all_files_relative:
        if fn_relative.endswith(mewc_out_filename):
            folder_relative = '/'.join(fn_relative.split('/')[:-1])
            assert folder_relative not in relative_path_to_mewc_info
            md_output_file_relative = os.path.join(folder_relative,md_out_filename).replace('\\','/')
            assert md_output_file_relative in all_files_relative, \
                'Could not find MD output file {} to match to {}'.format(
                    md_output_file_relative,fn_relative)
            relative_path_to_mewc_info[folder_relative] = \
                {'mewc_predict_file':fn_relative,'md_file':md_output_file_relative}

    del folder_relative

    print('Found {} MEWC results files'.format(len(relative_path_to_mewc_info)))


    ##%% Prepare to loop over results files

    md_results_all = {}
    md_results_all['images'] = []
    md_results_all['detection_categories'] = {}
    md_results_all['classification_categories'] = {}
    md_results_all['info'] = None

    classification_category_name_to_id = {}


    ##%% Loop over results files

    # relative_folder = next(iter(relative_path_to_mewc_info.keys()))
    for relative_folder in relative_path_to_mewc_info:

        ##%%

        mewc_info = relative_path_to_mewc_info[relative_folder]
        mewc_csv_fn_abs = os.path.join(mewc_input_folder,mewc_info['mewc_predict_file'])
        mewc_md_fn_abs = os.path.join(mewc_input_folder,mewc_info['md_file'])

        mewc_classification_info = pd.read_csv(mewc_csv_fn_abs)
        mewc_classification_info = mewc_classification_info.to_dict('records')

        assert os.path.isfile(mewc_md_fn_abs), \
            'Could not find file {}'.format(mewc_md_fn_abs)
        with open(mewc_md_fn_abs,'r') as f:
            md_results = json.load(f)


        ##%% Remove the mount prefix from MD files if necessary
        if mount_prefix is not None and len(mount_prefix) > 0:

            n_files_without_mount_prefix = 0

            # im = md_results['images'][0]
            for im in md_results['images']:
                if not im['file'].startswith(mount_prefix):
                    n_files_without_mount_prefix += 1
                else:
                    im['file'] = im['file'].replace(mount_prefix,'',1)

            if n_files_without_mount_prefix > 0:
                print('Warning {} of {} files in the MD results did not include the mount prefix {}'.format(
                    n_files_without_mount_prefix,len(md_results['images']),mount_prefix))


        ##%% Convert MEWC snip IDs to image files

        # r = mewc_classification_info[0]
        for r in mewc_classification_info:

            # E.g. "IMG0-0.jpg"
            snip_file = r['filename']

            # E.g. "IMG0-0"
            snip_file_no_ext = os.path.splitext(snip_file)[0]
            ext = os.path.splitext(snip_file)[1] # noqa

            tokens = snip_file_no_ext.split('-')

            if len(tokens) == 1:
                print('Warning: in folder {}, detection ID not found in snip filename {}, skipping'.format(
                relative_folder,snip_file_no_ext))
                r['image_filename_without_extension'] = snip_file_no_ext
                r['snip_id'] = None

                continue

            filename_without_snip_id = '-'.join(tokens[0:-1])
            snip_id = int(tokens[-1])
            image_filename_without_extension = filename_without_snip_id

            r['image_filename_without_extension'] = image_filename_without_extension
            r['snip_id'] = snip_id

        # ...for each MEWC result record


        ##%% Make sure MD results and MEWC results refer to the same files

        images_in_md_results_no_extension = \
            set([os.path.splitext(im['file'])[0] for im in md_results['images']])
        images_in_mewc_results_no_extension = set(r['image_filename_without_extension'] \
                                                  for r in mewc_classification_info)

        # All files with classification results should also have detection results
        for fn in images_in_mewc_results_no_extension:
            assert fn in images_in_md_results_no_extension, \
                'Error: file {} is present in mewc-predict results, but not in MD results'.format(fn)

        # This is just a note to self: no classification results are present for empty images
        if False:
            for fn in images_in_md_results_no_extension:
                if fn not in images_in_mewc_results_no_extension:
                    print('Warning: file {}/{} is present in MD results, but not in mewc-predict results'.format(
                        relative_folder,fn))


        ##%% Validate images

        for im in md_results['images']:
            fn_relative = im['file']
            fn_abs = os.path.join(mewc_input_folder,relative_folder,fn_relative)
            if not os.path.isfile(fn_abs):
                print('Warning: image file {} does not exist'.format(fn_abs))


        ##%% Map filenames to MEWC results

        image_id_to_mewc_records = defaultdict(list)
        for r in mewc_classification_info:
            image_id_to_mewc_records[r['image_filename_without_extension']].append(r)


        ##%% Add classification info to MD results

        # im = md_results['images'][0]
        for im in md_results['images']:

            if ('detections' not in im) or (im['detections'] is None) or (len(im['detections']) == 0):
                continue

            detections = im['detections']

            # *Don't* sort by confidence, it looks like snip IDs use the original sort order
            # detections = sort_list_of_dicts_by_key(detections,'conf',reverse=True)

            # This is just a debug assist, so I can run this cell more than once
            for det in detections:
                det['classifications'] = []

            image_id = os.path.splitext(im['file'])[0]
            mewc_records_this_image = image_id_to_mewc_records[image_id]

            # r = mewc_records_this_image[0]
            for r in mewc_records_this_image:

                if r['snip_id'] is None:
                    continue

                category_name = r[category_name_column]

                # This is a *global* list of category mappings, across all mewc .csv files
                if category_name not in classification_category_name_to_id:
                    category_id = str(len(classification_category_name_to_id))
                    classification_category_name_to_id[category_name] = category_id
                else:
                    category_id = classification_category_name_to_id[category_name]

                snip_id = r['snip_id']
                if snip_id >= len(detections):
                    print('Warning: image {} has a classified snip ID of {}, but only {} detections are present'.format(
                        image_id,snip_id,len(detections)))
                    continue

                det = detections[snip_id]

                if 'classifications' not in det:
                    det['classifications'] = []
                det['classifications'].append([category_id,r['prob']])

            # ...for each classification in this image

        # ...for each image

        ##%% Map MD results to the global level

        if md_results_all['info'] is None:
            md_results_all['info'] = md_results['info']

        for category_id in md_results['detection_categories']:
            if category_id not in md_results_all['detection_categories']:
                md_results_all['detection_categories'][category_id] = \
                    md_results['detection_categories'][category_id]
            else:
                assert md_results_all['detection_categories'][category_id] == \
                    md_results['detection_categories'][category_id], \
                    'MD results present with incompatible detection categories'

        # im = md_results['images'][0]
        for im in md_results['images']:
            im_copy = deepcopy(im)
            im_copy['file'] = os.path.join(relative_folder,im['file']).replace('\\','/')
            md_results_all['images'].append(im_copy)

    # ...for each folder that contains MEWC results

    del md_results

    ##%% Write output

    md_results_all['classification_categories'] = invert_dictionary(classification_category_name_to_id)

    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir,exist_ok=True)
        with open(output_file,'w') as f:
            json.dump(md_results_all,f,indent=1)

        validation_options = ValidateBatchResultsOptions()
        validation_options.check_image_existence = True
        validation_options.relative_path_base = mewc_input_folder
        validation_options.raise_errors = True
        validation_results = validate_batch_results(output_file,validation_options) # noqa

# ...def mewc_to_md(...)


#%% Interactive driver

if False:

    pass

    #%%

    mewc_input_folder = r'G:\temp\mewc-test'
    mount_prefix = '/images/'
    output_file = os.path.join(mewc_input_folder,'results_with_classes.json')

    _ = mewc_to_md(mewc_input_folder=mewc_input_folder,
                   output_file=output_file,
                   mount_prefix=mount_prefix,
                   category_name_column='class_id')


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_folder',type=str,
        help='Folder containing images and MEWC .json/.csv files')
    parser.add_argument(
        'output_file',type=str,
        help='.json file where output will be written')
    parser.add_argument(
        '--mount_prefix',type=str,default=default_mewc_mount_prefix,
        help='prefix to remove from each filename in MEWC results, typically the Docker mount point')
    parser.add_argument(
        '--category_name_column',type=str,default=default_mewc_category_name_column,
        help='column in the MEWC .csv file to use for category names')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    _ = mewc_to_md(mewc_input_folder=args.input_folder,
                   output_file=args.output_file,
                   mount_prefix=args.mount_prefix,
                   category_name_column=args.category_name_column)

if __name__ == '__main__':
    main()
