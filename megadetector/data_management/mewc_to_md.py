"""

mewc_to_md.py

Converts the output of the MEWC inference scripts to the MD output format.

"""

#%% Imports and constants

import os
import json
import pandas as pd

from collections import defaultdict
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key, invert_dictionary

default_mewc_mount_prefix = '/images/'
default_mewc_category_name_column = 'class_id'


#%% Functions

def mewc_to_md(mewc_csv_fn,
               mewc_md_fn,
               output_file=None,
               mount_prefix=default_mewc_mount_prefix,
               image_folder=None,
               category_name_column=default_mewc_category_name_column):
    """
    
    Args:
        mewc_csv_fn (str): .csv file written by mewc-detect
        mewc_md_fn (str): md_out.json file written by mewc-detect
        output_file (str, optional): .json file to write with class information
        mount_prefix (str, optional): string to remove from all filenames in the MD 
            .json file, typically the prefix used to mount the image folder.            
        image_folder (str, optional): local folder with the images that were processed by 
            MEWC.  Only used to validate image existence.
        category_name_column (str, optional): column in the MEWC results .csv to use for
            category naming.
            
    Returns:
        dict: an MD-formatted dict, the same as what's written to [output_file]        
    """
    
    ##%% Read input files
    
    assert os.path.isfile(mewc_csv_fn), \
        'Could not find file {}'.format(mewc_csv_fn)
    mewc_classification_info = pd.read_csv(mewc_csv_fn)
    mewc_classification_info = mewc_classification_info.to_dict('records')
    
    assert os.path.isfile(mewc_md_fn), \
        'Could not find file {}'.format(mewc_md_fn)
    with open(mewc_md_fn,'r') as f:
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
        
        # E.g. "a/b/c/IMG0-0.jpg"
        snip_file = r['filename']
        
        # E.g. "IMG0-0"
        snip_file_no_ext = os.path.splitext(snip_file)[0]
        ext = os.path.splitext(snip_file)[1] # noqa
        
        tokens = snip_file_no_ext.split('-')
        assert len(tokens) > 1, 'Error: detection ID not found in snip filename {}'.format(
            snip_file_no_ext)
        
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
    
    for fn in images_in_md_results_no_extension:
        assert fn in images_in_mewc_results_no_extension, \
            'Error: file {} is present in MD results, but not in MEWC .csv'.format(fn)
    
    for fn in images_in_mewc_results_no_extension:
        assert fn in images_in_md_results_no_extension, \
            'Error: file {} is present in MEWC .csv, but not in MD results'.format(fn)
        
    
    ##%% Validate images if necessary
    
    if image_folder is not None:
        
        assert os.path.isdir(image_folder), 'Could not find image folder {}'.format(image_folder)
        
        for im in md_results['images']:
            fn_relative = im['file']
            fn_abs = os.path.join(image_folder,fn_relative)
            assert os.path.isfile(fn_abs), 'Image file {} does not exist'.format(fn_abs)
                    
    
    ##%% Map filenames to MEWC results
    
    image_id_to_mewc_records = defaultdict(list)
    for r in mewc_classification_info:
        image_id_to_mewc_records[r['image_filename_without_extension']].append(r)
    
    
    ##%% Add classification info to MD results
    
    classification_category_name_to_id = {}
    
    # im = md_results['images'][0]
    for im in md_results['images']:
        
        if ('detections' not in im) or (im['detections'] is None) or (len(im['detections']) == 0):
            continue
            
        detections = im['detections']
        detections = sort_list_of_dicts_by_key(detections,'conf',reverse=True)
        
        # This is just a debug assist, so I can run this cell more than once
        for det in detections:
            det['classifications'] = []
        
        image_id = os.path.splitext(im['file'])[0]
        mewc_records_this_image = image_id_to_mewc_records[image_id]
        
        # r = mewc_records_this_image[0]
        for r in mewc_records_this_image:
            
            category_name = r[category_name_column]
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
    
    classification_categories = invert_dictionary(classification_category_name_to_id)
    md_results['classification_categories'] = classification_categories
    
    
    ##%% Write output
    
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(md_results,f,indent=1)
                
# ...def mewc_to_md(...)


#%% Interactive driver

if False:
    
    pass

    #%%    
    
    mewc_csv_fn = r'g:\temp\D100\mewc_out.csv'
    mewc_md_fn = r'g:\temp\D100\md_out.json'
    mount_prefix = '/images/'
    image_folder = r'g:\temp\D100'
    output_file = os.path.join(image_folder,'results_with_classes.json')

    _ = mewc_to_md(mewc_csv_fn=mewc_csv_fn,
                   mewc_md_fn=mewc_md_fn,
                   output_file=output_file,
                   mount_prefix=mount_prefix,
                   image_folder=image_folder,
                   category_name_column='class_id')


#%% Command-line driver

import sys
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'mewc_csv_file',type=str,
        help='MEWC .csv filename')
    parser.add_argument(
        'mewc_md_file',type=str,
        help='MegaDetector-formatted .json file written by MEWC')
    parser.add_argument(
        'output_file',type=str,
        help='.json file where output will be written')
    parser.add_argument(
        '--mount_prefix',type=str,default=default_mewc_mount_prefix,
        help='prefix to remove from each filename in MEWC results, typically the Docker mount point')
    parser.add_argument(
        '--image_folder',type=str,default=None,
        help='local copy of the images, only used for validating filenames')
    parser.add_argument(
        '--category_name_column',type=str,default=default_mewc_category_name_column,
        help='column in the MEWC .csv file to use for category names')
        
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
        
    _ = mewc_to_md(mewc_csv_fn=args.mewc_csv_file,
                   mewc_md_fn=args.mewc_md_file,
                   output_file=args.output_file,
                   mount_prefix=args.mount_prefix,
                   image_folder=args.image_folder,
                   category_name_column=args.category_name_column)

if __name__ == '__main__':
    main()
