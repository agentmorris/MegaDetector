"""

add_width_and_height_to_db.py

Grabs width and height from actual image files for a .json database that is missing w/h.

"""

#%% Imports and constants

import os
import sys
import json
import argparse

from tqdm import tqdm
from PIL import Image

from megadetector.utils import ct_utils


#%% Main resizing function

def add_width_and_height_to_db(input_file,output_file,image_base_folder):
    """
    Add width and height to images in the COCO db [input_file]
    that don't have non-None w/h values.  Does not verify correctness
    for images that already have non-None w/h values.  Ignores files that
    fail to open.

    Args:
        input_file (str): the COCO .json file to process
        output_file (str): the COCO .json file to write
        image_base_folder (str): image filenames in [input_file] should be relative
            to this folder

    Returns:
        list: the list of image dicts that were modified
    """

    with open(input_file,'r') as f:
        d = json.load(f)

    to_return = []

    for im in tqdm(d['images']):

        if ('height' not in im) or ('width' not in im) or \
           (im['height'] is None) or (im['width'] is None) or \
           (im['height'] <= 0) or (im['width'] <= 0):

            fn_relative = im['file_name']
            fn_abs = os.path.join(image_base_folder,fn_relative)

            if not os.path.isfile(fn_abs):
                print('Could not find image file {}'.format(fn_abs))
                continue

            try:
                im_w, im_h = Image.open(fn_abs).size
            except Exception as e:
                print('Error opening file {}: {}'.format(fn_abs,str(e)))
                continue

            assert isinstance(im_w,int) and isinstance(im_h,int) and \
                im_w > 0 and im_h > 0, \
                'Illegal size retrieved for {}'.format(fn_abs)

            im['height'] = im_h
            im['width'] = im_w
            to_return.append(im)

        # ...if we need to add width and/or height to this image

    # ...for each image

    ct_utils.write_json(output_file, d)

    print('Added size information to {} of {} images'.format(
        len(to_return), len(d['images'])))

    return to_return

# ...def add_width_and_height_to_db(...)


#%% Command-line driver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='Input COCO-formatted .json file')
    parser.add_argument('output_file', type=str,
                        help='Output COCO-formatted .json file')
    parser.add_argument('image_base_folder', type=str,
                        help='Base directory for images')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    add_width_and_height_to_db(args.input_file,
                               args.output_file,
                               args.image_base_folder)
