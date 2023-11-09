########
#
# wellington_to_json.py
#
# Convert the .csv file provided for the Wellington data set to a 
# COCO-camera-traps .json file
#
########

#%% Constants and environment

import pandas as pd
import os
import glob
import json
import re
import uuid
import time
import ntpath
import humanfriendly
import PIL

from tqdm import tqdm

input_metadata_file = os.path.expanduser('~/data/wct/wellington_camera_traps.csv')
output_file = os.path.expanduser('~/data/wct/wellington_camera_traps.json')
image_directory = os.path.expanduser('~/data/wct/images')
preview_dir = os.path.expanduser('~/data/wct/preview')

assert(os.path.isdir(image_directory))


#%% Read source data

input_metadata = pd.read_csv(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))

# Filenames were provided as *.jpg, but images were *.JPG, converting here
input_metadata['file'] = input_metadata['file'].apply(lambda x: x.replace('.jpg','.JPG'))

print('Converted extensions to uppercase')


#%% Map filenames to rows, verify image existence

# Takes ~30 seconds, since it's checking the existence of ~270k images

start_time = time.time()
filenames_to_rows = {}
image_filenames = input_metadata.file

duplicate_rows = []

# Build up a map from filenames to a list of rows, checking image existence as we go
for i_file,fn in enumerate(image_filenames):
    
    if (fn in filenames_to_rows):
        duplicate_rows.append(i_file)
        filenames_to_rows[fn].append(i_file)
    else:
        filenames_to_rows[fn] = [i_file]
        image_path = os.path.join(image_directory,fn)
        assert(os.path.isfile(image_path))

elapsed = time.time() - start_time
print('Finished verifying image existence in {}, found {} filenames with multiple labels'.format(
      humanfriendly.format_timespan(elapsed),len(duplicate_rows)))

# I didn't expect this to be true a priori, but it appears to be true, and
# it saves us the trouble of checking consistency across multiple occurrences
# of an image.
assert(len(duplicate_rows) == 0)    
    
    
#%% Check for images that aren't included in the metadata file

# Enumerate all images
image_full_paths = glob.glob(os.path.join(image_directory,'*.JPG'))

for i_image,image_path in enumerate(image_full_paths):
    
    fn = ntpath.basename(image_path)
    assert(fn in filenames_to_rows)

print('Finished checking {} images to make sure they\'re in the metadata'.format(
        len(image_full_paths)))


#%% Create CCT dictionaries

# Also gets image sizes, so this takes ~6 minutes
#
# Implicitly checks images for overt corruptness, i.e. by not crashing.

images = []
annotations = []

# Map categories to integer IDs (that's what COCO likes)
next_category_id = 0
categories_to_category_id = {}
categories_to_counts = {}

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

start_time = time.time()

sequence_frame_ids = set()

# image_name = image_filenames[0]
for image_name in tqdm(image_filenames):
    
    rows = filenames_to_rows[image_name]
    
    # As per above, this is convenient and appears to be true; asserting to be safe
    assert(len(rows) == 1)    
    i_row = rows[0]
    
    row = input_metadata.iloc[i_row]
    
    im = {}
    # Filenames look like "290716114012001a1116.jpg"
    im['id'] = image_name.split('.')[0]    
    im['file_name'] = image_name
    
    # This gets imported as an int64
    im['seq_id'] = str(row['sequence'])
        
    # These appear as "image1", "image2", etc.
    frame_id = row['image_sequence']
    m = re.match('^image(\d+)$',frame_id)
    assert (m is not None)
    im['frame_num'] = int(m.group(1))-1
    
    # Make sure we haven't seen this sequence before
    sequence_frame_id = im['seq_id'] + '_' + str(im['frame_num'])
    assert sequence_frame_id not in sequence_frame_ids
    sequence_frame_ids.add(sequence_frame_id)
    
    # In the form "001a"
    im['location'] = row['site']
    
    # Can be in the form '111' or 's46'
    im['camera'] = row['camera']
    
    # In the form "7/29/2016 11:40"
    im['datetime'] = row['date']
    
    # Check image height and width
    image_path = os.path.join(image_directory,fn)
    assert(os.path.isfile(image_path))
    pil_image = PIL.Image.open(image_path)
    width, height = pil_image.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
    category = row['label'].lower()
    
    # Use 'empty', to be consistent with other data on lila    
    if (category == 'nothinghere'):
        category = 'empty'
        
    # Have we seen this category before?
    if category in categories_to_category_id:
        category_id = categories_to_category_id[category]
        categories_to_counts[category] += 1
    else:
        category_id = next_category_id
        categories_to_category_id[category] = category_id
        categories_to_counts[category] = 0
        next_category_id += 1
    
    # Create an annotation
    ann = {}
    
    # The Internet tells me this guarantees uniqueness to a reasonable extent, even
    # beyond the sheer improbability of collisions.
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']    
    ann['category_id'] = category_id
    
    annotations.append(ann)
    
# ...for each image
    
# Convert categories to a CCT-style dictionary

categories = []

for category in categories_to_counts:
    print('Category {}, count {}'.format(category,categories_to_counts[category]))
    category_id = categories_to_category_id[category]
    cat = {}
    cat['name'] = category
    cat['id'] = category_id
    categories.append(cat)
    
elapsed = time.time() - start_time
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))
    

#%% Create info struct

info = {}
info['year'] = 2018
info['version'] = '1.01'
info['description'] = 'Wellington Camera Traps'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = 'Victor Anton'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data,open(output_file,'w'),indent=1)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Validate .json files

from data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

sorted_categories, data, error_info = integrity_check_json_db.integrity_check_json_db(output_file, options)


#%% Preview labels

from md_visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['test']
html_output_file, image_db = visualize_db.visualize_db(db_path=output_file,
                                                         output_dir=os.path.join(
                                                         preview_dir),
                                                         image_base_dir=image_directory,
                                                         options=viz_options)

from md_utils import path_utils
path_utils.open_file(html_output_file)
