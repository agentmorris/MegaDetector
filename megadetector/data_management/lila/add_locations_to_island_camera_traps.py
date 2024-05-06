"""

add_locations_to_island_camera_traps.py

The Island Conservation Camera Traps dataset had unique camera identifiers embedded
in filenames, but not in the proper metadata fields.  This script copies that information
to metadata.

"""

#%% Imports and constants

import os
import json
from tqdm import tqdm

input_fn = os.path.expanduser('~/lila/metadata/island_conservation.json')
output_fn = os.path.expanduser('~/tmp/island_conservation.json')
preview_folder = os.path.expanduser('~/tmp/island_conservation_preview')
image_directory = os.path.expanduser('~/data/icct/public/')


#%% Read input file

with open(input_fn,'r') as f:
    d = json.load(f)
    
d['info']
d['info']['version'] = '1.01'


#%% Find locations

images = d['images']

locations = set()

for i_image,im in tqdm(enumerate(images),total=len(images)):
    tokens_fn = im['file_name'].split('/')
    tokens_id = im['id'].split('_')
    assert tokens_fn[0] == tokens_id[0]
    assert tokens_fn[1] == tokens_id[1]
    location = tokens_fn[0] + '_' + tokens_fn[1]
    im['location'] = location
    locations.add(location)

locations = sorted(list(locations))
    
for s in locations:
    print(s)
    
    
#%% Write output file

with open(output_fn,'w') as f:
    json.dump(d,f,indent=1)
    

#%% Validate .json files

from megadetector.data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

sorted_categories, data, error_info = integrity_check_json_db.integrity_check_json_db(output_fn, options)


#%% Preview labels

from megadetector.visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['test']
html_output_file, image_db = visualize_db.visualize_db(db_path=output_fn,
                                                         output_dir=preview_folder,
                                                         image_base_dir=image_directory,
                                                         options=viz_options)

from megadetector.utils import path_utils
path_utils.open_file(html_output_file)


#%% Zip output file

from megadetector.utils.path_utils import zip_file

zip_file(output_fn, verbose=True)
assert os.path.isfile(output_fn + '.zip')
