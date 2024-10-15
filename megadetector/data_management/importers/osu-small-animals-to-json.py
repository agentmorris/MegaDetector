"""

Prepare the OSU Small Animals dataset for LILA release:
    
1. Convert metadata to COCO
2. Extract location, datestamp, and sequence information
3. Remove redundant or excluded images

"""

#%% Imports and constants

import os

input_folder = os.path.expanduser('~/osu-small-animals')
assert os.path.isdir(input_folder)

output_folder = os.path.expanduser('~/osu-small-animals-lila')
os.makedirs(output_folder,exist_ok=True)
output_file = os.path.join(output_folder,'osu-small-animals.json')

preview_folder = os.path.expanduser('~/osu-small-animals-preview')
os.makedirs(preview_folder,exist_ok=True)

common_to_latin_file = r'osu-small-animals-common-to-latin.txt'
assert os.path.isfile(common_to_latin_file)


#%% Support functions

def custom_relative_path_to_location(relative_path):
    
    bn = os.path.basename(relative_path).upper()

    # This only impacted six images    
    if bn.startswith('RCNX'):
        site = 'OSTN'
        return site
    
    # FCS1__2019-07-08__10-37-46(1).JPG 
    # BIWA4S2020-06-25_16-19-56.JPG
    # GRN3c__2019-05-05__01-39-23(1).JPG
    
    tokens = bn.split('_')    
    site = tokens[0]
    if '2020' in site:
        site = site.split('2020')[0]    
        
    assert len(site) <= 8
    assert site.isalnum()

    return site


#%% Read EXIF data from all images

from megadetector.data_management.read_exif import \
    ReadExifOptions, read_exif_from_folder
import json

exif_cache_file = os.path.join(input_folder,'exif_info.json')

if os.path.isfile(exif_cache_file):

    print('Reading EXIF data from cache')
    with open(exif_cache_file,'r') as f:
        exif_info = json.load(f)
        
else:
            
    read_exif_options = ReadExifOptions()
    read_exif_options.n_workers = 8
    
    exif_info = read_exif_from_folder(input_folder=input_folder,
                                      output_file=exif_cache_file,
                                      options=read_exif_options,
                                      filenames=None,
                                      recursive=True)


#%% Verify that no GPS data is present

from megadetector.data_management.read_exif import has_gps_info

missing_exif_tags = []
    
# im = exif_info[0]
for im in exif_info:
    if im['exif_tags'] is None:
        missing_exif_tags.append(im['file_name'])
        continue
    else:
        assert not has_gps_info(im)
    

#%% Read common --> latin mapping

with open(common_to_latin_file,'r') as f:
    lines = f.readlines()
    
common_to_latin = {}

# s = lines[0]
for s in lines:
    s = s.strip()
    tokens = s.split('\t')
    assert len(tokens) == 2
    common = tokens[0].lower().replace(' ','_')
    latin = tokens[1].replace('_',' ').lower()
    assert common not in common_to_latin.keys()
    assert latin not in common_to_latin.values()
    common_to_latin[common] = latin
    
    
#%% Convert non-excluded, non-split images to COCO format

from datetime import datetime

from tqdm import tqdm

# One-off typo fix
name_replacements = \
{
    'common_five-linked_skink':'common_five-lined_skink'
}

category_name_to_category = {}
# Force the empty category to be ID 0
empty_category = {}
empty_category['id'] = 0
empty_category['name'] = 'empty'
category_name_to_category['empty'] = empty_category
next_category_id = 1

images = []
annotations = []

error_images = []
excluded_images = []

# exif_im = exif_info[0]
for exif_im in tqdm(exif_info):
    
    fn_relative = exif_im['file_name']
    assert '\\' not in fn_relative    
    
    if 'Split_images' in fn_relative or 'Exclusions' in fn_relative:
        excluded_images.append(fn_relative)
        continue
    
    if 'error' in exif_im:
        assert exif_im['error'] is not None
        error_images.append(fn_relative)
        continue
    
    location_name = custom_relative_path_to_location(fn_relative)
    
    exif_tags = exif_im['exif_tags']
    
    # Convert '2021:05:27 14:42:00' to '2021-05-27 14:42:00'
    datestamp = exif_tags['DateTime']
    datestamp_tokens = datestamp.split(' ')
    assert len(datestamp_tokens) == 2
    date_string = datestamp_tokens[0]
    time_string = datestamp_tokens[1]
    assert len(date_string) == 10 and len(date_string.split(':')) == 3
    date_string = date_string.replace(':','-')
    assert len(time_string) == 8 and len(time_string.split(':')) == 3
    datestamp_string = date_string + ' ' + time_string
    datestamp_object = datetime.strptime(datestamp_string, '%Y-%m-%d %H:%M:%S')
    assert str(datestamp_object) == datestamp_string
    
    # E.g.:
    #
    # Images/Sorted_by_species/Testudines/Snapping Turtle/CBG10__2021-05-27__14-42-00(1).JPG'
    common_name = os.path.basename(os.path.dirname(fn_relative)).lower().replace(' ','_')
    
    if common_name in name_replacements:
        common_name = name_replacements[common_name]
        
    if common_name == 'blanks':
        common_name = 'empty'
    else:
        assert common_name in common_to_latin
    
    if common_name in category_name_to_category:
        
        category = category_name_to_category[common_name]
        
    else:
        
        category = {}
        category['name'] = common_name
        category['latin_name'] = common_to_latin[common_name]
        category['id'] = next_category_id
        next_category_id += 1
        category_name_to_category[common_name] = category
        
    im = {}
    im['id'] = fn_relative
    im['file_name'] = fn_relative
    im['datetime'] = datestamp_object
    im['location'] = location_name
    
    annotation = {}
    annotation['id'] = 'ann_' + fn_relative
    annotation['image_id'] = im['id']
    annotation['category_id'] = category['id']
    annotation['sequence_level_annotation'] = False
    
    images.append(im)
    annotations.append(annotation)
    
# ...for each image

cct_dict = {}
cct_dict['images'] = images
cct_dict['annotations'] = annotations
cct_dict['categories'] = list(category_name_to_category.values())

cct_dict['info'] = {}
cct_dict['info']['version'] = '2024.10.03'
cct_dict['info']['description'] = 'OSU small animals dataset'

print('\nExcluded {} of {} images ({} errors)'.format(
    len(excluded_images),
    len(exif_info),
    len(error_images)))

assert len(images) == len(exif_info) - (len(error_images) + len(excluded_images))


#%% Create sequences from timestamps

from megadetector.data_management import cct_json_utils

print('Assembling images into sequences')
cct_json_utils.create_sequences(cct_dict)

# Convert datetimes to strings so we can serialize to json
for im in cct_dict['images']:
    im['datetime'] = str(im['datetime'])

    
#%% Write COCO data

with open(output_file,'w') as f:
    json.dump(cct_dict,f,indent=1)
    

#%% Copy images (prep)

from megadetector.utils.path_utils import parallel_copy_files

input_file_to_output_file = {}

# im = cct_dict['images'][0]
for im in tqdm(cct_dict['images']):
    fn_relative = im['file_name']
    fn_source_abs = os.path.join(input_folder,fn_relative)
    assert os.path.isfile(fn_source_abs)
    fn_dest_abs = os.path.join(output_folder,fn_relative)
    assert fn_source_abs not in input_file_to_output_file
    input_file_to_output_file[fn_source_abs] = fn_dest_abs
    
    
#%% Copy images (execution)

parallel_copy_files(input_file_to_output_file, max_workers=10, 
                    use_threads=True, overwrite=False, verbose=False)


#%% Validate .json file

from megadetector.data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = input_folder
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = True

sorted_categories, data, _ = integrity_check_json_db.integrity_check_json_db(output_file, options)


#%% Preview labels

from megadetector.visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 5000
viz_options.parallelize_rendering = True
viz_options.htmlOptions['maxFiguresPerHtmlFile'] = 2500
viz_options.parallelize_rendering_with_threads = True

html_output_file, image_db = visualize_db.visualize_db(db_path=output_file,
                                                       output_dir=preview_folder,
                                                       image_base_dir=input_folder,
                                                       options=viz_options)

os.startfile(html_output_file)


#%% Print unique locations

all_locations = set()

for im in cct_dict['images']:
    all_locations.add(im['location'])

all_locations = sorted(list(all_locations))    


#%% Notes

"""
 31899 eastern_gartersnake
 14567 song_sparrow
 14169 meadow_vole
 11448 empty
 10548 white-footed_mouse
  5934 northern_house_wren
  5075 invertebrate
  5045 common_five-lined_skink
  4242 masked_shrew
  3263 eastern_cottontail
  2325 long-tailed_weasel
  1510 woodland_jumping_mouse
  1272 plains_gartersnake
  1189 eastern_massasauga
   985 virginia_opossum
   802 common_yellowthroat
   746 n._short-tailed_shrew
   529 dekay's_brownsnake
   425 american_mink
   340 american_toad
   293 eastern_racer_snake
   264 smooth_greensnake
   198 eastern_chipmunk
   193 northern_leopard_frog
   160 meadow_jumping_mouse
   155 butler's_gartersnake
   133 eastern_ribbonsnake
   121 northern_watersnake
   111 star-nosed_mole
   104 striped_skunk
    72 eastern_milksnake
    68 gray_ratsnake
    67 eastern_hog-nosed_snake
    62 raccoon
    47 green_frog
    44 woodchuck
    44 kirtland's_snake
    23 indigo_bunting
    23 painted_turtle
    13 sora
    12 american_bullfrog
    10 gray_catbird
     9 red-bellied_snake
     8 brown_rat
     6 snapping_turtle
     1 eastern_bluebird
"""
