########
#
# add_locations_to_nacti.py
#
# As of 10.2023, NACTI metadata only has very coarse location information (e.g. "Florida"),
# but camera IDs are embedded in filenames.  This script pulls that information from filenames
# and adds it to metadata.
#
########

#%% Imports and constants

import os
import json
import shutil

from tqdm import tqdm
from collections import defaultdict

input_file = r'd:\lila\nacti\nacti_metadata.json.1.13\nacti_metadata.json'
output_file = r'g:\temp\nacti_metadata.1.14.json'


#%% Read metadata

with open(input_file,'r') as f:
    d = json.load(f)
    
assert d['info']['version'] == 1.13


#%% Map images to locations (according to the metadata)

file_name_to_original_location = {}

# im = dataset_labels['images'][0]
for im in tqdm(d['images']):
    file_name_to_original_location[im['file_name']] = im['location']

original_locations = set(file_name_to_original_location.values())

print('Found {} locations in the original metadata:'.format(len(original_locations)))
for loc in original_locations:
    print('[{}]'.format(loc))


#%% Map images to new locations

def path_to_location(relative_path):
    
    relative_path = relative_path.replace('\\','/')
    if relative_path in file_name_to_original_location:
        location_name = file_name_to_original_location[relative_path]
        if location_name == 'San Juan Mntns, Colorado':
            # "part0/sub000/2010_Unit150_Ivan097_img0003.jpg"
            tokens = relative_path.split('/')[-1].split('_')
            assert tokens[1].startswith('Unit')
            location_name = 'sanjuan_{}_{}_{}'.format(tokens[0],tokens[1],tokens[2])
        elif location_name == 'Lebec, California':
            # "part0/sub035/CA-03_08_13_2015_CA-03_0009738.jpg"
            tokens = relative_path.split('/')[-1].split('_')
            assert tokens[0].startswith('CA-') or tokens[0].startswith('TAG-')
            location_name = 'lebec_{}'.format(tokens[0])   
        elif location_name == 'Archbold, FL':
            # "part1/sub110/FL-01_01_25_2016_FL-01_0040421.jpg"
            tokens = relative_path.split('/')[-1].split('_')
            assert tokens[0].startswith('FL-')
            location_name = 'archbold_{}'.format(tokens[0])   
        else:
            assert location_name == ''
            tokens = relative_path.split('/')[-1].split('_')
            if tokens[0].startswith('CA-') or tokens[0].startswith('TAG-') or tokens[0].startswith('FL-'):
                location_name = '{}'.format(tokens[0])
            
    else:
        
        location_name = 'unknown'    
    
    # print('Returning location {} for file {}'.format(location_name,relative_path))          

    return location_name

file_name_to_updated_location = {}
updated_location_to_count = defaultdict(int)
for im in tqdm(d['images']):
    
    updated_location = path_to_location(im['file_name'])
    file_name_to_updated_location[im['file_name']] = updated_location
    updated_location_to_count[updated_location] += 1

updated_location_to_count = {k: v for k, v in sorted(updated_location_to_count.items(), 
                         key=lambda item: item[1],
                         reverse=True)}

updated_locations = set(file_name_to_updated_location.values())

print('Found {} updated locations in the original metadata:'.format(len(updated_locations)))
for loc in updated_location_to_count:
    print('{}: {}'.format(loc,updated_location_to_count[loc]))


#%% Re-write metadata

for im in d['images']:
    im['location'] = file_name_to_updated_location[im['file_name']]
d['info']['version'] = 1.14

with open(output_file,'w') as f:
    json.dump(d,f,indent=1)
    

#%% For each location, sample some random images to make sure they look consistent

input_base = r'd:\lila\nacti-unzipped'
assert os.path.isdir(input_base)

location_to_images = defaultdict(list)

for im in d['images']:
    location_to_images[im['location']].append(im)
    
n_to_sample = 10
import random
random.seed(0)
sampling_folder_base = r'g:\temp\nacti_samples'

for location in tqdm(location_to_images):
    
    images_this_location = location_to_images[location]
    if len(images_this_location) > n_to_sample:
        images_this_location = random.sample(images_this_location,n_to_sample)
        
    for i_image,im in enumerate(images_this_location):
        
        fn_relative = im['file_name']
        source_fn_abs = os.path.join(input_base,fn_relative)
        assert os.path.isfile(source_fn_abs)
        ext = os.path.splitext(fn_relative)[1]            
        target_fn_abs = os.path.join(sampling_folder_base,'{}/{}'.format(
            location,'image_{}{}'.format(str(i_image).zfill(2),ext)))
        os.makedirs(os.path.dirname(target_fn_abs),exist_ok=True)
        shutil.copyfile(source_fn_abs,target_fn_abs)
        
    # ...for each image
    
# ...for each location
            
