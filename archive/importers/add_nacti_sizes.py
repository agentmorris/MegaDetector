"""

 add_nacti_sizes.py

 NACTI bounding box metadata was posted before we inclduded width and height as semi-standard
 fields; pull size information from the main metadata file and add to the bbox file.

"""

#%% Constants and environment

import json
from tqdm import tqdm

input_file = 'G:/temp/nacti_metadata.json'
input_bbox_file = 'G:/temp/nacti_20200401_bboxes.json'
output_bbox_file = 'G:/temp/nacti_20230920_bboxes.json'


#%% Read .json files

with open(input_file,'r') as f:
    input_data = json.load(f)

with open(input_bbox_file,'r') as f:
    input_bbox_data = json.load(f)
    
print('Finished reading .json data')


#%% Map image names to width and height

filename_to_size = {}
for im in tqdm(input_data['images']):
    filename_to_size[im['file_name']] = (im['width'],im['height'])
    

#%% Add to output data

for im in tqdm(input_bbox_data['images']):
    size = filename_to_size[im['file_name']]
    im['width'] = size[0]
    im['height'] = size[1]
    

#%% Write output

output_bbox_data = input_bbox_data
output_bbox_data['version'] = '2023-09-20'

with open(output_bbox_file,'w') as f:
    json.dump(output_bbox_data,f,indent=1)