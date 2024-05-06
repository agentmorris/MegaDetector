"""

 add_timestamps_to_icct.py

 The Island Conservation Camera Traps dataset was originally posted without timestamps
 in either .json metadata or EXIF metadata.  We pulled timestamps out using ocr_tools.py,
 this script adds those timestamps into the .json metadata.

"""

#%% Imports and constants

import json

ocr_results_file = r'g:\temp\ocr_results.2023.10.31.07.37.54.json'
input_metadata_file = r'd:\lila\islandconservationcameratraps\island_conservation.json'
output_metadata_file = r'g:\temp\island_conservation_camera_traps_1.02.json'
ocr_results_file_base = 'g:/temp/island_conservation_camera_traps/'
assert ocr_results_file_base.endswith('/')


#%% Read input metadata

with open(input_metadata_file,'r') as f:
    input_metadata = json.load(f)
    
assert input_metadata['info']['version'] == '1.01'

# im = input_metadata['images'][0]
for im in input_metadata['images']:
    assert 'datetime' not in im
    

#%% Read OCR results

with open(ocr_results_file,'r') as f:
    abs_filename_to_ocr_results = json.load(f)
    
relative_filename_to_ocr_results = {}

for fn_abs in abs_filename_to_ocr_results:
    assert ocr_results_file_base in fn_abs
    fn_relative = fn_abs.replace(ocr_results_file_base,'')
    relative_filename_to_ocr_results[fn_relative] = abs_filename_to_ocr_results[fn_abs]
    

#%% Add datetimes to metadata

images_not_in_datetime_results = []
images_with_failed_datetimes = []

for i_image,im in enumerate(input_metadata['images']):
    if im['file_name'] not in relative_filename_to_ocr_results:
        images_not_in_datetime_results.append(im)
        im['datetime'] = None
        continue
    ocr_results = relative_filename_to_ocr_results[im['file_name']]
    if ocr_results['datetime'] is None:
        images_with_failed_datetimes.append(im)
        im['datetime'] = None
        continue
    im['datetime'] = ocr_results['datetime']

print('{} of {} images were not in datetime results'.format(
    len(images_not_in_datetime_results),len(input_metadata['images'])))

print('{} of {} images were had failed datetime results'.format(
    len(images_with_failed_datetimes),len(input_metadata['images'])))

for im in input_metadata['images']:
    assert 'datetime' in im


#%% Write output

input_metadata['info']['version'] = '1.02'

with open(output_metadata_file,'w') as f:
    json.dump(input_metadata,f,indent=1)