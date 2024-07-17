"""

import_desert_lion_conservation_camera_traps.py

Prepare the Desert Lion Conservation Camera Traps dataset for release on LILA.

"""

#%% Imports and constants

import os
import json

input_base_folder = r'i:/data/desert-lion'
assert os.path.isdir(input_base_folder)

# md_results_file = r'i:/data/desert-lion/desert-lion-camera-traps-2024-07-14-v5a.0.0_detections-all.json'
md_results_file = r'i:/data/desert-lion/desert-lion-camera-traps-2024-07-14-v5a.0.0_detections.json'
assert os.path.isfile(md_results_file)

export_base = os.path.join(input_base_folder,'annotated-imgs')
assert os.path.isdir(export_base)

preview_dir = r'g:\temp\desert-lion-viz'
output_file = os.path.join(input_base_folder,'desert_lion_camera_traps.json')
output_zipfile = os.path.join(input_base_folder,'desert-lion-camera-traps-images.zip')

exif_cache_file_post_exif_removal = os.path.join(input_base_folder,'exif_data_post_exif_removal.json')
exif_cache_file = os.path.join(input_base_folder,'exif_data.json')


#%% Find images and videos

from megadetector.detection.video_utils import find_videos
from megadetector.utils.path_utils import find_images

video_files = find_videos(input_base_folder,recursive=True,return_relative_paths=True,convert_slashes=True)
image_files = find_images(input_base_folder,recursive=True,return_relative_paths=True,convert_slashes=True)

n_annotated_imgs = len([fn for fn in image_files if 'annotated-imgs' in fn])
print('Found {} images ({} in the annotated-imgs folder), {} videos'.format(
    len(image_files),n_annotated_imgs,len(video_files)))


#%% Read EXIF data

from megadetector.data_management.read_exif import read_exif_from_folder, ReadExifOptions

exif_options = ReadExifOptions()
exif_options.n_workers = 10

if os.path.isfile(exif_cache_file):
    print('EXIF cache {} exists, skipping EXIF read'.format(exif_cache_file))
    with open(exif_cache_file,'r') as f:
        exif_data = json.load(f)    
else:
    exif_data = read_exif_from_folder(input_folder=input_base_folder,
                                      output_file=exif_cache_file,
                                      options=exif_options,
                                      filenames=None,
                                      recursive=True)

assert len(exif_data) == len(image_files)


#%% Remove EXIF data

from megadetector.data_management.remove_exif import remove_exif
remove_exif(input_base_folder,recursive=True,n_processes=1)


#%% Read EXIF data again

exif_data_post_exif_removal = read_exif_from_folder(input_folder=input_base_folder,
                                                    output_file=exif_cache_file_post_exif_removal,
                                                    options=exif_options,
                                                    filenames=None,
                                                    recursive=True)


#%% Make sure no lat/lon data is present

from tqdm import tqdm

for i_image,im in enumerate(tqdm(exif_data_post_exif_removal)):
    tags = im['exif_tags']
    if tags is None:
        continue
    for k in tags:
        assert 'gps' not in str(k).lower()


#%% Look for images that contain humans

with open(md_results_file,'r') as f:
    md_results = json.load(f)

assert len(md_results['images']) == len(image_files)

human_threshold = 0.1
human_categories = ['2','3']

candidate_human_images = set()
failed_images = set()

# i_image = 0; im = md_results['images'][0]
for i_image,im in tqdm(enumerate(md_results['images']),total=len(md_results['images'])):

    if 'failure' in im:
        failed_images.add(im['file'])
        continue
    
    for det in im['detections']:
        if det['category'] in human_categories and det['conf'] >= human_threshold:
            candidate_human_images.add(im['file'])
            break
        
    # ...for each detection
    
# ...for each image    

print('Found {} failed images and {} candidate human images'.format(
    len(failed_images),len(candidate_human_images)))


#%% Copy failed images and human images to a temporary folder for review

review_folder_base = r'g:/temp/review_images'
os.makedirs(review_folder_base,exist_ok=True)

images_to_review = failed_images.union(candidate_human_images)
images_to_review = sorted(list(images_to_review))

source_file_to_target_file = {}

# fn_relative = images_to_review[0]
for fn_relative in images_to_review:
    assert '\\' not in fn_relative
    fn_abs_source = input_base_folder + '/' + fn_relative
    assert os.path.isfile(fn_abs_source)
    fn_abs_dest = review_folder_base + '/' + fn_relative.replace('/','_')
    source_file_to_target_file[fn_abs_source] = fn_abs_dest

from megadetector.utils.path_utils import parallel_copy_files

parallel_copy_files(input_file_to_output_file=source_file_to_target_file,
                    max_workers=16, 
                    use_threads=True,
                    overwrite=False,verbose=False)


#%% Copy videos to a temporary folder for review

review_folder_base = r'g:/temp/review_videos'
os.makedirs(review_folder_base,exist_ok=True)

source_file_to_target_file = {}

# fn_relative = video_files[0]
for fn_relative in video_files:
    assert '\\' not in fn_relative
    fn_abs_source = input_base_folder + '/' + fn_relative
    assert os.path.isfile(fn_abs_source)
    fn_abs_dest = review_folder_base + '/' + fn_relative.replace('/','_')
    source_file_to_target_file[fn_abs_source] = fn_abs_dest

from megadetector.utils.path_utils import parallel_copy_files

parallel_copy_files(input_file_to_output_file=source_file_to_target_file,
                    max_workers=16, 
                    use_threads=True,
                    overwrite=False,verbose=False)


#%% Track removed images

removed_images = [
    "annotated-imgs\panthera leo\Camera Trap\Events\X73Okngwe\2013\02\PvL_seq_41468415-4518-44d6-acac-2113b442f723\PICT0190.JPG",
    "annotated-imgs\panthera leo\Camera Trap\Hoanib\FldPln_Arch\211011\PvL_seq_5a9c6379-6980-4ab8-903a-b3bcba2ad21b\PICT0039.JPG",
    "annotated-imgs\panthera leo\Camera Trap\Hoanib\FldPln_Arch\211011\PvL_seq_5a9c6379-6980-4ab8-903a-b3bcba2ad21b\PICT0037.JPG",
    "annotated-imgs\panthera leo\Camera Trap\Hoanib\FldPln_Arch\211011\PvL_seq_5a9c6379-6980-4ab8-903a-b3bcba2ad21b\PICT0038.JPG",
    "annotated-imgs\panthera leo\Camera Trap\2015\09\PvL_seq_da9c9ab1-74a2-485e-b6e7-3827b0c2a2f0\20150924-RCX_0835.JPG",
    "annotated-imgs\panthera leo\Camera Trap\2015\09\PvL_seq_b0c1c6c5-474e-4844-a66c-e2bf5513d47a\20150924-RCX_0841.JPG",
    "annotated-imgs\oryx gazella\Camera Trap\Video_Clips\Leylands\CDY_0003.AVI"
]

removed_images = [fn.replace('\\','/') for fn in removed_images]


#%% Map filenames to datetimes

filename_to_datetime = {}
n_valid_datetimes = 0

# im = exif_data[0]
for im in tqdm(exif_data):
    if im['exif_tags'] is None or len(im['exif_tags']) == 0:
        filename_to_datetime[im['file_name']] = None
        continue
    dt = im['exif_tags']['DateTime']
    assert len(dt) == 19
    filename_to_datetime[im['file_name']] = dt
    n_valid_datetimes += 1

print('\nFound datetime information for {} of {} images'.format(
    n_valid_datetimes,len(exif_data)))    


#%% Convert "annotated_imgs" folder to COCO Camera Traps

from megadetector.utils.path_utils import recursive_file_list

species_name_to_category_id = {}

filenames_relative = \
    recursive_file_list(export_base,return_relative_paths=True,recursive=True,convert_slashes=True)
    
short_species_names = ['aves','cn-owls','cn-francolins','cn-raptors',
                       'columbidae','equus zebra hartmannae','numididae',
                       'pteroclidae']

images = []
annotations = []
n_datetimes = 0

for fn in filenames_relative:
    
    assert fn.lower().endswith('.jpg') or fn.lower().endswith('.avi') or fn.lower().endswith('.json')
    
    if fn.lower().endswith('.json'):
        continue

    tokens = fn.split('/')
    species_name = tokens[0]
    assert species_name in short_species_names or len(species_name.split(' ')) == 2
    
    if species_name not in species_name_to_category_id:
        category_id = len(species_name_to_category_id)
        species_name_to_category_id[species_name] = category_id
    else:
        category_id = species_name_to_category_id[species_name]
                
    im = {}
    im['id'] = fn
    im['file_name'] = fn
    im['location'] = 'unknown'
    
    fn_for_datetime_lookup = 'annotated-imgs/' + fn
    if (fn_for_datetime_lookup in filename_to_datetime) and \
        (filename_to_datetime[fn_for_datetime_lookup] is not None):
        im['datetime'] = filename_to_datetime[fn_for_datetime_lookup]
        n_datetimes += 1

    ann = {}
    ann['image_id'] = im['id']
    ann['id'] = im['id'] + ':ann_00'
    ann['sequence_level_annotation'] = False
    ann['category_id'] = category_id
    
    images.append(im)
    annotations.append(ann)
    
# ...for each filename

categories = []
for species_name in species_name_to_category_id:
    category = {}
    category['name'] = species_name
    category['id'] = species_name_to_category_id[species_name]
    categories.append(category)

info = {}
info['version'] = '2024.07.15_00'
info['description'] = 'Desert Lion Camera Traps'

d = {}
d['info'] = info
d['images'] = images
d['annotations'] = annotations
d['categories'] = categories

with open(output_file,'w') as f:
    json.dump(d,f,indent=1)


#%% Integrity check

from megadetector.data_management.databases.integrity_check_json_db import \
    IntegrityCheckOptions, integrity_check_json_db
    
integrity_check_options = IntegrityCheckOptions()

integrity_check_options.baseDir = export_base
integrity_check_options.bCheckImageExistence = True
integrity_check_options.bRequireLocation = True
integrity_check_options.nThreads = 10
integrity_check_options.verbose = True
integrity_check_options.allowIntIDs = False

integrity_check_results = integrity_check_json_db(output_file,integrity_check_options)


#%% Preview    

from megadetector.visualization.visualize_db \
    import DbVizOptions, visualize_db
    
viz_options = DbVizOptions()
viz_options.num_to_visualize = 2500

html_output_file,_ = visualize_db(output_file, preview_dir, export_base, options=viz_options)

from megadetector.utils.path_utils import open_file
open_file(html_output_file)


#%% Make MD results paths line up with the output

md_results_remapped_file = md_results_file.replace('-all','')
assert md_results_remapped_file != md_results_file

with open(output_file,'r') as f:
    d = json.load(f)
    
image_filenames = [im['file_name'] for im in d['images']]
image_filenames_set = set(image_filenames)

with open(md_results_file,'r') as f:
    md_results = json.load(f)

md_results_images_remapped = []

# im = md_results['images'][0]
for im in md_results['images']:
    assert im['file'].startswith('annotated-imgs/') or im['file'].startswith('bboxes/')
    if im['file'].startswith('bboxes/'):
        continue
    im['file'] = im['file'].replace('annotated-imgs/','')
    md_results_images_remapped.append(im)
    
print('Keeping {} of {} images in MD results'.format(
    len(md_results_images_remapped),len(md_results['images'])))

d['images'] = md_results_images_remapped

with open(md_results_remapped_file,'w') as f:
    json.dump(d,f,indent=1)


#%% Zip MD results and COCO file

from megadetector.utils.path_utils import zip_file

zip_file(input_fn=md_results_remapped_file, output_fn=None, overwrite=True, verbose=True, compresslevel=9)
zip_file(input_fn=output_file, output_fn=None, overwrite=True, verbose=True, compresslevel=9)


#%% Zip images

from megadetector.utils.path_utils import zip_folder

zip_folder(input_folder=export_base, output_fn=output_zipfile, overwrite=True, verbose=True, compresslevel=0)


#%% Copy lion images to a folder for thumbnail selection

review_folder_base = r'g:/temp/thumbnail-candidates'
os.makedirs(review_folder_base,exist_ok=True)

source_file_to_target_file = {}

# fn_relative = image_files[0]
for fn_relative in image_files:
    assert '\\' not in fn_relative
    if '/lion/' not in fn_relative and '/panthera leo/' not in fn_relative:
        continue
    fn_abs_source = input_base_folder + '/' + fn_relative
    assert os.path.isfile(fn_abs_source)
    fn_abs_dest = review_folder_base + '/' + fn_relative.replace('/','_')
    source_file_to_target_file[fn_abs_source] = fn_abs_dest

from megadetector.utils.path_utils import parallel_copy_files

parallel_copy_files(input_file_to_output_file=source_file_to_target_file,
                    max_workers=16, 
                    use_threads=True,
                    overwrite=False,verbose=False)
