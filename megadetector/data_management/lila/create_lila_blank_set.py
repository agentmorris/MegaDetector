"""

create_lila_blank_set.py

Create a folder of blank images sampled from LILA.  We'll aim for diversity, so less-common
locations will be oversampled relative to more common locations.  We'll also run MegaDetector
(with manual review) to remove some incorrectly-labeled, not-actually-empty images from our 
blank set.

We'll store location information for each image in a .json file, so we can split locations
into train/val in downstream tasks.

"""

#%% Constants and imports

import os
import random
import math
import json

import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse
from collections import defaultdict

from megadetector.data_management.lila.lila_common import read_lila_all_images_file
from megadetector.utils.url_utils import download_url
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.utils.path_utils import recursive_file_list

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

project_base = os.path.join(lila_local_base,'lila_blanks')

candidate_blanks_base = os.path.join(project_base,'candidate_blanks')
os.makedirs(candidate_blanks_base,exist_ok=True)

confirmed_blanks_base = os.path.join(project_base,'confirmed_blanks')
os.makedirs(confirmed_blanks_base,exist_ok=True)

md_possible_non_blanks_folder = os.path.join(project_base,'candidate_non_blanks')
os.makedirs(md_possible_non_blanks_folder,exist_ok=True)

location_to_blank_image_urls_cache_file = os.path.join(project_base,
                                                       'location_to_blank_image_urls.json')

md_results_file = os.path.join(project_base,'lila_blanks_md_results.json')

all_fn_relative_to_location_file = os.path.join(project_base,'all_fn_relative_to_location.json')
confirmed_fn_relative_to_location_file = os.path.join(project_base,'confirmed_fn_relative_to_location.json')

preferred_image_download_source = 'gcp'

# Number of concurrent download threads
n_download_threads = 20

n_blanks = 100000

random.seed(0)


#%% Download and open the giant table of image URLs and labels

# ~60 seconds to download, unzip, and open
df = read_lila_all_images_file(metadata_dir)


#%% Explore blank labels

# Original labels we're treating as blank:
blank_original_labels = (
    'empty','misfire'
)

# Some notable original labels we're *not* treating as blank:
nonblank_original_labels = (
    'unclassifiable', 'unidentifiable', 'unidentified', 'unknown', 'fire',
    'foggy lens', 'foggy weather', 'blurred', 'end', 'eye_shine', 'ignore',
    'lens obscured', 'misdirected', 'other', 'start', 'sun', 'problem',
    'tilted', 'vegetation obstruction', 'snow on lens', 'malfunction'
)

other_labels_without_common_names = (
    'car', 'motorcycle', 'vehicle'
)

common_names = sorted(list(df['common_name'].unique()), 
                      key=lambda x:str(x) if isinstance(x,float) else x)
original_labels = sorted(list(df['original_label'].unique()),
                         key=lambda x:str(x) if isinstance(x,float) else x)

# Blanks are represented as NaN in the "common_name" column (though not all NaN's are blanks)
assert '' not in common_names
assert all([s not in common_names for s in blank_original_labels])
assert all([s not in common_names for s in nonblank_original_labels])
assert np.nan in common_names

# Blanks are represented as "empty" or "misfire" in the "original_label" column
assert all([s in original_labels for s in blank_original_labels])
assert all([s in original_labels for s in nonblank_original_labels])
assert all([s in original_labels for s in other_labels_without_common_names])
assert all([s not in original_labels for s in ('','blank','none',np.nan)])


#%% Count empty labels and common names

common_names_with_empty_original_labels = set()
original_labels_with_nan_common_names = set()

common_name_to_count = defaultdict(int)
original_label_to_count = defaultdict(int)

# This loop takes ~10 mins
for i_row,row in tqdm(df.iterrows(),total=len(df)):
    
    common_name = row['common_name']
    original_label = row['original_label']
    
    if isinstance(common_name,float):
        assert np.isnan(common_name)
        original_labels_with_nan_common_names.add(original_label)
        
    common_name = str(common_name)
    
    assert isinstance(original_label,str)
    if original_label in blank_original_labels:
        common_names_with_empty_original_labels.add(common_name)
    common_name_to_count[common_name] += 1
    original_label_to_count[original_label] += 1


#%% Look at the most common labels and common names

from megadetector.utils.ct_utils import sort_dictionary_by_value
common_name_to_count = sort_dictionary_by_value(common_name_to_count,reverse=True)
original_label_to_count = sort_dictionary_by_value(original_label_to_count,reverse=True)

k = 10

print('\nMost frequent common names:\n')

i_label = 0
for i_label,s in enumerate(common_name_to_count):
    if i_label >= k:
        break
    print('{}: {}'.format(s,common_name_to_count[s]))

print('\nMost frequent original labels:\n')

i_label = 0
for i_label,s in enumerate(original_label_to_count):
    if i_label >= k:
        break
    print('{}: {}'.format(s,original_label_to_count[s]))


#%% Do some consistency checks over the empty labels and stats

# All images called 'empty' should have NaN as their common name
assert (len(common_names_with_empty_original_labels) == 1)
assert next(iter(common_names_with_empty_original_labels)) == 'nan'

# 'empty' should be the most frequent original label overall
assert next(iter(original_label_to_count)) == 'empty'

# NaN should be the most frequent common name overall
assert next(iter(common_name_to_count)) == 'nan'

for s in original_labels_with_nan_common_names:
    assert \
        (s in blank_original_labels) or \
        (s in nonblank_original_labels) or \
        (s in other_labels_without_common_names)


#%% Map locations to blank images

force_map_locations = False

# Load from .json if available
if (not force_map_locations) and (os.path.isfile(location_to_blank_image_urls_cache_file)):
    
    with open(location_to_blank_image_urls_cache_file,'r') as f:
        location_to_blank_image_urls = json.load(f)

else:
    
    location_to_blank_image_urls = defaultdict(list)
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in tqdm(df.iterrows(),total=len(df)):
        
        location_id = row['location_id']
        url = row['url']
        
        original_label = row['original_label']
        if original_label in blank_original_labels:
            assert np.isnan(row['common_name'])
            location_to_blank_image_urls[location_id].append(url)

    with open(location_to_blank_image_urls_cache_file,'w') as f:
        json.dump(location_to_blank_image_urls,f,indent=1)

n_locations_with_blanks = len(location_to_blank_image_urls)
print('Found {} locations with blank images'.format(n_locations_with_blanks))

    
#%% Sample blanks

random.seed(0)

# Make a fresh copy of the lists
location_to_unsampled_blank_image_urls = {}

# location = next(iter(location_to_blank_image_urls.keys()))
for location in location_to_blank_image_urls:
    blank_image_urls_this_location = location_to_blank_image_urls[location]
    unsampled_blank_image_urls_this_location = blank_image_urls_this_location.copy()
    location_to_unsampled_blank_image_urls[location] = unsampled_blank_image_urls_this_location
    
# Put locations in a random order
location_ids = list(location_to_unsampled_blank_image_urls.keys())
random.shuffle(location_ids)

blank_urls = []
location_to_sampled_blanks = defaultdict(list)
fully_sampled_locations = set()

# Pick from each location until we hit our limit or have no blanks left
while(True):
    
    found_sample = False
    
    # location = location_ids[0]
    for location in location_ids:
        
        unsampled_images_this_location = location_to_unsampled_blank_image_urls[location]
        if len(unsampled_images_this_location) == 0:
            fully_sampled_locations.add(location)
            continue
        
        url = random.choice(unsampled_images_this_location)
        blank_urls.append(url)        
        location_to_unsampled_blank_image_urls[location].remove(url)
        location_to_sampled_blanks[location].append(url)
        found_sample = True
        
        if len(blank_urls) == n_blanks:
            break
        
    # ...for each location
    
    if not found_sample:
        print('Terminating after {} blanks, we ran out before hitting {}'.format(
            len(blank_urls),n_blanks))
        
    if len(blank_urls) == n_blanks:
        break

# ...while(True)

assert len(blank_urls) <= n_blanks
min_blanks_per_location = math.floor(n_blanks/n_locations_with_blanks)
max_blanks_per_location = -1
for location in location_to_sampled_blanks:
    n_blanks_this_location = len(location_to_sampled_blanks[location])
    if n_blanks_this_location >= max_blanks_per_location:
        max_blanks_per_location = n_blanks_this_location
    assert (location in fully_sampled_locations) or \
        n_blanks_this_location >= min_blanks_per_location

print('Choose {} blanks from {} locations'.format(n_blanks,len(location_ids)))
print('Fully sampled {} locations'.format(len(fully_sampled_locations)))
print('Max samples per location: {}'.format(max_blanks_per_location))
    

#%% Download those image files (prep)

container_to_url_base = {
                         'lilawildlife.blob.core.windows.net':'/lila-wildlide/',
                         'storage.googleapis.com':'/public-datasets-lila/'
                         }

def download_relative_filename(url, output_base, verbose=False, url_base=None, overwrite=False):
    """
    Download a URL to output_base, preserving relative path
    """
    
    result = {'status':'unknown','url':url,'destination_filename':None}
    
    if url_base is None:
        assert url.startswith('https://')
        container = url.split('/')[2]
        assert container in container_to_url_base
        url_base = container_to_url_base[container]
    
    assert url_base.startswith('/') and url_base.endswith('/')
    
    p = urlparse(url)
    relative_filename = str(p.path)
    # remove the leading '/'
    assert relative_filename.startswith(url_base)
    relative_filename = relative_filename.replace(url_base,'',1)        
    
    destination_filename = os.path.join(output_base,relative_filename)
    result['destination_filename'] = destination_filename
    
    if ((os.path.isfile(destination_filename)) and (not overwrite)):
        result['status'] = 'skipped'
        return result
    try:
        download_url(url, destination_filename, verbose=verbose)
    except Exception as e:
        print('Warning: error downloading URL {}: {}'.format(
            url,str(e)))     
        result['status'] = 'error: {}'.format(str(e))
        return result
    
    result['status'] = 'success'
    return result


def azure_url_to_gcp_http_url(url,error_if_not_azure_url=True):
    """
    Most URLs point to Azure by default, but most files are available on both Azure and GCP.
    This function converts an Azure URL to the corresponding GCP http:// url.
    """
    
    lila_azure_storage_account = 'https://lilawildlife.blob.core.windows.net'
    gcp_bucket_api_url = 'https://storage.googleapis.com/public-datasets-lila'
    error_if_not_azure_url = False
    
    if error_if_not_azure_url:
        assert url.startswith(lila_azure_storage_account)
    gcp_url = url.replace(lila_azure_storage_account,gcp_bucket_api_url,1)
    return gcp_url

# Convert Azure URLs to GCP URLs if necessary
if preferred_image_download_source != 'azure':
    assert preferred_image_download_source == 'gcp'
    blank_urls = [azure_url_to_gcp_http_url(url) for url in blank_urls]    


#%% Download those image files (execution)

print('Downloading {} images on {} workers'.format(len(blank_urls),n_download_threads))

if n_download_threads <= 1:

    results = []
    
    # url = all_urls[0]
    for url in tqdm(blank_urls):        
        results.append(download_relative_filename(url,candidate_blanks_base,url_base=None))
    
else:

    pool = ThreadPool(n_download_threads)        
    results = list(tqdm(pool.imap(lambda s: download_relative_filename(
        s,candidate_blanks_base,url_base=None), 
        blank_urls), total=len(blank_urls)))

# pool.terminate()


#%% Review results

error_urls = []
for r in results:
    if r['status'] != 'success':
        error_urls.append(r['url'])

print('Errors on {} of {} downloads'.format(len(error_urls),len(results)))


#%% Run MegaDetector on the folder

cmd = 'python run_detector_batch.py MDV5A "{}" "{}"'.format(
    candidate_blanks_base,md_results_file)
cmd += ' --recursive --output_relative_filenames'

import clipboard; clipboard.copy(cmd); print(cmd)


#%% Review MD results that suggests images are non-empty

assert os.path.isfile(md_results_file)

category_name_to_threshold = {'animal':0.25,'person':0.25,'vehicle':0.25}
min_threshold = min(category_name_to_threshold.values())
with open(md_results_file,'r') as f:
    md_results = json.load(f)

images_to_review_to_detections = {}

category_id_to_threshold = {}
for category_id in md_results['detection_categories']:
    category_name = md_results['detection_categories'][category_id]
    category_id_to_threshold[category_id] = category_name_to_threshold[category_name]

# im = md_results['images'][0]
for im in md_results['images']:
    
    if 'detections' not in im:
        continue
    
    found_object = False    
    for det in im['detections']:
        threshold = category_id_to_threshold[det['category']]
        if det['conf'] >= threshold:
            found_object = True
            break
    if found_object:
        images_to_review_to_detections[im['file']] = im['detections']

print('Flagging {} of {} images for review'.format(len(images_to_review_to_detections),len(md_results['images'])))

output_file_to_source_file = {}

# i_fn = 0; source_file_relative = images_to_review[i_fn]
for i_fn,source_file_relative in tqdm(enumerate(images_to_review_to_detections),
                                      total=len(images_to_review_to_detections)):
    
    source_file_abs = os.path.join(candidate_blanks_base,source_file_relative)
    assert os.path.isfile(source_file_abs)
    ext = os.path.splitext(source_file_abs)[1]
    target_file_relative = str(i_fn).zfill(8) + ext
    target_file_abs = os.path.join(md_possible_non_blanks_folder,target_file_relative)
    output_file_to_source_file[target_file_relative] = source_file_relative
    # shutil.copyfile(source_file_abs,target_file_abs)
    vis_utils.draw_bounding_boxes_on_file(input_file=source_file_abs,
                                          output_file=target_file_abs, 
                                          detections=images_to_review_to_detections[source_file_relative], 
                                          confidence_threshold=min_threshold,
                                          target_size=(1280,-1))

# This is a temporary file I just used during debugging
with open(os.path.join(project_base,'output_file_to_source_file.json'),'w') as f:
    json.dump(output_file_to_source_file,f,indent=1)
    
    
#%% Manual review

# Delete images that are *not* empty


#%% Figure out which images are still there; these are the actually-blank ones

remaining_images = set(os.listdir(md_possible_non_blanks_folder))
print('Kept {} of {} candidate blank images'.format(len(remaining_images),
                                                    len(images_to_review_to_detections)))

removed_blank_images_relative = []

# output_file = next(iter(output_file_to_source_file.keys()))
for output_file in tqdm(output_file_to_source_file.keys()):
    if output_file not in remaining_images:
        source_file_relative = output_file_to_source_file[output_file]
        removed_blank_images_relative.append(source_file_relative)
        
removed_blank_images_relative_set = set(removed_blank_images_relative)
assert len(removed_blank_images_relative) + len(remaining_images) == len(output_file_to_source_file)


#%% Copy only the confirmed blanks to the confirmed folder

from megadetector.utils.path_utils import is_image_file

all_candidate_blanks = recursive_file_list(candidate_blanks_base,return_relative_paths=True)
print('Found {} candidate blanks'.format(len(all_candidate_blanks)))

skipped_images_relative = []
skipped_non_images = []

for source_fn_relative in tqdm(all_candidate_blanks):
    
    # Skip anything we removed from the "candidate non-blanks" folder; these weren't really
    # blank.
    if source_fn_relative in removed_blank_images_relative_set:
        skipped_images_relative.append(source_fn_relative)
        continue
    
    if not is_image_file(source_fn_relative):
        # Not a typo; "skipped images" really means "skipped files"
        skipped_images_relative.append(source_fn_relative)
        skipped_non_images.append(source_fn_relative)
    
    
    source_fn_abs = os.path.join(candidate_blanks_base,source_fn_relative)
    assert os.path.isfile(source_fn_abs)
    target_fn_abs = os.path.join(confirmed_blanks_base,source_fn_relative)
    os.makedirs(os.path.dirname(target_fn_abs),exist_ok=True)
    # shutil.copyfile(source_fn_abs,target_fn_abs)

print('Skipped {} files ({} non-image files)'.format(len(skipped_images_relative),
                                                     len(skipped_non_images)))


#%% Validate the folder of confirmed blanks

from megadetector.utils.path_utils import find_images
# all_confirmed_blanks = recursive_file_list(confirmed_blanks_base,return_relative_paths=True)
all_confirmed_blanks = find_images(confirmed_blanks_base,return_relative_paths=True,recursive=True)
assert len(all_confirmed_blanks) < len(all_candidate_blanks)
print('Found {} confirmed blanks'.format(len(all_confirmed_blanks)))


#%% Manually review a few of the images we skipped

# ...to make sure they're non-blank
i_image = random.randint(0, len(skipped_images_relative))
fn_relative = skipped_images_relative[i_image]
fn_abs = os.path.join(candidate_blanks_base,fn_relative)
assert os.path.isfile(fn_abs)
import clipboard
clipboard.copy('feh --scale-down "{}"'.format(fn_abs))


#%% Record location information for each confirmed file

# Map every URL's path to the corresponding location
#
# This is *all empty URLs*, not just the ones we downloaded
all_fn_relative_to_location = {}

# location = next(iter(location_to_blank_image_urls.keys()))
for location in tqdm(location_to_blank_image_urls):
    urls_this_location = location_to_blank_image_urls[location]
    
    # url = urls_this_location[0]
    for url in urls_this_location:
        # Turn:
        # 
        # https://lilablobssc.blob.core.windows.net/caltech-unzipped/cct_images/5968c0f9-23d2-11e8-a6a3-ec086b02610b.jpg'
        #
        # ...into:
        #
        # caltech-unzipped/cct_images/5968c0f9-23d2-11e8-a6a3-ec086b02610b.jpg'   
        p = urlparse(url)
        fn_relative = str(p.path)[1:]
        all_fn_relative_to_location[fn_relative] = location

# Build a much smaller mapping of just the confirmed blanks
confirmed_fn_relative_to_location = {}        
for i_fn,fn_relative in tqdm(enumerate(all_confirmed_blanks),total=len(all_confirmed_blanks)):
    confirmed_fn_relative_to_location[fn_relative] = all_fn_relative_to_location[fn_relative]

with open(all_fn_relative_to_location_file,'w') as f:
    json.dump(all_fn_relative_to_location,f,indent=1)
    
with open(confirmed_fn_relative_to_location_file,'w') as f:
    json.dump(confirmed_fn_relative_to_location,f,indent=1)    
