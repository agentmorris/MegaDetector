########
#
# download_lila_subset.py
#
# Example of how to download a list of files from LILA, e.g. all the files
# in a data set corresponding to a particular species.
#
# Organizes the downloaded images by dataset.  How you actually want to organize files,
# what you want to query for, etc., is very application-specific; this is just meant as a 
# demo.
#
# Can download from either Azure or GCP.
#
########

#%% Constants and imports

import os
import random

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse
from collections import defaultdict

from data_management.lila.lila_common import \
    read_lila_all_images_file, is_empty, azure_url_to_gcp_http_url
from md_utils.url_utils import download_url

# If any of these strings appear in the common name of a species, we'll download that image
species_of_interest = ['grey fox','red fox','leopard cat']

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_dir = os.path.join(lila_local_base,'lila_downloads_by_dataset')
os.makedirs(output_dir,exist_ok=True)

# Number of concurrent download threads
n_download_threads = 20

max_images_per_dataset = 10 # None

# This impacts the data download, but not the metadata download
image_download_source = 'azure' # 'azure' or 'gcp'

random.seed(0)


#%% Download and open the giant table of image URLs and labels

# Make take 30-60 seconds to download, unzip, and open
df = read_lila_all_images_file(metadata_dir)


#%% Find all the images we want to download

# Searching over the giant table can take a couple of minutes

ds_name_to_urls = defaultdict(list)

def find_items(row):
    
    if is_empty(row['common_name']):
        return
    
    match = False
    
    for species_name in species_of_interest:
        if species_name in row['common_name']:
            match = True
            break
    
    if match:
        ds_name_to_urls[row['dataset_name']].append(row['url'])

tqdm.pandas()
_ = df.progress_apply(find_items,axis=1)

all_urls = list(ds_name_to_urls.values())
all_urls = [item for sublist in all_urls for item in sublist]
print('Found {} matching URLs across {} datasets'.format(len(all_urls),len(ds_name_to_urls)))

from copy import deepcopy
ds_name_to_urls_raw = deepcopy(ds_name_to_urls)


#%% Trim to a fixed number of URLs per dataset

if max_images_per_dataset is None:
    pass
else:
    # ds_name = next(iter(ds_name_to_urls.keys()))
    for ds_name in ds_name_to_urls:
        if len(ds_name_to_urls[ds_name]) > max_images_per_dataset:
            ds_name_to_urls[ds_name] = random.sample(ds_name_to_urls[ds_name],max_images_per_dataset)


#%% Download those image files

def download_relative_filename(url, output_base, verbose=False, url_base=None, overwrite=False):
    """
    Download a URL to output_base, preserving relative path
    """
    
    result = {'status':'unknown','url':url,'destination_filename':None}
    
    if url_base is None:
        url_base = '/'
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


# ds_name_to_urls maps dataset names to lists of URLs; flatten to a single list of URLs
all_urls = list(ds_name_to_urls.values())
all_urls = [item for sublist in all_urls for item in sublist]

url_base = '/'

# Convert Azure URLs to GCP URLs if necessary
if image_download_source != 'azure':
    assert image_download_source == 'gcp'
    url_base = '/public-datasets-lila/'
    all_urls = [azure_url_to_gcp_http_url(url) for url in all_urls]    

print('Downloading {} images on {} workers'.format(len(all_urls),n_download_threads))

if n_download_threads <= 1:

    results = []
    
    # url = all_urls[0]
    for url in tqdm(all_urls):        
        results.append(download_relative_filename(url,output_dir,url_base=url_base))
    
else:

    pool = ThreadPool(n_download_threads)        
    results = list(tqdm(pool.imap(lambda s: download_relative_filename(
        s,output_dir,url_base=url_base), 
        all_urls), total=len(all_urls)))
