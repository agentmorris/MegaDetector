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
# Can download from GCP (all datasets), AWS (all datasets), or Azure (most datasets).
#
########

#%% Constants and imports

import os
import random

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from collections import defaultdict

from data_management.lila.lila_common import read_lila_all_images_file, is_empty, lila_base_urls
from md_utils.url_utils import download_url

for s in lila_base_urls.values():
    assert s.endswith('/')

# If any of these strings appear in the common name of a species, we'll download that image
species_of_interest = ['grey fox','gray fox','cape fox','red fox','kit fox']

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_dir = os.path.join(lila_local_base,'lila_downloads_by_dataset')
os.makedirs(output_dir,exist_ok=True)

# Number of concurrent download threads
n_download_threads = 20

verbose = False

max_images_per_dataset = 10 # None

# This impacts the data download, but not the metadata download
#
# Setting this to "Azure" really means "Azure if available"; some datasets are
# not available on Azure.
preferred_provider = 'gcp' # 'azure', 'gcp', 'aws'

random.seed(0)


#%% Support functions

def download_relative_url(relative_url, output_base, provider='gcp', 
                          verbose=False, overwrite=False):
    """
    Download a URL to output_base, preserving the path relative to the common LILA root.
    """
    
    assert not relative_url.startswith('/')
    
    # Not all datasets are available on Azure, fall back in these cases.  The decision
    # to fall back to GCP rather than AWS is arbitrary.
    if provider == 'azure':
        nominal_provider = relative_url_to_nominal_provider[relative_url]
        if nominal_provider != 'azure':
            if verbose:
                print('URL {} not available on Azure, falling back to GCP'.format(
                    relative_url))
            provider = 'gcp'
            
    url = lila_base_urls[provider] + relative_url
    
    result = {'status':'unknown','url':url,'destination_filename':None}
    
    destination_filename = os.path.join(output_base,relative_url)
    result['destination_filename'] = destination_filename
    
    if ((os.path.isfile(destination_filename)) and (not overwrite)):
        result['status'] = 'skipped'
        return result
    try:
        download_url(url, destination_filename, verbose=verbose, force_download=overwrite)
    except Exception as e:
        print('Warning: error downloading URL {}: {}'.format(
            url,str(e)))     
        result['status'] = 'error: {}'.format(str(e))
        return result
    
    result['status'] = 'success'
    return result


#%% Download and open the giant table of image URLs and labels

# ~60 seconds to download, unzip, and open
df = read_lila_all_images_file(metadata_dir)


#%% Find all the images we want to download

# ~2 minutes

common_name_to_count = defaultdict(int)

ds_name_to_urls = defaultdict(list)

def find_items(row):
    
    if is_empty(row['common_name']):
        return
    
    match = False
    
    for species_name in species_of_interest:
        if species_name in row['common_name']:
            match = True
            common_name_to_count[species_name] += 1
            break
    
    if match:
        ds_name_to_urls[row['dataset_name']].append(row['url'])

tqdm.pandas()
_ = df.progress_apply(find_items,axis=1)

# We have a list of URLs for each dataset, flatten them all into a list of URLs
all_urls = list(ds_name_to_urls.values())
all_urls = [item for sublist in all_urls for item in sublist]
print('Found {} matching URLs across {} datasets'.format(len(all_urls),len(ds_name_to_urls)))

for common_name in common_name_to_count:
    print('{}: {}'.format(common_name,common_name_to_count[common_name]))
    
from copy import deepcopy
ds_name_to_urls_raw = deepcopy(ds_name_to_urls)


#%% Optionally trim to a fixed number of URLs per dataset

if max_images_per_dataset is None:
    pass
else:
    # ds_name = next(iter(ds_name_to_urls.keys()))
    for ds_name in ds_name_to_urls:
        if len(ds_name_to_urls[ds_name]) > max_images_per_dataset:
            ds_name_to_urls[ds_name] = random.sample(ds_name_to_urls[ds_name],max_images_per_dataset)


#%% Convert URLs to be relative to the common LILA base 

all_urls = list(ds_name_to_urls.values())
all_urls = [item for sublist in all_urls for item in sublist]

all_urls_relative = []

# Each file has a nominal URL in the .csv file.  For now, the only thing this tells is
# is that if the nominal URL isn't an Azure URL, the file isn't on Azure.  All files are on
# GCP and AWS.
#
# Keep track of the nominal provider for each URL.
relative_url_to_nominal_provider = {}

for url in all_urls:
    found_base = False
    for provider in lila_base_urls.keys():
        base = lila_base_urls[provider]
        if url.startswith(base):
            relative_url = url.replace(base,'')
            all_urls_relative.append(relative_url)
            relative_url_to_nominal_provider[relative_url] = provider
            found_base = True
            break
    assert found_base
    
assert len(all_urls) == len(all_urls_relative)


#%% Download image files

print('Downloading {} images on {} workers, preferred provider is {}'.format(
    len(all_urls),n_download_threads,preferred_provider))

if n_download_threads <= 1:

    results = []
    
    # url_relative = all_urls_relative[0]
    for url_relative in tqdm(all_urls_relative):        
        result = download_relative_url(url_relative,
                                       output_base=output_dir,
                                       provider=preferred_provider,
                                       verbose=verbose)
        results.append(result)
    
else:

    pool = ThreadPool(n_download_threads)        
    results = list(tqdm(pool.imap(lambda s: download_relative_url(
        s,output_base=output_dir,provider=preferred_provider,verbose=verbose),
        all_urls_relative), total=len(all_urls_relative)))


#%% Scrap

if False:
    
    pass

    #%% Find all the reptiles on LILA

    reptile_rows = df.loc[df['class'] == 'reptilia']
    
    # i_row = 0; row = reptile_rows.iloc[i_row]
    
    common_name_to_count = defaultdict(int)
    dataset_to_count = defaultdict(int)
    for i_row,row in reptile_rows.iterrows():
        common_name_to_count[row['common_name']] += 1
        dataset_to_count[row['dataset_name']] += 1
        
    from md_utils.ct_utils import sort_dictionary_by_value
    
    print('Found {} reptiles\n'.format(len(reptile_rows)))
    
    common_name_to_count = sort_dictionary_by_value(common_name_to_count,reverse=True)
    dataset_to_count = sort_dictionary_by_value(dataset_to_count,reverse=True)
    
    print('Common names by count:\n')
    for k in common_name_to_count:
        print('{} ({})'.format(k,common_name_to_count[k]))
    
    print('\nDatasets by count:\n')    
    for k in dataset_to_count:
        print('{} ({})'.format(k,dataset_to_count[k]))
    
