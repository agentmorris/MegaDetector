"""

 test_lila_metadata_urls.py

 Test that all the metadata URLs for LILA camera trap datasets are valid, and 
 test that at least one image within each URL is valid, including MegaDetector results
 files.

"""

#%% Constants and imports

import json
import os

from data_management.lila.lila_common import read_lila_metadata,\
    read_metadata_file_for_dataset, read_lila_taxonomy_mapping

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

output_dir = os.path.join(lila_local_base,'lila_metadata_tests')
os.makedirs(output_dir,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

md_results_dir = os.path.join(lila_local_base,'md_results')
os.makedirs(md_results_dir,exist_ok=True)

md_results_keys = ['mdv4_results_raw','mdv5a_results_raw','mdv5b_results_raw','md_results_with_rde']


#%% Load category and taxonomy files

taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)


#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)

print('Loaded metadata URLs for {} datasets'.format(len(metadata_table)))


#%% Download and extract metadata and MD results for each dataset

for ds_name in metadata_table.keys():    

    metadata_table[ds_name]['json_filename'] = read_metadata_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)
    for k in md_results_keys:
        md_results_url = metadata_table[ds_name][k]
        if md_results_url is None:
            metadata_table[ds_name][k + '_filename'] = None
        else:
            metadata_table[ds_name][k + '_filename'] = read_metadata_file_for_dataset(ds_name=ds_name,
                                                                        metadata_dir=md_results_dir,
                                                                        json_url=md_results_url)


#%% Build up a list of URLs to test

url_to_source = {}

# The first image in a dataset is disproportionately likely to be human (and thus 404)
image_index = 1000

# ds_name = list(metadata_table.keys())[0]
for ds_name in metadata_table.keys():
    
    if 'bbox' in ds_name:
        print('Skipping bbox dataset {}'.format(ds_name))
        continue

    print('Processing dataset {}'.format(ds_name))
    
    json_filename = metadata_table[ds_name]['json_filename']
    with open(json_filename, 'r') as f:
        data = json.load(f)

    image_base_url = metadata_table[ds_name]['image_base_url']
    assert not image_base_url.endswith('/')
    # Download a test image
    test_image_relative_path = data['images'][image_index]['file_name']
    test_image_url = image_base_url + '/' + test_image_relative_path
    
    url_to_source[test_image_url] = ds_name + ' metadata'
    
    # k = md_results_keys[2]
    for k in md_results_keys:
        k_fn = k + '_filename'
        if metadata_table[ds_name][k_fn] is not None:
            with open(metadata_table[ds_name][k_fn],'r') as f:
                md_results = json.load(f)
                im = md_results['images'][image_index]
                md_image_url = image_base_url + '/' + im['file']
                url_to_source[md_image_url] = ds_name + ' ' + k
    
# ...for each dataset


#%% Test URLs

from md_utils.url_utils import test_urls

urls_to_test = sorted(url_to_source.keys())
urls_to_test = [fn.replace('\\','/') for fn in urls_to_test]

status_codes = test_urls(urls_to_test,error_on_failure=False)

for i_url,url in enumerate(urls_to_test):
    if status_codes[i_url] != 200:
        print('Status {} for {} ({})'.format(
            status_codes[i_url],url,url_to_source[url]))
