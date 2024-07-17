"""

test_lila_metadata_urls.py

Test that all the metadata URLs for LILA camera trap datasets are valid, including MegaDetector 
results files.

Also pick an arbitrary image from each dataset and make sure that URL is valid.

Also picks an arbitrary image from each dataset's MD results and make sure the corresponding URL is valid.

"""

#%% Constants and imports

import json
import os

from megadetector.data_management.lila.lila_common import read_lila_metadata,\
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

preferred_cloud = 'gcp' # 'azure', 'aws'


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

# Takes ~15 mins, since it has to open all the giant .json files

url_to_source = {}

# The first image in a dataset is disproportionately likely to be human (and thus 404),
# so we pick a semi-arbitrary image that isn't the first.  How about the 2000th?
image_index = 2000

# ds_name = list(metadata_table.keys())[0]
for ds_name in metadata_table.keys():
    
    if 'bbox' in ds_name:
        print('Skipping bbox dataset {}'.format(ds_name))
        continue

    print('Processing dataset {}'.format(ds_name))
    
    json_filename = metadata_table[ds_name]['json_filename']
    with open(json_filename, 'r') as f:
        data = json.load(f)

    image_base_url = metadata_table[ds_name]['image_base_url_' + preferred_cloud]
    assert not image_base_url.endswith('/')
    # Download a test image
    test_image_relative_path = data['images'][image_index]['file_name']
    test_image_url = image_base_url + '/' + test_image_relative_path
    
    url_to_source[test_image_url] = ds_name + ' metadata'
    
    # Grab an image from the MegaDetector results
    
    # k = md_results_keys[2]
    for k in md_results_keys:
        k_fn = k + '_filename'
        if metadata_table[ds_name][k_fn] is not None:
            with open(metadata_table[ds_name][k_fn],'r') as f:
                md_results = json.load(f)
                im = md_results['images'][image_index]
                md_image_url = image_base_url + '/' + im['file']
                url_to_source[md_image_url] = ds_name + ' ' + k
            del md_results
    del data
        
# ...for each dataset


#%% Test URLs

from megadetector.utils.url_utils import test_urls

urls_to_test = sorted(url_to_source.keys())
urls_to_test = [fn.replace('\\','/') for fn in urls_to_test]

status_codes = test_urls(urls_to_test,
                         error_on_failure=False,
                         pool_type='thread',
                         n_workers=10,
                         timeout=2.0)

for i_url,url in enumerate(urls_to_test):
    if status_codes[i_url] != 200:
        print('Status {} for {} ({})'.format(
            status_codes[i_url],url,url_to_source[url]))
