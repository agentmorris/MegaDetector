"""

get_lila_annotation_counts.py

Generates a .json-formatted dictionary mapping each LILA dataset to all categories
that exist for that dataset, with counts for the number of occurrences of each category 
(the number of *annotations* for each category, not the number of *images*).

Also loads the taxonomy mapping file, to include scientific names for each category.

get_lila_image_counts.py counts the number of *images* for each category in each dataset.

"""

#%% Constants and imports

import json
import os

from megadetector.data_management.lila.lila_common import \
    read_lila_metadata, read_metadata_file_for_dataset, read_lila_taxonomy_mapping

# cloud provider to use for downloading images; options are 'gcp', 'azure', or 'aws'
preferred_cloud = 'gcp'

# array to fill for output
category_list = []

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

output_dir = os.path.join(lila_local_base,'lila_categories_list')
os.makedirs(output_dir,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_file = os.path.join(output_dir,'lila_dataset_to_categories.json')


#%% Load category and taxonomy files

taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)


#%% Map dataset names and category names to scientific names

ds_query_to_scientific_name = {}

unmapped_queries = set()

datasets_with_taxonomy_mapping = set()

# i_row = 1; row = taxonomy_df.iloc[i_row]; row
for i_row,row in taxonomy_df.iterrows():
    
    datasets_with_taxonomy_mapping.add(row['dataset_name'])
    
    ds_query = row['dataset_name'] + ':' + row['query']
    ds_query = ds_query.lower()
    
    if not isinstance(row['scientific_name'],str):
        unmapped_queries.add(ds_query)
        ds_query_to_scientific_name[ds_query] = 'unmapped'
        continue
        
    ds_query_to_scientific_name[ds_query] = row['scientific_name']
    
print('Loaded taxonomy mappings for {} datasets'.format(len(datasets_with_taxonomy_mapping)))
    

#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)

print('Loaded metadata URLs for {} datasets'.format(len(metadata_table)))


#%% Download and extract metadata for each dataset

for ds_name in metadata_table.keys():    
    metadata_table[ds_name]['json_filename'] = read_metadata_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)


#%% Get category names and counts for each dataset

from collections import defaultdict

dataset_to_categories = {}

# ds_name = 'NACTI'
for ds_name in metadata_table.keys():
    
    taxonomy_mapping_available = (ds_name in datasets_with_taxonomy_mapping)
    
    if not taxonomy_mapping_available:
        print('Warning: taxonomy mapping not available for {}'.format(ds_name))
        
    print('Finding categories in {}'.format(ds_name))

    json_filename = metadata_table[ds_name]['json_filename']
    base_url = metadata_table[ds_name]['image_base_url_' + preferred_cloud]
    assert not base_url.endswith('/')
    
    # Open the metadata file    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # Collect list of categories and mappings to category name
    categories = data['categories']
    
    category_id_to_count = defaultdict(int)
    annotations = data['annotations']    
    
    # ann = annotations[0]
    for ann in annotations:
        category_id_to_count[ann['category_id']] = category_id_to_count[ann['category_id']] + 1
    
    # c = categories[0]
    for c in categories:
       count = category_id_to_count[c['id']] 
       if 'count' in c:
           assert 'bbox' in ds_name or c['count'] == count       
       c['count'] = count
       
       # Don't do taxonomy mapping for bbox data sets, which are sometimes just binary and are
       # always redundant with the class-level data sets.
       if 'bbox' in ds_name:
           c['scientific_name_from_taxonomy_mapping'] = None
       elif not taxonomy_mapping_available:
           c['scientific_name_from_taxonomy_mapping'] = None
       else:
           taxonomy_query_string = ds_name.lower().strip() + ':' + c['name'].lower()
           if taxonomy_query_string not in ds_query_to_scientific_name:
               print('No match for query string {}'.format(taxonomy_query_string))
               # As of right now, this is the only quirky case
               assert '#ref!' in taxonomy_query_string and 'wcs' in ds_name.lower()
               c['scientific_name_from_taxonomy_mapping'] = None
           else:
               sn = ds_query_to_scientific_name[taxonomy_query_string]
               assert sn is not None and len(sn) > 0
               c['scientific_name_from_taxonomy_mapping'] = sn
    
    dataset_to_categories[ds_name] = categories

# ...for each dataset


#%% Print the results

# ds_name = list(dataset_to_categories.keys())[0]
for ds_name in dataset_to_categories:
    
    print('\n** Category counts for {} **\n'.format(ds_name))
    
    categories = dataset_to_categories[ds_name]
    categories = sorted(categories, key=lambda x: x['count'], reverse=True)
    
    for c in categories:
        print('{} ({}): {}'.format(c['name'],c['scientific_name_from_taxonomy_mapping'],c['count']))
        
# ...for each dataset


#%% Save the results

with open(output_file, 'w') as f:
    json.dump(dataset_to_categories,f,indent=1)
