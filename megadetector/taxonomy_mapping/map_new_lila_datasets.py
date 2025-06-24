"""

map_new_lila_datasets.py

Given a subset of LILA datasets, find all the categories, and start the taxonomy
mapping process.

"""

#%% Constants and imports

import os
import json

# Created by get_lila_category_list.py
input_lila_category_list_file = os.path.expanduser('~/lila/lila_categories_list/lila_dataset_to_categories.json')

output_file = os.path.expanduser('~/lila/lila_additions_2025.06.23.csv')

datasets_to_map = [
    'Nkhotakota Camera Traps'
    ]


#%% Initialize taxonomic lookup

# Takes ~2 mins

from megadetector.taxonomy_mapping.species_lookup import \
    initialize_taxonomy_lookup, get_preferred_taxonomic_match

initialize_taxonomy_lookup(force_init=False)


#%% Read the list of datasets

with open(input_lila_category_list_file,'r') as f:
    input_lila_categories = json.load(f)

lila_datasets = set()

for dataset_name in input_lila_categories.keys():
    # The script that generates this dictionary creates a separate entry for bounding box
    # metadata files, but those don't represent new dataset names, so we ignore them here.
    lila_datasets.add(dataset_name.replace('_bbox',''))

for s in datasets_to_map:
    assert s in lila_datasets


#%% Find all categories

category_mappings = []

# dataset_name = datasets_to_map[0]
for dataset_name in datasets_to_map:

    ds_categories = input_lila_categories[dataset_name]
    for category in ds_categories:
        category_name = category['name']
        assert ':' not in category_name
        mapping_name = dataset_name + ':' + category_name
        category_mappings.append(mapping_name)

print('Need to create {} mappings'.format(len(category_mappings)))


#%% Match every query against our taxonomies

output_rows = []

taxonomy_preference = 'inat'

allow_non_preferred_matches = True

# mapping_string = category_mappings[1]; print(mapping_string)
for mapping_string in category_mappings:

    tokens = mapping_string.split(':')
    assert len(tokens) == 2

    dataset_name = tokens[0]
    query = tokens[1]

    taxonomic_match = get_preferred_taxonomic_match(query,taxonomy_preference=taxonomy_preference)

    if (taxonomic_match.source == taxonomy_preference) or allow_non_preferred_matches:

        output_row = {
            'dataset_name': dataset_name,
            'query': query,
            'source': taxonomic_match.source,
            'taxonomy_level': taxonomic_match.taxonomic_level,
            'scientific_name': taxonomic_match.scientific_name,
            'common_name': taxonomic_match.common_name,
            'taxonomy_string': taxonomic_match.taxonomy_string
        }

    else:

        output_row = {
            'dataset_name': dataset_name,
            'query': query,
            'source': '',
            'taxonomy_level': '',
            'scientific_name': '',
            'common_name': '',
            'taxonomy_string': ''
        }

    output_rows.append(output_row)

# ...for each mapping


#%% Write output rows

import os
import pandas as pd

assert not os.path.isfile(output_file), 'Delete the output file before re-generating'

output_df = pd.DataFrame(data=output_rows, columns=[
    'dataset_name', 'query', 'source', 'taxonomy_level',
    'scientific_name', 'common_name', 'taxonomy_string'])
output_df.to_csv(output_file, index=None, header=True)

# from megadetector.utils.path_utils import open_file; open_file(output_file)


#%% Manual lookup

if False:

    #%% You probably want to open the .csv file first

    from megadetector.utils.path_utils import open_file
    open_file(output_file)


    #%%

    q = 'animalia'

    taxonomy_preference = 'inat'
    m = get_preferred_taxonomic_match(q,taxonomy_preference)
    # print(m.scientific_name); import clipboard; clipboard.copy(m.scientific_name)

    if (m is None) or (len(m.taxonomy_string) == 0):
        print('No match')
    else:
        if m.source != taxonomy_preference:
            print('\n*** non-preferred match ***\n')
            # raise ValueError('')
        print(m.source)
        print(m.taxonomy_string)
        import clipboard; clipboard.copy(m.taxonomy_string)
