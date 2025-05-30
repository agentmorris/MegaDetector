"""

map_lila_taxonomy_to_wi_taxonomy.py

Loads the LILA category mapping (in which taxonomy information comes from an
iNat taxonomy snapshot) and tries to map each class to the Wildlife Insights taxonomy.

"""

#%% Constants and imports

import numpy as np
import json
import os

from tqdm import tqdm

from megadetector.data_management.lila.lila_common import \
    read_lila_taxonomy_mapping, read_wildlife_insights_taxonomy_mapping


#%% Prevent execution during infrastructural imports

if False:

    #%%

    lila_local_base = os.path.expanduser('~/lila')

    metadata_dir = os.path.join(lila_local_base, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    # Created by get_lila_category_list.py... contains counts for each category
    category_list_dir = os.path.join(lila_local_base, 'lila_categories_list')
    lila_dataset_to_categories_file = os.path.join(
        category_list_dir, 'lila_dataset_to_categories.json')

    # This is a manually-curated file used to store mappings that had to be made manually
    lila_to_wi_supplementary_mapping_file = os.path.expanduser(
        '~/git/MegaDetector/taxonomy_mapping/lila_to_wi_supplementary_mapping_file.csv')

    assert os.path.isfile(lila_dataset_to_categories_file)

    # This is the main output file from this whole process
    wi_mapping_table_file = os.path.join(lila_local_base,'lila_wi_mapping_table.csv')

    id_column = 'uniqueIdentifier' # 'id'


    #%% Load category and taxonomy files

    with open(lila_dataset_to_categories_file, 'r') as f:
        lila_dataset_to_categories = json.load(f)

    lila_taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)

    wi_taxonomy_df = read_wildlife_insights_taxonomy_mapping(metadata_dir)


    #%% Pull everything out of pandas

    lila_taxonomy = lila_taxonomy_df.to_dict('records')
    wi_taxonomy = wi_taxonomy_df.to_dict('records')


    #%% Cache WI taxonomy lookups

    def _is_empty_wi_item(v):
        if isinstance(v, str):
            return len(v) == 0
        elif v is None:
            return True
        else:
            assert isinstance(v, float) and np.isnan(v), 'Invalid item: {}'.format(str(v))
            return True


    def _taxonomy_items_equal(a, b):
        if isinstance(a, str) and (not isinstance(b, str)):
            return False
        if isinstance(b, str) and (not isinstance(a, str)):
            return False
        if (not isinstance(a, str)) or (not isinstance(b, str)):
            assert (a is None and b is None) or (isinstance(a, float) and isinstance(b, float))
            return True
        return a == b


    for taxon in wi_taxonomy:
        taxon['taxon_name'] = None

    from collections import defaultdict
    wi_taxon_name_to_taxa = defaultdict(list)

    # This is just a handy lookup table that we'll use to debug mismatches
    wi_common_name_to_taxon = {}

    blank_taxon_name = 'blank'
    blank_taxon = None

    animal_taxon_name = 'animal'
    animal_taxon = None

    unknown_taxon_name = 'unknown'
    unknown_taxon = None

    ignore_taxa = set(['No CV Result', 'CV Needed', 'CV Failed'])

    known_problematic_taxon_ids = ['f94e6d97-59cf-4d38-a05a-a75efdd2863b']

    human_taxa = []

    # taxon = wi_taxonomy[21653]; print(taxon)
    for taxon in tqdm(wi_taxonomy):

        taxon_name = None

        assert taxon['taxonomyType'] == 'object' or taxon['taxonomyType'] == 'biological'

        for k in taxon.keys():
            v = taxon[k]
            if isinstance(v,str):
                taxon[k] = v.strip()

        if taxon['commonNameEnglish'] in ignore_taxa:
            continue

        if isinstance(taxon['commonNameEnglish'], str):

            wi_common_name_to_taxon[taxon['commonNameEnglish'].strip(
            ).lower()] = taxon

            special_taxon = False

            # Look for keywords that don't refer to specific taxa: blank/animal/unknown
            if taxon['commonNameEnglish'].strip().lower() == blank_taxon_name:
                blank_taxon = taxon
                special_taxon = True

            elif taxon['commonNameEnglish'].strip().lower() == animal_taxon_name:
                animal_taxon = taxon
                special_taxon = True

            elif taxon['commonNameEnglish'].strip().lower() == unknown_taxon_name:
                unknown_taxon = taxon
                special_taxon = True

            if special_taxon:
                taxon_name = taxon['commonNameEnglish'].strip().lower()
                taxon['taxon_name'] = taxon_name
                wi_taxon_name_to_taxa[taxon_name].append(taxon)
                continue

        # Do we have a species name?
        if not _is_empty_wi_item(taxon['species']):

            # If 'species' is populated, 'genus' should always be populated; one item currently breaks
            # this rule.
            assert not _is_empty_wi_item(taxon['genus'])

            taxon_name = (taxon['genus'].strip() + ' ' +
                        taxon['species'].strip()).strip().lower()
            assert not _is_empty_wi_item(taxon['class']) and \
                not _is_empty_wi_item(taxon['order']) and \
                not _is_empty_wi_item(taxon['family'])

        elif not _is_empty_wi_item(taxon['genus']):

            assert not _is_empty_wi_item(taxon['class']) and \
                not _is_empty_wi_item(taxon['order']) and \
                not _is_empty_wi_item(taxon['family'])
            taxon_name = taxon['genus'].strip().lower()

        elif not _is_empty_wi_item(taxon['family']):

            assert not _is_empty_wi_item(taxon['class']) and \
                not _is_empty_wi_item(taxon['order'])
            taxon_name = taxon['family'].strip().lower()

        elif not _is_empty_wi_item(taxon['order']):

            assert not _is_empty_wi_item(taxon['class'])
            taxon_name = taxon['order'].strip().lower()

        elif not _is_empty_wi_item(taxon['class']):

            taxon_name = taxon['class'].strip().lower()

        if taxon_name is not None:
            assert taxon['taxonomyType'] == 'biological'
        else:
            assert taxon['taxonomyType'] == 'object'
            taxon_name = taxon['commonNameEnglish'].strip().lower()

        if taxon_name in wi_taxon_name_to_taxa:
            if taxon[id_column] in known_problematic_taxon_ids:
                print('Skipping problematic taxon ID {}'.format(taxon[id_column]))
            else:
                previous_taxa = wi_taxon_name_to_taxa[taxon_name]
                for previous_taxon in previous_taxa:
                    for level in ['class', 'order', 'family', 'genus', 'species']:
                        error_string = 'Error: taxon {} appeared previously in {} {} (as {}), now in {} {}'.format(
                            taxon_name,
                            level,previous_taxon[level],
                            previous_taxon['taxon_name'],
                            level,taxon[level])
                        assert _taxonomy_items_equal(previous_taxon[level], taxon[level]), error_string

        taxon['taxon_name'] = taxon_name
        if taxon_name == 'homo sapiens':
            human_taxa.append(taxon)
        wi_taxon_name_to_taxa[taxon_name].append(taxon)

    # ...for each taxon

    assert unknown_taxon is not None
    assert animal_taxon is not None
    assert blank_taxon is not None


    #%% Find redundant taxa

    taxon_names_with_multiple_entries = []
    for wi_taxon_name in wi_taxon_name_to_taxa:
        if len(wi_taxon_name_to_taxa[wi_taxon_name]) > 1:
            taxon_names_with_multiple_entries.append(wi_taxon_name)

    print('{} names have multiple entries\n:'.format(len(taxon_names_with_multiple_entries)))

    for s in taxon_names_with_multiple_entries:
        print(s)

    if False:
        pass

        #%% Manual review of redundant taxa

        s = taxon_names_with_multiple_entries[15]
        taxa = wi_taxon_name_to_taxa[s]
        for t in taxa:
            for k in t.keys():
                print('{}: {}'.format(k,t[k]))
            print()
            # print(t,end='\n\n')


    #%% Clean up redundant taxa

    taxon_name_to_preferred_taxon_id = {}

    # "helmeted guineafowl" vs "domestic guineafowl"
    taxon_name_to_preferred_taxon_id['numida meleagris'] = '83133617-8358-4910-82ee-4c23e40ba3dc' # 2005826

    # "domestic turkey" vs. "wild turkey"
    taxon_name_to_preferred_taxon_id['meleagris gallopavo'] = 'c10547c3-1748-48bf-a451-8066c820f22f' # 2021598

    # multiple sensible human entries
    taxon_name_to_preferred_taxon_id['homo sapiens'] = '990ae9dd-7a59-4344-afcb-1b7b21368000' # 2002045

    # "domestic dog" and "dog-on-leash"
    taxon_name_to_preferred_taxon_id['canis familiaris'] = '3d80f1d6-b1df-4966-9ff4-94053c7a902a' # 2021548

    # "small mammal" vs. "mammal"
    taxon_name_to_preferred_taxon_id['mammalia'] = 'f2d233e3-80e3-433d-9687-e29ecc7a467a' # 2021108

    # "Hispaniolan Mango" vs. NaN
    taxon_name_to_preferred_taxon_id['anthracothorax dominicus'] = 'f94e6d97-59cf-4d38-a05a-a75efdd2863b'

    # "millipedes" vs. "Millipede"
    taxon_name_to_preferred_taxon_id['diplopoda'] =  '065884eb-4e64-4233-84dc-de25bd06ffd2' # 2021760

    # Different suborders: Squamata vs. Lacertilia
    taxon_name_to_preferred_taxon_id['squamata'] = '710c4066-bd5d-4313-bcf4-0217c4c84da7' # 2021703

    # Redundancy (both "beautiful firetail")
    taxon_name_to_preferred_taxon_id['stagonopleura bella'] = '7fec8e7e-fd3b-4d7f-99fd-3ade6f3bbaa5' # 2021939

    # "yellow wagtail" vs. "yellow crowned-wagtail"
    taxon_name_to_preferred_taxon_id['motacilla flava'] = 'ac6669bc-9f9e-4473-b609-b9082f9bf50c' # 2016194

    # "dremomys species" vs. "dremomys genus"
    taxon_name_to_preferred_taxon_id['dremomys'] = '1507d153-af11-46f1-bfb8-77918d035ab3' # 2019370

    # "elk" vs. "domestic elk"
    taxon_name_to_preferred_taxon_id['cervus canadensis'] = 'c5ce946f-8f0d-4379-992b-cc0982381f5e'

    # "American bison" vs. "domestic bison"
    taxon_name_to_preferred_taxon_id['bison bison'] = '539ebd55-081b-429a-9ae6-5a6a0f6999d4' # 2021593

    # "woodrat or rat or mouse species" vs. "mouse species"
    taxon_name_to_preferred_taxon_id['muridae'] = 'e7503287-468c-45af-a1bd-a17821bb62f2' # 2021642

    # both "southern sand frog"
    taxon_name_to_preferred_taxon_id['tomopterna adiastola'] = 'a5dc63cb-41be-4090-84a7-b944b16dcee4' # 2021834

    # sericornis species vs. scrubwren species
    taxon_name_to_preferred_taxon_id['sericornis'] = 'ad82c0ac-df48-4028-bf71-d2b2f4bc4129' # 2021776


    # taxon_name = list(taxon_name_to_preferred_taxon_id.keys())[0]
    for taxon_name in taxon_name_to_preferred_taxon_id.keys():

        candidate_taxa = wi_taxon_name_to_taxa[taxon_name]

        # If we've gotten this far, we should be choosing from multiple taxa.
        #
        # This will become untrue if any of these are resolved later, at which point we should
        # remove them from taxon_name_to_preferred_id
        assert len(candidate_taxa) > 1, 'Only one taxon available for {}'.format(taxon_name)

        # Choose the preferred taxa
        selected_taxa = [t for t in candidate_taxa if t[id_column] == \
                        taxon_name_to_preferred_taxon_id[taxon_name]]
        assert len(selected_taxa) == 1
        wi_taxon_name_to_taxa[taxon_name] = selected_taxa

    wi_taxon_name_to_taxon = {}

    for taxon_name in wi_taxon_name_to_taxa.keys():
        taxa = wi_taxon_name_to_taxa[taxon_name]
        assert len(taxa) == 1
        wi_taxon_name_to_taxon[taxon_name] = taxa[0]


    #%% Read supplementary mappings

    with open(lila_to_wi_supplementary_mapping_file, 'r') as f:
        lines = f.readlines()

    supplementary_lila_query_to_wi_query = {}

    for line in lines:
        # Each line is [lila query],[WI taxon name],[notes]
        tokens = line.strip().split(',')
        assert len(tokens) == 3
        lila_query = tokens[0].strip().lower()
        wi_taxon_name = tokens[1].strip().lower()
        assert wi_taxon_name in wi_taxon_name_to_taxa
        supplementary_lila_query_to_wi_query[lila_query] = wi_taxon_name


    #%% Map LILA categories to WI categories

    mismatches = set()
    mismatches_with_common_mappings = set()
    supplementary_mappings = set()

    all_searches = set()

    # Must be ordered from kingdom --> species
    lila_taxonomy_levels = ['kingdom', 'phylum', 'subphylum', 'superclass', 'class', 'subclass',
                            'infraclass', 'superorder', 'order', 'suborder', 'infraorder',
                            'superfamily', 'family', 'subfamily', 'tribe', 'genus', 'species']

    unknown_queries = set(
        ['unidentifiable', 'other', 'unidentified', 'unknown', 'unclassifiable'])
    blank_queries = set(['empty'])
    animal_queries = set(['animalia'])

    lila_dataset_category_to_wi_taxon = {}

    # i_taxon = 0; taxon = lila_taxonomy[i_taxon]; print(taxon)
    for i_taxon, lila_taxon in enumerate(lila_taxonomy):

        query = None

        lila_dataset_category = lila_taxon['dataset_name'] + ':' + lila_taxon['query']

        # Go from kingdom --> species, choosing the lowest-level description as the query
        for level in lila_taxonomy_levels:
            if isinstance(lila_taxon[level], str):
                query = lila_taxon[level]
                all_searches.add(query)

        if query is None:
            # E.g., 'car'
            query = lila_taxon['query']

        wi_taxon = None

        if query in unknown_queries:

            wi_taxon = unknown_taxon

        elif query in blank_queries:

            wi_taxon = blank_taxon

        elif query in animal_queries:

            wi_taxon = animal_taxon

        elif query in wi_taxon_name_to_taxon:

            wi_taxon = wi_taxon_name_to_taxon[query]

        elif query in supplementary_lila_query_to_wi_query:

            wi_taxon = wi_taxon_name_to_taxon[supplementary_lila_query_to_wi_query[query]]
            supplementary_mappings.add(query)
            # print('Made a supplementary mapping from {} to {}'.format(query,wi_taxon['taxon_name']))

        else:

            # print('No match for {}'.format(query))
            lila_common_name = lila_taxon['common_name']

            if lila_common_name in wi_common_name_to_taxon:
                wi_taxon = wi_common_name_to_taxon[lila_common_name]
                wi_common_name = wi_taxon['commonNameEnglish']
                wi_taxon_name = wi_taxon['taxon_name']
                if False:
                    print('LILA common name {} maps to WI taxon {} ({})'.format(lila_common_name,
                                                                                wi_taxon_name,
                                                                                wi_common_name))
                mismatches_with_common_mappings.add(query)

            else:

                mismatches.add(query)

        lila_dataset_category_to_wi_taxon[lila_dataset_category] = wi_taxon

    # ...for each LILA taxon

    print('Of {} entities, there are {} mismatches ({} mapped by common name) ({} mapped by supplementary mapping file)'.format(
        len(all_searches), len(mismatches), len(mismatches_with_common_mappings), len(supplementary_mappings)))

    assert len(mismatches) == 0


    #%% Manual mapping

    if not os.path.isfile(lila_to_wi_supplementary_mapping_file):
        print('Creating mapping file {}'.format(
            lila_to_wi_supplementary_mapping_file))
        with open(lila_to_wi_supplementary_mapping_file, 'w') as f:
            for query in mismatches:
                f.write(query + ',' + '\n')
    else:
        print('{} exists, not re-writing'.format(lila_to_wi_supplementary_mapping_file))


    #%% Build a dictionary from LILA dataset names and categories to LILA taxa

    lila_dataset_category_to_lila_taxon = {}

    # i_d = 0; d = lila_taxonomy[i_d]
    for i_d,d in enumerate(lila_taxonomy):
        lila_dataset_category = d['dataset_name'] + ':' + d['query']
        assert lila_dataset_category not in lila_dataset_category_to_lila_taxon
        lila_dataset_category_to_lila_taxon[lila_dataset_category] = d


    #%% Map LILA datasets to WI taxa, and count the number of each taxon available in each dataset

    with open(wi_mapping_table_file,'w') as f:

        f.write('lila_dataset_name,lila_category_name,wi_guid,wi_taxon_name,wi_common,count\n')

        # dataset_name = list(lila_dataset_to_categories.keys())[0]
        for dataset_name in lila_dataset_to_categories.keys():

            if '_bbox' in dataset_name:
                continue

            dataset_categories = lila_dataset_to_categories[dataset_name]

            # dataset_category = dataset_categories[0]
            for category in dataset_categories:

                lila_dataset_category = dataset_name + ':' + category['name'].strip().lower()
                if '#' in lila_dataset_category:
                    continue
                assert lila_dataset_category in lila_dataset_category_to_lila_taxon
                assert lila_dataset_category in lila_dataset_category_to_wi_taxon
                assert 'count' in category

                wi_taxon = lila_dataset_category_to_wi_taxon[lila_dataset_category]

                # Write out the dataset name, category name, WI GUID, WI scientific name, WI common name,
                # and count
                s = f"{dataset_name},{category['name']},{wi_taxon['uniqueIdentifier']},"+\
                    f"{wi_taxon['taxon_name']},{wi_taxon['commonNameEnglish']},{category['count']}\n"
                f.write(s)

            # ...for each category in this dataset

        # ...for each dataset

    # ...with open()
