"""

map_lila_taxonomy_to_wi_taxonomy.py

Loads the LILA category mapping (in which taxonomy information comes from an
iNat taxonomy snapshot) and tries to map each class to the Wildlife Insights taxonomy.

What I do when I run this:

* Run all cells up to "Manual review of redundant taxa"

* Run the cell called "Go through each redundant taxon, one at a time" for each of the redundant taxa (~25 as of 2026.01), and update or create a corresponding entry in the "Decisions for redundant taxa" cell

* Run cells up to and including "Map LILA categories to WI categories", there will be errors, edit lila_to_wi_supplementary_mapping_file.csv to fix them

* Run through the last cell

* Upload lila_wi_mapping_table.csv to LILA

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
    #
    # Specifically, this maps names that appear in LILA to WI names.  Some of these are scientific
    # names getting mapped from a taxonomy level that SpeciesNet doesn't use (e.g. mapping the pteromyini
    # tribe (flying squirrel) up to the sciuridae family (squirrels)), some are various versions of
    # "unknown" (e.g. "problem", "foggy weather").
    lila_to_wi_supplementary_mapping_file = os.path.expanduser(
        '~/git/MegaDetector/megadetector/taxonomy_mapping/lila_to_wi_supplementary_mapping_file.csv')

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

            # If 'species' is populated, 'genus' should always be populated
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
                print('Skipping known-problematic taxon ID {}: {}'.format(
                    taxon[id_column],str(taxon)))
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

    print('{} names have multiple entries:\n'.format(len(taxon_names_with_multiple_entries)))

    # wi_taxon_name = taxon_names_with_multiple_entries[0]
    for wi_taxon_name in taxon_names_with_multiple_entries:

        # Look at disagreements in the different entries for this taxon name
        taxa = wi_taxon_name_to_taxa[wi_taxon_name]
        keys = set(taxa[0].keys())

        for taxon in taxa:
            assert set(taxon.keys()) == keys

        disagreements = defaultdict(list)

        for k in keys:
            if k.lower() in ('createdat','updatedat','uniqueidentifier','id','referenceUrl','authority'):
                continue
            values = set()
            for taxon in taxa:
                v = taxon[k]
                if isinstance(v,float) and np.isnan(v):
                    v = 'nan'
                values.add(v)
            # Is there more than one distinct value for this field?
            if len(values) > 1:
                disagreements[k] = values
        # ...for each field name

        print('  {}'.format(wi_taxon_name))

        for k in disagreements:
            print('    {}: {}'.format(k,str(disagreements[k])))


    #%% Manual review of redundant taxa

    if False:

        pass

        #%% Go through each redundant taxon, one at a time

        import clipboard


        i_taxon = 12

        if False:
            s = 'sericornis'
            for i_taxon,taxon_name in enumerate(taxon_names_with_multiple_entries):
                if s == taxon_name:
                    break

        s = taxon_names_with_multiple_entries[i_taxon]
        taxa = wi_taxon_name_to_taxa[s]
        for i_option,t in enumerate(taxa):
            print('Index {}:'.format(i_option))
            for k in t.keys():
                print('{}: {}'.format(k,t[k]))
            print('')
            # print(t,end='\n\n')

        i_selection = 0
        if i_selection >= len(taxa):
            i_selection = 0

        """
        # "helmeted guineafowl" vs "domestic guineafowl"
        taxon_name_to_preferred_taxon_id['numida meleagris'] = '83133617-8358-4910-82ee-4c23e40ba3dc' # 2005826
        """

        common_names = [t['commonNameEnglish'] for t in taxa]

        # Replace NaN entries in "common_names" with the string "nan"
        for i_cn in range(len(common_names)):
            if not isinstance(common_names[i_cn], str):
                common_names[i_cn] = 'nan'

        selected_common_name = common_names[i_selection]
        non_selected_names = [s for s in common_names if s != selected_common_name]
        non_selected_names = ', '.join(non_selected_names)
        output_string = '# "{}" (instead of {})'.format(selected_common_name,
                                                        non_selected_names)
        selected_entry = taxa[i_selection]
        # taxon_name_to_preferred_taxon_id['numida meleagris'] = '83133617-8358-4910-82ee-4c23e40ba3dc' # 2005826
        output_string += \
            "\ntaxon_name_to_preferred_taxon_id['{}'] = '{}' # {}".format(
                selected_entry['taxon_name'],
                selected_entry['uniqueIdentifier'],
                selected_entry['id'])

        print(output_string)
        clipboard.copy(output_string)

        #%%

    # ...if False (manual review)


    #%% Decisions for redundant taxa

    taxon_name_to_preferred_taxon_id = {}

    # "Helmeted Guineafowl" (instead of Domestic Guineafowl)
    taxon_name_to_preferred_taxon_id['numida meleagris'] = '83133617-8358-4910-82ee-4c23e40ba3dc' # 2005826

    # "Wild Turkey" (instead of Domestic Turkey)
    taxon_name_to_preferred_taxon_id['meleagris gallopavo'] = '94d8284e-6193-43b5-b80c-9029cf951675' # 2006047

    # "Human" (instead of Horseback Rider, Human-Faces, Human-Herder, Homo Species, Human-Camera Trapper, Human-Hunter,
    # Human-Researcher, Human-Tourist, Human-Park Ranger, Human-horseback rider, Human-Resident, Human-Maintenance crew
    # Human-Pedestrian, Human - Soldier, Human - Biker, Human-Skier, Human-Snowshoer, Human-Hiker)
    taxon_name_to_preferred_taxon_id['homo sapiens'] = '990ae9dd-7a59-4344-afcb-1b7b21368000' # 2002045

    # "Domestic Dog" (instead of Dog-on-leash)
    taxon_name_to_preferred_taxon_id['canis familiaris'] = '3d80f1d6-b1df-4966-9ff4-94053c7a902a' # 2021548

    # "Mammal" (instead of Small Mammal)
    taxon_name_to_preferred_taxon_id['mammalia'] = 'f2d233e3-80e3-433d-9687-e29ecc7a467a' # 2021108

    # "gastropods" (instead of Snail species, Slug species)
    taxon_name_to_preferred_taxon_id['gastropoda'] = '5b91403b-7361-4bb0-bf42-9a1687a88198' # 2021755

    # "Hispaniolan Mango" (instead of nan)
    taxon_name_to_preferred_taxon_id['anthracothorax dominicus'] = 'f94e6d97-59cf-4d38-a05a-a75efdd2863b' # 3

    # "Lizards and Snakes" (instead of Serpentes Suborder, Lacertilia Suborder)
    taxon_name_to_preferred_taxon_id['squamata'] = '0af344ad-6657-42c4-85d8-03fd2106a22a' # 2021551

    # "Beautiful Firetail" (instead of )
    taxon_name_to_preferred_taxon_id['stagonopleura bella'] = '7fec8e7e-fd3b-4d7f-99fd-3ade6f3bbaa5' # 2021939

    # "Rodent" (instead of Mouse species, Woodrat or Rat or Mouse Species, Woodrat or Rat Species)
    taxon_name_to_preferred_taxon_id['rodentia'] = '90d950db-2106-4bd9-a4c1-777604c3eada' # 2021173

    # "Yellow Crowned-wagtail" (instead of Yellow Wagtail)
    taxon_name_to_preferred_taxon_id['motacilla flava'] = 'a51fd282-2db6-4945-bebe-d64e52153bcf' # 2016200

    # "Dremomys Species" (instead of Dremomys Genus)
    taxon_name_to_preferred_taxon_id['dremomys'] = '1507d153-af11-46f1-bfb8-77918d035ab3' # 2019370

    # "Snowmobile" (instead of )
    taxon_name_to_preferred_taxon_id['snowmobile'] = 'c091eeb7-d1c5-4779-988f-82dd6fa12aae' # 2021826

    # "Elk" (instead of Domestic Elk)
    taxon_name_to_preferred_taxon_id['cervus canadensis'] = 'c5ce946f-8f0d-4379-992b-cc0982381f5e' # 2021592

    # "American Bison" (instead of Domestic Bison)
    taxon_name_to_preferred_taxon_id['bison bison'] = '539ebd55-081b-429a-9ae6-5a6a0f6999d4' # 2021593

    # "Cricetidae Family" (instead of Arvicolinae Subfamily)
    taxon_name_to_preferred_taxon_id['cricetidae'] = '523439f4-dee3-4f41-afea-edc0c891ef9c' # 2021115

    # "Heteromyidae Family" (instead of Perognathinae Subfamily)
    taxon_name_to_preferred_taxon_id['heteromyidae'] = 'a181f9da-8101-45dc-8bf4-06d6dcbebaef' # 2021148

    # "Weasel Family" (instead of Mustelinae Subfamily)
    taxon_name_to_preferred_taxon_id['mustelidae'] = '80d82668-81f1-4ffe-96a5-2325ce78ecf1' # 2021337

    # "Anniella Species" (instead of Aniella species)
    taxon_name_to_preferred_taxon_id['anniella'] = '46fd4962-c310-4a4a-833e-fa5fd068caa9' # 2021645

    # "Southern Sand Frog" (instead of )
    taxon_name_to_preferred_taxon_id['tomopterna adiastola'] = 'd8e212e7-6bd5-4539-a7f8-4eb43eba97d1' # 2021829

    # "Aspidoscelis Species" (instead of Aspidocelis species)
    taxon_name_to_preferred_taxon_id['aspidoscelis'] = 'd5bbf997-d2b4-4d4f-bcca-4b0b2e867e97' # 2021651

    # "Phrynosomatidae Family" (instead of Sceloporus/Uta species)
    taxon_name_to_preferred_taxon_id['phrynosomatidae'] = 'f38cb662-8569-48cd-b93b-34d20310b989' # 2021691

    # "White-rumped Shama" (instead of )
    taxon_name_to_preferred_taxon_id['copsychus malabaricus'] = '15ecc823-cbf1-4ada-bc5d-fc5a69cf1574' # 2015152

    # "Rosy Boa" (instead of Northern Three-lined Boa)
    taxon_name_to_preferred_taxon_id['lichanura orcutti'] = '7c373a44-7add-430d-88b0-197b7673459a' # 1107

    # "Comb-eared Skinks" (instead of Ctenotus genus)
    taxon_name_to_preferred_taxon_id['ctenotus'] = 'eae2e6df-e515-4b27-963e-3bb267205760' # 1180

    # "Agamidae Family" (instead of Dragon)
    taxon_name_to_preferred_taxon_id['agamidae'] = 'b3a3aaf6-dd72-4fdc-95ca-bcb282bca698' # 1271

    # "Sericornis Species" (instead of Scrubwren Species)
    taxon_name_to_preferred_taxon_id['sericornis'] = 'ad82c0ac-df48-4028-bf71-d2b2f4bc4129' # 2020133

    print('Defined preferences for {} taxa'.format(
        len(taxon_name_to_preferred_taxon_id)))

    assert len(taxon_name_to_preferred_taxon_id) <= \
        len(taxon_names_with_multiple_entries)

    for s in taxon_names_with_multiple_entries:
        if s not in taxon_name_to_preferred_taxon_id:
            print('Warning: no mapping for {}'.format(s))


    #%% Reduce each set of redundant taxa to a single choice

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

    # Discard header
    lines = lines[1:]
    lines = [s.strip() for s in lines if len(s.strip()) > 0]

    supplementary_lila_query_to_wi_query = {}

    for line in lines:
        assert '\t' not in line
        # Each line is [lila query],[WI taxon name],[notes]
        tokens = line.strip().split(',')
        assert len(tokens) == 3
        lila_query = tokens[0].strip().lower()
        wi_taxon_name = tokens[1].strip().lower()
        if wi_taxon_name not in wi_taxon_name_to_taxa:
            raise ValueError('{} not in WI taxonomy'.format(wi_taxon_name))
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
        ['unidentifiable', 'other', 'unidentified', 'unknown', 'unclassifiable', 'no cv result'])
    blank_queries = set(['empty'])
    # This makes sense in the context of the WI taxonomy, which doesn't include invertebrates
    animal_queries = set(['animalia','vertebrata'])

    print_common_name_matches = True
    print_supplementary_matches = False

    lila_dataset_category_to_wi_taxon = {}

    # i_taxon = 0; lila_taxon = lila_taxonomy[i_taxon]; print(taxon)
    for i_taxon, lila_taxon in enumerate(lila_taxonomy):

        # What is the taxon name we should search for in the WI taxonomy?
        query = None

        lila_dataset_category = lila_taxon['dataset_name'] + ':' + lila_taxon['query']

        # Go from kingdom --> species, choosing the lowest-level description as the query
        for level in lila_taxonomy_levels:
            if isinstance(lila_taxon[level], str):
                query = lila_taxon[level]
                all_searches.add(query)

        # If we didn't find a taxon name, that means this is a non-taxonomic entity, like "car"
        if query is None:
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
            if print_supplementary_matches:
                print('Made a supplementary mapping from {} to {}'.format(query,wi_taxon['taxon_name']))

        else:

            # print('No match for {}'.format(query))
            lila_common_name = lila_taxon['common_name']

            if lila_common_name in wi_common_name_to_taxon:
                wi_taxon = wi_common_name_to_taxon[lila_common_name]
                wi_common_name = wi_taxon['commonNameEnglish']
                wi_taxon_name = wi_taxon['taxon_name']
                if print_common_name_matches:
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


    #%% Write mapping file

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


    #%% Write mapping table file

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

    #%%

# ...if False (prevent execution at import)
