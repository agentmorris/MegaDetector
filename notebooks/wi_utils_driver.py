#%% Header

"""

wi_utils_driver.py

Utility cells to work with the wi_taxonomy_utils module.

"""

#%% Imports and constants

import os

model_base = os.path.expanduser('~/models/speciesnet')
geofencing_file = os.path.join(model_base,'crop','geofence_release.20251208.json')
taxonomy_file = os.path.join(model_base,'crop','taxonomy_release.20251208.txt')

# This is not part of the model package, it comes from:
#
# https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
#
# wget "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/refs/heads/master/all/all.csv" -O ~/models/speciesnet/country-codes.csv
country_code_file = os.path.join(model_base,'country-codes.csv')

for fn in [geofencing_file,country_code_file,taxonomy_file]:
    assert os.path.isfile(fn), 'Could not find file {}'.format(fn)

us_state_codes = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC'
]


#%% Initialize taxonomy functions

from megadetector.utils.wi_taxonomy_utils import TaxonomyHandler
from megadetector.utils.wi_taxonomy_utils import taxonomy_info_to_taxonomy_string

taxonomy_handler = TaxonomyHandler(taxonomy_file=taxonomy_file,
                                   geofencing_file=geofencing_file,
                                   country_code_file=country_code_file)


#%% Generate a block list

taxon_name = 'sciurus vulgaris'
taxonomy_info = taxonomy_handler.binomial_name_to_taxonomy_info[taxon_name]
taxonomy_string_short = taxonomy_info_to_taxonomy_string(taxonomy_info)
assert len(taxonomy_string_short.split(';')) == 5

block_list = 'USA'

rows = taxonomy_handler.generate_csv_rows_for_species(species_string=taxonomy_string_short,
                                                      allow_countries=None,
                                                      block_countries=block_list,
                                                      allow_states=None,
                                                      block_states=None)

# import clipboard; clipboard.copy('\n'.join(rows))
for s in rows:
    print(s)


#%% Look up taxonomy info for a common name

common_name = 'sika deer'
info = taxonomy_handler.common_name_to_taxonomy_info[common_name]
s = taxonomy_info_to_taxonomy_string(info,include_taxon_id_and_common_name=True)
print(s)


#%% Generate a block-except list

block_except_list = 'ALB,AND,ARM,AUT,AZE,BEL,BGR,BIH,BLR,CHE,CYP,CZE,DEU,DNK,ESP,EST,FIN,FRA,GBR,GEO,GRC,HRV,HUN,IRL,IRN,IRQ,ISL,ISR,ITA,KAZ,LIE,LTU,LUX,LVA,MDA,MKD,MLT,MNE,NLD,NOR,POL,PRT,ROU,RUS,SMR,SRB,SVK,SVN,SWE,TUR,UKR,UZB'
species = 'eurasian badger'
species_string = taxonomy_handler.species_string_to_canonical_species_string(species)
rows = taxonomy_handler.generate_csv_rows_to_block_all_countries_except(species_string,block_except_list)

# import clipboard; clipboard.copy('\n'.join(rows))

for s in rows:
    print(s)


#%% Generate a block-except list

block_except_list = 'AUS'
species = 'potoroidae'
species_string = taxonomy_handler.species_string_to_canonical_species_string(species)
rows = taxonomy_handler.generate_csv_rows_to_block_all_countries_except(species_string,block_except_list)

# import clipboard; clipboard.copy('\n'.join(rows))

for s in rows:
    print(s)


#%% Generate a block-except list for states

block_states_except = ['NJ','DE']

species = 'sika deer'
species_string = taxonomy_handler.species_string_to_canonical_species_string(species)

rows = []

for state_code in us_state_codes:
    if state_code in block_states_except:
        row = species_string + ',allow,USA,' + state_code
    else:
        row = species_string + ',block,USA,' + state_code
    rows.append(row)

# import clipboard; clipboard.copy('\n'.join(rows))


#%% Generate an allow-list

taxon = 'sciurus carolinensis'
# taxon = 'eastern fox squirrel'
species_string = taxonomy_handler.species_string_to_canonical_species_string(taxon)
taxonomy_info = taxonomy_handler.taxonomy_string_to_taxonomy_info[species_string]

if False:
    taxonomy_info = taxonomy_handler.binomial_name_to_taxonomy_info[taxon]
    taxonomy_string_short = taxonomy_info_to_taxonomy_string(taxonomy_info)
    assert len(taxonomy_string_short.split(';')) == 5

rows = taxonomy_handler.generate_csv_rows_for_species(species_string=species_string,
                                                      allow_countries=['USA'],
                                                      block_countries=None,
                                                      allow_states=['CA','OR','ID','MT','WA'],
                                                      block_states=None)

# import clipboard; clipboard.copy('\n'.join(rows))

for s in rows:
    print(s)


#%% Generate a block+allow list

common_name = 'sika deer'
info = taxonomy_handler.common_name_to_taxonomy_info[common_name]
taxon_name = info['binomial_name']
taxonomy_info = taxonomy_handler.binomial_name_to_taxonomy_info[taxon_name]
taxonomy_string_short = taxonomy_info_to_taxonomy_string(taxonomy_info)
assert len(taxonomy_string_short.split(';')) == 5

rows = taxonomy_handler.generate_csv_rows_for_species(species_string=taxonomy_string_short,
                                                     allow_countries=None,
                                                     block_countries=None,
                                                     allow_states=['NJ','DE'],
                                                     block_states=['OH','NY'])

# import clipboard; clipboard.copy('\n'.join(rows))

for s in rows:
    print(s)


#%% Determine whether a species is allowed in a location

taxon = 'eastern gray squirrel'
country = 'USA'
state = 'CA'
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=state,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))


#%% Other locations

taxon = 'mustela altaica'
country = 'USA'
state = 'CA'
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=state,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'pittidae'
country = 'canada'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'pittidae'
country = 'india'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'hippopotamidae'
country = 'canada'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'hippopotamidae'
country = 'colombia'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'capreolus capreolus'
country = 'united states of america'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'capreolus capreolus'
country = 'france'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'meles meles'
country = 'united states of america'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'meles meles'
country = 'france'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'capreolus'
country = 'united states of america'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'capreolus'
country = 'france'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'canis lupus dingo'
country = 'guatemala'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'canis lupus dingo'
country = 'australia'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'macropodidae'
country = 'united states of america'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'macropodidae'
country = 'AUS'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'potoroidae'
country = 'united states of america'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'potoroidae'
country = 'AUS'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=None,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'sika deer'
country = 'china'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=state,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'sika deer'
country = 'USA'
state = 'OH'
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=state,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'sciurus vulgaris'
country = 'USA'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=state,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))

taxon = 'sciurus vulgaris'
country = 'RUS'
state = None
allowed = taxonomy_handler.species_allowed_in_country(taxon,country,state=state,return_status=False)
taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(taxon)
common_name = taxonomy_info['common_name']
print('{} ({}) in {} ({}): {}'.format(taxon,common_name,country,state,allowed))


#%% Geofence updates

import os

repo_root = os.path.expanduser('~/git/cameratrapai')
script_file = os.path.join(repo_root,'speciesnet/scripts/build_geofence_release.py')
geofence_base_file = os.path.join(repo_root,'data/geofence_base.json')
geofence_fixes_file = os.path.join(repo_root,'data/geofence_fixes.csv')
taxonomy_file = os.path.join(repo_root,'data/model_package/taxonomy_release.txt')
model_category_list_file = os.path.join(repo_root,'data/model_package/always_crop_99710272_22x8_v12_epoch_00148.labels.txt')
# output_file = os.path.join(repo_root,'data/model_package/geofence_release_updated.json')
output_file = os.path.expanduser('~/models/speciesnet/crop/geofence_release.temporary.json')

assert os.path.isfile(geofence_base_file)
assert os.path.isfile(geofence_fixes_file)
assert os.path.isfile(model_category_list_file)
assert os.path.isfile(taxonomy_file)
assert os.path.isfile(script_file)

cmd = f'python "{script_file}" --base "{geofence_base_file}" --fixes "{geofence_fixes_file}"'
cmd += f' --output "{output_file}" --trim "{model_category_list_file}"'
cmd += f' --taxonomy "{taxonomy_file}"'

cmd = cmd.replace('\\','/')
print(cmd)
# import clipboard; clipboard.copy(cmd)


#%% Bulk geofence lookups

if True:

    # Make sure some Guatemalan species are allowed in Guatemala
    all_species = [
        'didelphis marsupialis',
        'didelphis virginiana',
        'dasypus novemcinctus',
        'urocyon cinereoargenteus',
        'nasua narica',
        'eira barbara',
        'conepatus semistriatus',
        'leopardus wiedii',
        'leopardus pardalis',
        'puma concolor',
        'panthera onca',
        'tapirus bairdii',
        'pecari tajacu',
        'tayassu pecari',
        'mazama temama',
        'mazama pandora',
        'odocoileus virginianus',
        'dasyprocta punctata',
        'tinamus major',
        'crax rubra',
        'meleagris ocellata',
        'gulo gulo' # Consistency check; this species should be blocked
        ]

    country ='guatemala'
    state = None

if False:

    # Make sure some PNW species are allowed in the right states
    all_species = \
        ['Taxidea taxus',
        'Martes americana',
        'Ovis canadensis',
        'Ursus americanus',
        'Lynx rufus',
        'Lynx canadensis',
        'Puma concolor',
        'Canis latrans',
        'Cervus canadensis',
        'Canis lupus',
        'Ursus arctos',
        'Marmota caligata',
        'Alces alces',
        'Oreamnos americanus',
        'Odocoileus hemionus',
        'Vulpes vulpes',
        'Lepus americanus',
        'Mephitis mephitis',
        'Odocoileus virginianus',
        'Marmota flaviventris',
        'tapirus bairdii' # Consistency check; this species should be blocked
        ]

    all_species = [s.lower() for s in all_species]

    country = 'USA'
    state = 'WA'
    # state = 'MT'

if False:

    all_species = ['ammospermophilus harrisii']
    country = 'USA'
    state = 'CA'

for species in all_species:

    taxonomy_info = taxonomy_handler.binomial_name_to_taxonomy_info[species]
    allowed = taxonomy_handler.species_allowed_in_country(
        species, country, state=state, return_status=True)
    state_string = ''
    if state is not None:
        state_string = ' ({})'.format(state)
    print('{} ({}) for {}{}: {}'.format(taxonomy_info['common_name'],species,country,state_string,allowed))


#%% Compare the SpeciesNet release taxonomy file to the .json WI taxonomy file

"""
At some point I grabbed the WI taxonomy and stored in a .json file mapping five-
to seven-token IDs.  wi_taxonomy_utils took a dependency on that file, and before
porting wi_taxonomy_utils over to the SpeciesNet release taxonomy, I want to double-check
that they exist in the same universe.

tl;dr: they are in the same universe.  Differences are just changes to GUIDs or common names.
"""

import os
import json

from tqdm import tqdm

speciesnet_model_folder = os.path.expanduser('~/models/speciesnet/crop')
release_taxonomy_file = os.path.join(speciesnet_model_folder,'taxonomy_release.txt')

with open(release_taxonomy_file,'r') as f:
    release_taxonomy_lines = f.readlines()
release_taxonomy_lines = [s.strip() for s in release_taxonomy_lines]

print('Read {} lines from release taxonomy file'.format(len(release_taxonomy_lines)))

json_taxonomy_file = os.path.join(speciesnet_model_folder,'..','taxonomy_mapping.json')

with open(json_taxonomy_file,'r') as f:
    json_taxonomy = json.load(f)

print('Read {} lines from json taxonomy file'.format(len(json_taxonomy)))

all_seven_token_ids = set()

for five_token_id in json_taxonomy.keys():
    assert len(five_token_id.split(';')) == 5
    seven_token_id = json_taxonomy[five_token_id]
    assert len(seven_token_id.split(';')) == 7
    assert five_token_id in seven_token_id
    all_seven_token_ids.add(seven_token_id
                            )
# release_taxonomy_line = release_taxonomy_lines[0]

speciesnet_taxa_not_in_wi_taxonomy_as_seven_token_ids = []
speciesnet_taxa_not_in_wi_taxonomy_as_five_token_ids = []

for release_taxonomy_line in release_taxonomy_lines:
    tokens = release_taxonomy_line.split(';')
    assert len(tokens) == 7
    five_token_id = ';'.join(tokens[1:-1])
    if release_taxonomy_line not in all_seven_token_ids:
        speciesnet_taxa_not_in_wi_taxonomy_as_seven_token_ids.append(release_taxonomy_line)
    if five_token_id not in json_taxonomy:
        speciesnet_taxa_not_in_wi_taxonomy_as_five_token_ids.append(release_taxonomy_line)

print('{} of {} SpeciesNet taxa don\'t match the WI taxonomy file exactly:'.format(
    len(speciesnet_taxa_not_in_wi_taxonomy_as_seven_token_ids),len(release_taxonomy_lines)))

for s in speciesnet_taxa_not_in_wi_taxonomy_as_seven_token_ids:
    print(s)

print('\n{} of {} SpeciesNet taxa don\'t match the WI taxonomy file even as five-token IDs:'.format(
    len(speciesnet_taxa_not_in_wi_taxonomy_as_five_token_ids),len(release_taxonomy_lines)))

for s in speciesnet_taxa_not_in_wi_taxonomy_as_five_token_ids:
    print(s)


#%% Taxonomy mapping (init)

import os
from megadetector.utils.path_utils import insert_before_extension

# A .csv file with columns:
#
# latin,common,original_latin,original_common
#
# Typically latin and common are empty when we start
# this process.  Often one of the other columns is empty as well.
# input_fn = 'c:/git/agentmorrisprivate/taxonomy-lists/seattleish-camera-traps_speciesnet.csv'
# input_fn = 'c:/git/agentmorrisprivate/taxonomy-lists/snapshot_serengeti_input.csv'
input_fn = 'c:/git/agentmorrisprivate/taxonomy-lists/fstop_input.csv'
output_fn = insert_before_extension(input_fn,'speciesnet',separator='_')
assert os.path.isfile(input_fn)

required_columns = ('latin','common','original_latin','original_common')


#%% Taxonomy mapping (iterative lookup)

assert taxonomy_handler is not None

import pandas as pd
from megadetector.utils.ct_utils import is_empty

df = pd.read_csv(input_fn)

input_length = len(df)

# Remove rows where is_empty(x) is true for every column in the row
df = df[~df.apply(lambda row: all(is_empty(val) for val in row), axis=1)]

n_empty = input_length - len(df)
print('Removed {} empty rows (of {})'.format(n_empty,input_length))

for s in required_columns:
    assert s in df.columns

failed_matches = []

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():

    if ((not is_empty(row['latin'])) or (not is_empty(row['common']))):
        assert ((not is_empty(row['latin'])) and (not is_empty(row['common']))), \
            'If one of the SpeciesNet names is populated, both should be'

    if not is_empty(row['latin']):
        taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(row['latin'])
        assert taxonomy_info['binomial_name'] == row['latin']
        assert taxonomy_info['common_name'] == row['common']
        continue

    if not is_empty(row['common']):
        taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(row['common'])
        assert taxonomy_info['binomial_name'] == row['latin']
        assert taxonomy_info['common_name'] == row['common']
        continue

    if not is_empty(row['original_latin']):

        latin_match = False

        try:
            taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(row['original_latin'])
            latin_match = True
        except Exception:
            pass

        if latin_match:
            latin = taxonomy_info['binomial_name']
            common = taxonomy_info['common_name']
            print('Mapped {} ({}) to {} ({})'.format(
                row['original_latin'],row['original_common'],
                latin,common))
            df.loc[i_row, 'latin'] = latin
            df.loc[i_row, 'common'] = common
            continue

    if not is_empty(row['original_common']):

        common_match = False

        try:
            taxonomy_info = taxonomy_handler.species_string_to_taxonomy_info(row['original_common'])
            common_match = True
        except Exception:
            pass

        if common_match:
            latin = taxonomy_info['binomial_name']
            common = taxonomy_info['common_name']
            print('Mapped {} ({}) to {} ({})'.format(
                row['original_latin'],row['original_common'],
                latin,common))
            df.loc[i_row, 'latin'] = latin
            df.loc[i_row, 'common'] = common
            continue

    failed_matches.append((row['original_latin'],row['original_common']))

# ...for each row

for failed_match in failed_matches:
    print('No match for {} ({})'.format(
        failed_match[0],failed_match[1]))

df.to_csv(output_fn,index=False)


#%% Find non-species geofencing rules

import json

geofencing_file = os.path.join(model_base,'crop','geofence_release.2025.02.27.0702.json')

non_species_rules = set()

# geofence_base.json only contains species-level results
# geofencing_file = 'c:/git/cameratrapai/data/geofence_base.json'

with open(geofencing_file,'r') as f:
    geofence_dict = json.load(f)

for taxon in geofence_dict.keys():
    tokens = taxon.split(';')
    assert len(tokens) == 5
    for token in tokens:
        if len(token) == 0:
            non_species_rules.add(taxon)
            break

print('Found {} non-species-level rules (of {})'.format(
    len(non_species_rules),len(geofence_dict)))


#%% Check whether a species should get geofenced...

# ...according to the speciesnet package

import json

from speciesnet.geofence_utils import should_geofence_animal_classification
from speciesnet.geofence_utils import geofence_animal_classification
from speciesnet.ensemble import _load_taxonomy_from_file

country = 'USA'
state = 'CA'
# taxon_name = 'neotamias species'
taxon_name = 'eastern chipmunk'

taxonomy_map = _load_taxonomy_from_file(taxonomy_file)
species_string = taxonomy_handler.species_string_to_canonical_species_string(taxon_name)
taxonomy_info = taxonomy_handler.taxonomy_string_to_taxonomy_info[species_string]

# Generate a seven-token string
taxonomy_string_long = taxonomy_info_to_taxonomy_string(taxonomy_info=taxonomy_info,
                                                        include_taxon_id_and_common_name=True)

with open(geofencing_file,'r') as f:
    geofence_map = json.load(f)

should_gefence = should_geofence_animal_classification(
                    label=taxonomy_string_long,
                    country=country,
                    admin1_region=state,
                    geofence_map=geofence_map,
                    enable_geofence=True)

print('Should geofence {} ({},{}): {}'.format(taxon_name,country,state,should_gefence))

label,score,prediction_source = geofence_animal_classification(
                                    labels=[taxonomy_string_long],
                                    scores=[1.0],
                                    country=country,
                                    admin1_region=state,
                                    taxonomy_map=taxonomy_map,
                                    geofence_map=geofence_map,
                                    enable_geofence=True)
