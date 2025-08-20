"""

wi_utils_driver.py

Utility cells to work with the wi_taxonomy_utils module.

"""

#%% Shared cell to initialize geofencing and taxonomy information

from megadetector.utils.wi_taxonomy_utils import species_allowed_in_country # noqa
from megadetector.utils.wi_taxonomy_utils import initialize_geofencing, initialize_taxonomy_info # noqa
from megadetector.utils.wi_taxonomy_utils import _species_string_to_canonical_species_string # noqa
from megadetector.utils.wi_taxonomy_utils import generate_csv_rows_for_species # noqa
from megadetector.utils.wi_taxonomy_utils import _generate_csv_rows_to_block_all_countries_except # noqa

from megadetector.utils.wi_taxonomy_utils import taxonomy_string_to_geofencing_rules # noqa
from megadetector.utils.wi_taxonomy_utils import taxonomy_string_to_taxonomy_info # noqa
from megadetector.utils.wi_taxonomy_utils import common_name_to_taxonomy_info # noqa
from megadetector.utils.wi_taxonomy_utils import binomial_name_to_taxonomy_info # noqa
from megadetector.utils.wi_taxonomy_utils import country_to_country_code # noqa
from megadetector.utils.wi_taxonomy_utils import country_code_to_country # noqa

model_base = os.path.expanduser('~/models/speciesnet')
geofencing_file = os.path.join(model_base,'crop','geofence_release.2025.02.27.0702.json')
country_code_file = os.path.join(model_base,'country-codes.csv')
# encoding = 'cp1252'; taxonomy_file = r'g:\temp\taxonomy_mapping-' + encoding + '.json'
encoding = None; taxonomy_file = os.path.join(model_base,'taxonomy_mapping.json')

initialize_geofencing(geofencing_file, country_code_file, force_init=True)
initialize_taxonomy_info(taxonomy_file, force_init=True, encoding=encoding)

# from megadetector.utils.path_utils import open_file; open_file(geofencing_file)


#%% Generate a block list

taxon_name = 'cercopithecidae'
taxonomy_info = binomial_name_to_taxonomy_info[taxon_name]
taxonomy_string_short = taxonomy_info_to_taxonomy_string(taxonomy_info)
assert len(taxonomy_string_short.split(';')) == 5

block_list = 'ATG,BHS,BRB,BLZ,CAN,CRI,CUB,DMA,DOM,SLV,GRD,GTM,HTI,HND,JAM,' + \
                'MEX,NIC,PAN,KNA,LCA,VCT,TTO,USA,ARG,BOL,BRA,CHL,COL,ECU,GUY,PRY,PER,' + \
                'SUR,URY,VEN,ALB,AND,ARM,AUT,AZE,BLR,BEL,BIH,BGR,HRV,CYP,CZE,DNK,EST,FIN,' + \
                'FRA,GEO,DEU,GRC,HUN,ISL,IRL,ITA,KAZ,XKX,LVA,LIE,LTU,LUX,MLT,MDA,MCO,MNE,' + \
                'NLD,MKD,NOR,POL,PRT,ROU,RUS,SMR,SRB,SVK,SVN,ESP,SWE,CHE,TUR,UKR,GBR,VAT,AUS'

rows = generate_csv_rows_for_species(species_string=taxonomy_string_short,
                                        allow_countries=None,
                                        block_countries=block_list,
                                        allow_states=None,
                                        block_states=None)

# import clipboard; clipboard.copy('\n'.join(rows))
print(rows)


#%% Look up taxonomy info for a common name

common_name = 'domestic horse'
info = common_name_to_taxonomy_info[common_name]
s = taxonomy_info_to_taxonomy_string(info,include_taxon_id_and_common_name=True)
print(s)


#%% Generate a block-except list

block_except_list = 'ALB,AND,ARM,AUT,AZE,BEL,BGR,BIH,BLR,CHE,CYP,CZE,DEU,DNK,ESP,EST,FIN,FRA,GBR,GEO,GRC,HRV,HUN,IRL,IRN,IRQ,ISL,ISR,ITA,KAZ,LIE,LTU,LUX,LVA,MDA,MKD,MLT,MNE,NLD,NOR,POL,PRT,ROU,RUS,SMR,SRB,SVK,SVN,SWE,TUR,UKR,UZB'
species = 'eurasian badger'
species_string = _species_string_to_canonical_species_string(species)
rows = _generate_csv_rows_to_block_all_countries_except(species_string,block_except_list)

# import clipboard; clipboard.copy('\n'.join(rows))
print(rows)


#%% Generate an allow-list

taxon_name = 'potoroidae'
taxonomy_info = binomial_name_to_taxonomy_info[taxon_name]
taxonomy_string_short = taxonomy_info_to_taxonomy_string(taxonomy_info)
assert len(taxonomy_string_short.split(';')) == 5

rows = generate_csv_rows_for_species(species_string=taxonomy_string_short,
                                        allow_countries=['AUS'],
                                        block_countries=None,
                                        allow_states=None,
                                        block_states=None)

# import clipboard; clipboard.copy('\n'.join(rows))
print(rows)


#%% Test the effects of geofence changes

species = 'canis lupus dingo'
country = 'guatemala'
species_allowed_in_country(species,country,state=None,return_status=False)


#%% Geofencing lookups

# This can be a latin or common name
taxon = 'potoroidae'
# print(common_name_to_taxonomy_info[taxon])

# This can be a name or country code
country = 'AUS'
print(species_allowed_in_country(taxon, country))


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

if True:

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

if True:

    all_species = ['ammospermophilus harrisii']
    country = 'USA'
    state = 'CA'

for species in all_species:

    taxonomy_info = binomial_name_to_taxonomy_info[species]
    allowed = species_allowed_in_country(species, country, state=state, return_status=True)
    state_string = ''
    if state is not None:
        state_string = ' ({})'.format(state)
    print('{} ({}) for {}{}: {}'.format(taxonomy_info['common_name'],species,country,state_string,allowed))
