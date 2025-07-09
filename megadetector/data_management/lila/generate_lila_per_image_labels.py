"""

generate_lila_per_image_labels.py

Generate a .csv file with one row per annotation, containing full URLs to every
camera trap image on LILA, with taxonomically expanded labels.

Typically there will be one row per image, though images with multiple annotations
will have multiple rows.

Some images may not physically exist, particularly images that are labeled as "human".
This script does not validate image URLs.

Does not include bounding box annotations.

"""

#%% Constants and imports

import os
import json
import pandas as pd
import numpy as np
import dateparser
import csv

from collections import defaultdict
from tqdm import tqdm

from megadetector.data_management.lila.lila_common import \
    read_lila_metadata, \
    read_metadata_file_for_dataset, \
    read_lila_taxonomy_mapping

from megadetector.utils import write_html_image_list
from megadetector.utils.path_utils import zip_file
from megadetector.utils.path_utils import open_file
from megadetector.utils.url_utils import parallel_download_urls

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')
preview_folder = os.path.join(lila_local_base,'csv_preview')

os.makedirs(lila_local_base,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_file = os.path.join(lila_local_base,'lila_image_urls_and_labels.csv')

# Some datasets don't have "sequence_level_annotation" fields populated, but we know their
# annotation level
ds_name_to_annotation_level = {}
ds_name_to_annotation_level['Caltech Camera Traps'] = 'image'
ds_name_to_annotation_level['ENA24'] = 'image'
ds_name_to_annotation_level['Island Conservation Camera Traps'] = 'image'
ds_name_to_annotation_level['Channel IslandsCamera Traps'] = 'image'
ds_name_to_annotation_level['WCS Camera Traps'] = 'sequence'
ds_name_to_annotation_level['Wellington Camera Traps'] = 'sequence'
ds_name_to_annotation_level['NACTI'] = 'unknown'
ds_name_to_annotation_level['Seattle(ish) Camera Traps'] = 'image'

known_unmapped_labels = set(['WCS Camera Traps:#ref!'])

debug_max_images_per_dataset = -1
if debug_max_images_per_dataset > 0:
    print('Running in debug mode')
    output_file = output_file.replace('.csv','_debug.csv')

taxonomy_levels_to_include = \
    ['kingdom','phylum','subphylum','superclass','class','subclass','infraclass','superorder','order',
     'suborder','infraorder','superfamily','family','subfamily','tribe','genus','subgenus',
     'species','subspecies','variety']

def _clearnan(v):
    if isinstance(v,float):
        assert np.isnan(v)
        v = ''
    assert isinstance(v,str)
    return v


#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)

# To select an individual data set for debugging
if False:
    k = 'Idaho Camera Traps'
    metadata_table = {k:metadata_table[k]}


#%% Download and extract metadata for each dataset

for ds_name in metadata_table.keys():
    metadata_table[ds_name]['metadata_filename'] = read_metadata_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)

#%% Load taxonomy data

taxonomy_df = read_lila_taxonomy_mapping(metadata_dir, force_download=True)


#%% Build a dictionary that maps each [dataset,query] pair to the full taxonomic label set

ds_label_to_taxonomy = {}

# i_row = 0; row = taxonomy_df.iloc[i_row]
for i_row,row in taxonomy_df.iterrows():

    ds_label = row['dataset_name'] + ':' + row['query']
    assert ds_label.strip() == ds_label
    assert ds_label not in ds_label_to_taxonomy
    ds_label_to_taxonomy[ds_label] = row.to_dict()


#%% Process annotations for each dataset

# Takes a few hours

# The order of these headers needs to match the order in which fields are added later in this cell;
# don't mess with this order.
header = ['dataset_name','url_gcp','url_aws','url_azure',
          'image_id','sequence_id','location_id','frame_num',
          'original_label','scientific_name','common_name','datetime','annotation_level']

header.extend(taxonomy_levels_to_include)

missing_annotations = set()

with open(output_file,'w',encoding='utf-8',newline='') as f:

    csv_writer = csv.writer(f)
    csv_writer.writerow(header)

    # ds_name = list(metadata_table.keys())[0]
    for ds_name in metadata_table.keys():

        if 'bbox' in ds_name:
            print('Skipping bbox dataset {}'.format(ds_name))
            continue

        print('Processing dataset {}'.format(ds_name))

        json_filename = metadata_table[ds_name]['metadata_filename']
        with open(json_filename, 'r') as f:
            data = json.load(f)

        categories = data['categories']
        category_ids = [c['id'] for c in categories]
        for c in categories:
            category_id_to_name = {c['id']:c['name'] for c in categories}

        annotations = data['annotations']
        images = data['images']

        image_id_to_annotations = defaultdict(list)

        # Go through annotations, marking each image with the categories that are present
        #
        # ann = annotations[0]
        for ann in annotations:
            image_id_to_annotations[ann['image_id']].append(ann)

        unannotated_images = []

        found_date = False
        found_location = False
        found_annotation_level = False

        if ds_name in ds_name_to_annotation_level:
            expected_annotation_level = ds_name_to_annotation_level[ds_name]
        else:
            expected_annotation_level = None

        # im = images[10]
        for i_image,im in tqdm(enumerate(images),total=len(images)):

            if (debug_max_images_per_dataset is not None) and (debug_max_images_per_dataset > 0) \
                and (i_image >= debug_max_images_per_dataset):
                break

            file_name = im['file_name'].replace('\\','/')
            base_url_gcp = metadata_table[ds_name]['image_base_url_gcp']
            base_url_aws = metadata_table[ds_name]['image_base_url_aws']
            base_url_azure = metadata_table[ds_name]['image_base_url_azure']
            assert not base_url_gcp.endswith('/')
            assert not base_url_aws.endswith('/')
            assert not base_url_azure.endswith('/')

            url_gcp = base_url_gcp + '/' + file_name
            url_aws = base_url_aws + '/' + file_name
            url_azure = base_url_azure + '/' + file_name

            for k in im.keys():
                if ('date' in k or 'time' in k) and (k not in ['datetime','date_captured']):
                    raise ValueError('Unrecognized datetime field')

            # This field name was only used for Caltech Camera Traps
            if 'date_captured' in im:
                assert ds_name == 'Caltech Camera Traps'
                im['datetime'] = im['date_captured']

            def _has_valid_datetime(im):
                if 'datetime' not in im:
                    return False
                v = im['datetime']
                if v is None:
                    return False
                if isinstance(v,str):
                    return len(v) > 0
                else:
                    assert isinstance(v,float) and np.isnan(v)
                    return False

            dt_string = ''
            if (_has_valid_datetime(im)):

                dt = dateparser.parse(im['datetime'])

                if dt is None or dt.year < 1990 or dt.year > 2025:

                    # raise ValueError('Suspicious date parsing result')

                    # Special case we don't want to print a warning about... this is
                    # in invalid date that very likely originates on the camera, not at
                    # some intermediate processing step.
                    #
                    # print('Suspicious date for image {}: {} ({})'.format(
                    #    im['id'], im['datetime'], ds_name))
                    pass

                else:

                    found_date = True
                    dt_string = dt.strftime("%m-%d-%Y %H:%M:%S")

            # Location, sequence, and image IDs are only guaranteed to be unique within
            # a dataset, so for the output .csv file, include both
            if 'location' in im:
                found_location = True
                location_id = ds_name + ' : ' + str(im['location'])
            else:
                location_id = ds_name

            image_id = ds_name + ' : ' + str(im['id'])

            if 'seq_id' in im:
                sequence_id = ds_name + ' : ' + str(im['seq_id'])
            else:
                sequence_id = ds_name + ' : ' + 'unknown'

            if 'frame_num' in im:
                frame_num = im['frame_num']
            else:
                frame_num = -1

            annotations_this_image = image_id_to_annotations[im['id']]

            categories_this_image = set()

            annotation_level = 'unknown'

            for ann in annotations_this_image:
                assert ann['image_id'] == im['id']
                categories_this_image.add(category_id_to_name[ann['category_id']])
                if 'sequence_level_annotation' in ann:
                    found_annotation_level = True
                    if ann['sequence_level_annotation']:
                        annotation_level = 'sequence'
                    else:
                        annotation_level = 'image'
                    if expected_annotation_level is not None:
                        assert expected_annotation_level == annotation_level,\
                            'Unexpected annotation level'
                elif expected_annotation_level is not None:
                    annotation_level = expected_annotation_level

            if len(categories_this_image) == 0:
                unannotated_images.append(im)
                continue

            # category_name = list(categories_this_image)[0]
            for category_name in categories_this_image:

                ds_label = ds_name + ':' + category_name.lower()

                if ds_label not in ds_label_to_taxonomy:

                    assert ds_label in known_unmapped_labels

                    # Only print a warning the first time we see an unmapped label
                    if ds_label not in missing_annotations:
                        print('Warning: {} not in taxonomy file'.format(ds_label))
                    missing_annotations.add(ds_label)
                    continue

                taxonomy_labels = ds_label_to_taxonomy[ds_label]

                """
                header =
                    ['dataset_name','url','image_id','sequence_id','location_id',
                     'frame_num','original_label','scientific_name','common_name',
                     'datetime','annotation_level']
                """

                row = []
                row.append(ds_name)
                row.append(url_gcp)
                row.append(url_aws)
                row.append(url_azure)
                row.append(image_id)
                row.append(sequence_id)
                row.append(location_id)
                row.append(frame_num)
                row.append(taxonomy_labels['query'])
                row.append(_clearnan(taxonomy_labels['scientific_name']))
                row.append(_clearnan(taxonomy_labels['common_name']))
                row.append(dt_string)
                row.append(annotation_level)

                for s in taxonomy_levels_to_include:
                    row.append(_clearnan(taxonomy_labels[s]))

                assert len(row) == len(header)

                csv_writer.writerow(row)

            # ...for each category that was applied at least once to this image

        # ...for each image in this dataset

        if not found_date:
            pass
            # print('Warning: no date information available for this dataset')

        if not found_location:
            pass
            # print('Warning: no location information available for this dataset')

        if not found_annotation_level and (ds_name not in ds_name_to_annotation_level):
            print('Warning: no annotation level information available for this dataset')

        if len(unannotated_images) > 0:
            print('Warning: {} of {} images are un-annotated\n'.\
                  format(len(unannotated_images),len(images)))

    # ...for each dataset

# ...with open()

print('\nProcessed {} datasets'.format(len(metadata_table)))


#%% Read the .csv back

df = pd.read_csv(output_file)
print('Read {} rows from {}'.format(len(df),output_file))


#%% Do some post-hoc integrity checking

# Takes ~5 minutes with apply(), or ~10 minutes without apply()
#
# Using apply() is faster, but more annoying to debug.
use_pandas_apply_for_integrity_checking = True

tqdm.pandas()

def _isint(v):
    return isinstance(v,int) or isinstance(v,np.int64)

valid_annotation_levels = set(['sequence','image','unknown'])

# Collect a list of locations within each dataset; we'll use this
# in the next cell to look for datasets that only have a single location
dataset_name_to_locations = defaultdict(set)

def _check_row(row):

    assert row['dataset_name'] in metadata_table.keys()
    for url_column in ['url_gcp','url_aws','url_azure']:
        assert row[url_column].startswith('https://') or row[url_column].startswith('http://')
    assert ' : ' in row['image_id']
    assert 'seq' not in row['location_id'].lower()
    assert row['annotation_level'] in valid_annotation_levels

    # frame_num should either be NaN or an integer
    if isinstance(row['frame_num'],float):
        assert np.isnan(row['frame_num'])
    else:
        # -1 is sometimes used for sequences of unknown length
        assert _isint(row['frame_num']) and row['frame_num'] >= -1

    ds_name = row['dataset_name']
    dataset_name_to_locations[ds_name].add(row['location_id'])

if use_pandas_apply_for_integrity_checking:

    df.progress_apply(_check_row, axis=1)

else:

    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in tqdm(df.iterrows(),total=len(df)):
        _check_row(row)


#%% Check for datasets that have only one location string (typically "unknown")

# Expected: ENA24, Missouri Camera Traps, Desert Lion Conservation Camera Traps

for ds_name in dataset_name_to_locations.keys():
    if len(dataset_name_to_locations[ds_name]) == 1:
        print('No location information for {}'.format(ds_name))


#%% Preview constants

n_empty_images_per_dataset = 3
n_non_empty_images_per_dataset = 10

os.makedirs(preview_folder,exist_ok=True)


#%% Choose images to download

np.random.seed(0)
images_to_download = []

# ds_name = list(metadata_table.keys())[2]
for ds_name in metadata_table.keys():

    if 'bbox' in ds_name:
        continue

    # Find all rows for this dataset
    ds_rows = df.loc[df['dataset_name'] == ds_name]

    print('{} rows available for {}'.format(len(ds_rows),ds_name))
    assert len(ds_rows) > 0

    empty_rows = ds_rows[ds_rows['scientific_name'].isnull()]
    non_empty_rows = ds_rows[~ds_rows['scientific_name'].isnull()]

    if len(empty_rows) == 0:
        print('No empty images available for {}'.format(ds_name))
    elif len(empty_rows) > n_empty_images_per_dataset:
        empty_rows = empty_rows.sample(n=n_empty_images_per_dataset)
    images_to_download.extend(empty_rows.to_dict('records'))

    # All LILA datasets have non-empty images
    if len(non_empty_rows) == 0:
        raise ValueError('No non-empty images available for {}'.format(ds_name))
    elif len(non_empty_rows) > n_non_empty_images_per_dataset:
        non_empty_rows = non_empty_rows.sample(n=n_non_empty_images_per_dataset)
    images_to_download.extend(non_empty_rows.to_dict('records'))

 # ...for each dataset

print('Selected {} total images'.format(len(images_to_download)))


#%% Download images (prep)

# Expect a few errors for images with human or vehicle labels (or things like "ignore" that *could* be humans)

preferred_cloud = 'gcp'

url_to_target_file = {}

# i_image = 10; image = images_to_download[i_image]
for i_image,image in tqdm(enumerate(images_to_download),total=len(images_to_download)):

    url = image['url_' + preferred_cloud]
    ext = os.path.splitext(url)[1]
    fn_relative = 'image_{}'.format(str(i_image).zfill(4)) + ext
    fn_abs = os.path.join(preview_folder,fn_relative)
    image['relative_file'] = fn_relative
    image['url'] = url
    url_to_target_file[url] = fn_abs


#%% Download images (execution)

download_results = parallel_download_urls(url_to_target_file,verbose=False,overwrite=True,
                                          n_workers=20,pool_type='thread')

# 10-20 errors is normal; they should all be images that are labeled as "human"
errors = []

for r in download_results:
    if r['status'] != 'success':
        errors.append(r)

assert len(download_results) == len(url_to_target_file)
print('Errors on {} of {} downloads:\n'.format(len(errors),len(download_results)))

for err in errors:
    print(err['url'])


#%% Write preview HTML

html_filename = os.path.join(preview_folder,'index.html')

html_images = []

# im = images_to_download[0]
for im in images_to_download:

    if im['relative_file'] is None:
        continue

    output_im = {}
    output_im['filename'] = im['relative_file']
    output_im['linkTarget'] = im['url']
    output_im['title'] = '<b>{}: {}</b><br/><br/>'.format(im['dataset_name'],im['original_label']) + str(im)
    output_im['imageStyle'] = 'width:600px;'
    output_im['textStyle'] = 'font-weight:normal;font-size:100%;'
    html_images.append(output_im)

write_html_image_list.write_html_image_list(html_filename,html_images)

open_file(html_filename)


#%% Zip output file

zipped_output_file = zip_file(output_file,verbose=True,overwrite=True)

print('Zipped {} to {}'.format(output_file,zipped_output_file))


#%% Convert to .json

"""
The .csv file "output_file" (already loaded into the variable "df" at this point) has the following columns:

dataset_name,url_gcp,url_aws,url_azure,image_id,sequence_id,location_id,frame_num,original_label,scientific_name,common_name,datetime,annotation_level,kingdom,phylum,subphylum,superclass,class,subclass,infraclass,superorder,order,suborder,infraorder,superfamily,family,subfamily,tribe,genus,subgenus,species,subspecies,variety

Each row in the .csv represents an image.  The URL columns represent the location of that
image on three different clouds; for a given image, the value of those columns differs only
in the prefix.  The columns starting with "kingdom" represent a taxonomic wildlife identifier.  Not
all rows have values in all of these columns; some rows represent non-wildlife images where all of these
columns are blank.

This cell converts this to a .json dictionary, with the following top-level keys:

## datasets (dict)

A dict mapping integer IDs to strings.

Each unique value in the "dataset_name" column should become an element in this dict with a unique ID.

## sequences (dict)

A dict mapping integer IDs to strings.

Each unique value in the "sequence_id" column should become an element in this dict with a unique ID.

## locations (dict)

A dict mapping integer IDs to strings.

Each unique value in the "location_id" column should become an element in this dict with a unique ID.

## base_urls (dict)

This key should point to the following dict:

{
"gcp": "https://storage.googleapis.com/public-datasets-lila/",
"aws": "http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/",
"azure": "https://lilawildlife.blob.core.windows.net/lila-wildlife/",
}

All values in the url_gcp, url_aws, and url_azure columns start with these values, respectively.

## taxa (dict)

A dict mapping integer IDs to dicts, where each dict has the fields:

kingdom,phylum,subphylum,superclass,class,subclass,infraclass,superorder,order,suborder,infraorder,superfamily,family,subfamily,tribe,genus,subgenus,species,subspecies,variety

The value of each of these fields in each row is either a string or None.

## images (list)

A list of images, where each image is a dict with the following fields:

### dataset (int)

The integer ID corresponding to the dataset_name column for this image

### path (str)

The suffix for this image's URL, which should be the same across the three URL columns.

### seq (int)

The integer ID corresponding to the sequence_id column for this image

### loc (int)

The integer ID corresponding to the location_id column for this image

### frame_num

The value of the frame_num column for this image, unless the original value was -1,
in which case this is omitted.

### original_label

The value of the original_label column for this image

### common_name

The value of the common_name column for this image, if not empty

### datetime

The value of the datetime column for this image

### ann_level

The value of the annotation_level column for this image

### taxon

The integer ID corresponding to the taxonomic identifier columns for this image

--

The original .csv file is large (~15GB); this may impact the implementation of the .json conversion.  Speed of
conversion is not a priority.

"""

print('Converting to JSON...')

output_json_file = output_file.replace('.csv', '.json')

json_data = {}

# Create mappings for datasets, sequences, and locations
dataset_to_id = {}
sequence_to_id = {}
location_to_id = {}
taxa_to_id = {}

next_dataset_id = 0
next_sequence_id = 0
next_location_id = 0
next_taxa_id = 0

json_data['datasets'] = {}
json_data['sequences'] = {}
json_data['locations'] = {}
json_data['taxa'] = {}

json_data['base_urls'] = {
    "gcp": "https://storage.googleapis.com/public-datasets-lila/",
    "aws": "http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/",
    "azure": "https://lilawildlife.blob.core.windows.net/lila-wildlife/",
}

json_data['images'] = []

debug_max_json_conversion_rows = None

print('Counting rows in .csv file...')

# Get total number of lines for progress bar (optional, but helpful for large files)
def _count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for line in f) - 1

total_rows = _count_lines(output_file)
print('Total rows to process: {}'.format(total_rows))

# Read CSV file line by line
with open(output_file, 'r', encoding='utf-8') as csvfile:

    reader = csv.DictReader(csvfile)

    # Process each row
    for i_row, row in enumerate(tqdm(reader, total=total_rows, desc="Processing rows")):

        if (debug_max_json_conversion_rows is not None) and (i_row >= debug_max_json_conversion_rows):
            break

        # Datasets
        dataset_name = row['dataset_name']
        if dataset_name not in dataset_to_id:
            dataset_to_id[dataset_name] = next_dataset_id
            json_data['datasets'][str(next_dataset_id)] = dataset_name
            next_dataset_id += 1
        dataset_id = dataset_to_id[dataset_name]

        # Sequences
        sequence_id_str = row['sequence_id']
        assert sequence_id_str.startswith(dataset_name + ' : ')
        if sequence_id_str not in sequence_to_id:
            sequence_to_id[sequence_id_str] = next_sequence_id
            json_data['sequences'][str(next_sequence_id)] = sequence_id_str
            next_sequence_id += 1
        sequence_id = sequence_to_id[sequence_id_str]

        # Locations
        location_id_str = row['location_id']
        assert location_id_str.startswith(dataset_name) # + ' : ')
        if location_id_str not in location_to_id:
            location_to_id[location_id_str] = next_location_id
            json_data['locations'][str(next_location_id)] = location_id_str
            next_location_id += 1
        location_id = location_to_id[location_id_str]

        # Taxa
        taxa_data = {level: _clearnan(row[level]) for level in taxonomy_levels_to_include}
        taxa_tuple = tuple(taxa_data.items())  # use tuple for hashable key
        if taxa_tuple not in taxa_to_id:
            taxa_to_id[taxa_tuple] = next_taxa_id
            json_data['taxa'][str(next_taxa_id)] = taxa_data
            next_taxa_id += 1
        taxa_id = taxa_to_id[taxa_tuple]

        # Image path
        url_gcp = row['url_gcp']
        assert url_gcp.startswith(json_data['base_urls']['gcp'])
        path = url_gcp.replace(json_data['base_urls']['gcp'], '')

        common_name = _clearnan(row['common_name'])

        frame_num = int(row['frame_num'])

        # Image data
        image_entry = {
            'dataset': dataset_id,
            'path': path,
            'seq': sequence_id,
            'loc': location_id,
            'ann_level': row['annotation_level'],
            'original_label': row['original_label'],
            'datetime': row['datetime'],
            'taxon': taxa_id
        }

        if frame_num >= 0:
           image_entry['frame_num'] = frame_num

        if len(common_name) > 0:
            image_entry['common_name'] = common_name

        json_data['images'].append(image_entry)

    # ...for each line

# ...with open(...)

# Save the JSON data
print('Saving JSON file...')
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=1)

print(f'Converted to JSON and saved to {output_json_file}')
print(f'JSON file size: {os.path.getsize(output_json_file)/(1024*1024*1024):.2f} GB')

# Print summary statistics
print(f'Total datasets: {len(json_data["datasets"])}')
print(f'Total sequences: {len(json_data["sequences"])}')
print(f'Total locations: {len(json_data["locations"])}')
print(f'Total taxa: {len(json_data["taxa"])}')
print(f'Total images: {len(json_data["images"])}')
