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

known_unmapped_labels = set(['WCS Camera Traps:#ref!'])

debug_max_images_per_dataset = -1
if debug_max_images_per_dataset > 0:
    print('Running in debug mode')
    output_file = output_file.replace('.csv','_debug.csv')


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

taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)


#%% Build a dictionary that maps each [dataset,query] pair to the full taxonomic label set

ds_label_to_taxonomy = {}

# i_row = 0; row = taxonomy_df.iloc[i_row]
for i_row,row in taxonomy_df.iterrows():
    
    ds_label = row['dataset_name'] + ':' + row['query']
    assert ds_label.strip() == ds_label
    assert ds_label not in ds_label_to_taxonomy
    ds_label_to_taxonomy[ds_label] = row.to_dict()
    

#%% Process annotations for each dataset

# Takes several hours

# The order of these headers needs to match the order in which fields are added later in this cell;
# don't mess with this order.
header = ['dataset_name','url_gcp','url_aws','url_azure',
          'image_id','sequence_id','location_id','frame_num',
          'original_label','scientific_name','common_name','datetime','annotation_level']

taxonomy_levels_to_include = \
    ['kingdom','phylum','subphylum','superclass','class','subclass','infraclass','superorder','order',
     'suborder','infraorder','superfamily','family','subfamily','tribe','genus','species','subspecies',
     'variety']
    
header.extend(taxonomy_levels_to_include)

missing_annotations = set()

def clearnan(v):
    if isinstance(v,float):
        assert np.isnan(v)
        v = ''
    assert isinstance(v,str)
    return v

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
        for i_image,im in enumerate(images):
            
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
                
            def has_valid_datetime(im):
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
            if (has_valid_datetime(im)):
                
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
                row.append(clearnan(taxonomy_labels['scientific_name']))
                row.append(clearnan(taxonomy_labels['common_name']))
                row.append(dt_string)
                row.append(annotation_level)
                
                for s in taxonomy_levels_to_include:
                    row.append(clearnan(taxonomy_labels[s]))
                
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

print('Processed {} datasets'.format(len(metadata_table)))


#%% Read the .csv back

df = pd.read_csv(output_file)
print('Read {} rows from {}'.format(len(df),output_file))


#%% Do some post-hoc integrity checking

# Takes ~10 minutes without using apply()

tqdm.pandas()

def isint(v):
    return isinstance(v,int) or isinstance(v,np.int64)

valid_annotation_levels = set(['sequence','image','unknown'])

# Collect a list of locations within each dataset; we'll use this
# in the next cell to look for datasets that only have a single location
dataset_name_to_locations = defaultdict(set)

def check_row(row):
    
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
        assert isint(row['frame_num']) and row['frame_num'] >= -1

    ds_name = row['dataset_name']
    dataset_name_to_locations[ds_name].add(row['location_id'])
    
# Faster, but more annoying to debug
if False:
    
    df.progress_apply(check_row, axis=1)

else:
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in tqdm(df.iterrows(),total=len(df)):
        check_row(row)


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

    if len(non_empty_rows) == 0:
        print('No non-empty images available for {}'.format(ds_name))
    elif len(non_empty_rows) > n_non_empty_images_per_dataset:
        non_empty_rows = non_empty_rows.sample(n=n_non_empty_images_per_dataset)
    images_to_download.extend(non_empty_rows.to_dict('records'))
    
 # ...for each dataset

print('Selected {} total images'.format(len(images_to_download)))


#%% Download images (prep)

# Expect a few errors for images with human or vehicle labels (or things like "ignore" that *could* be humans)

preferred_cloud = 'aws'

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

from megadetector.utils.url_utils import parallel_download_urls
download_results = parallel_download_urls(url_to_target_file,verbose=False,overwrite=True,
                                          n_workers=20,pool_type='thread')


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

zipped_output_file = zip_file(output_file,verbose=True)

print('Zipped {} to {}'.format(output_file,zipped_output_file))
