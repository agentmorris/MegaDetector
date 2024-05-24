"""

camtrap_dp_to_coco.py

Parse a very limited subset of the Camtrap DP data package format:

https://camtrap-dp.tdwg.org/

...and convert to COCO format.  Assumes that all required metadata files have been
put in the same directory (which is standard).

Does not currently parse bounding boxes, just attaches species labels to images.

Currently supports only sequence-level labeling.

"""

#%% Imports and constants

import os
import json
import pandas as pd

from dateutil import parser as dateparser

from collections import defaultdict


#%% Functions

def camtrap_dp_to_coco(camtrap_dp_folder,output_file=None):
    """
    Convert the Camtrap DP package in [camtrap_dp_folder] to COCO.
    
    Does not validate images, just converts.  Use integrity_check_json_db to validate
    the resulting COCO file.  
    
    Optionally writes the results to [output_file]
    """
    
    required_files = ('datapackage.json','deployments.csv','events.csv','media.csv','observations.csv')
    
    for fn in required_files:
        fn_abs = os.path.join(camtrap_dp_folder,fn)
        assert os.path.isfile(fn_abs), 'Could not find required file {}'.format(fn_abs)
        
    with open(os.path.join(camtrap_dp_folder,'datapackage.json'),'r') as f:
        datapackage = json.load(f)
        
    assert datapackage['profile'] == 'https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0/camtrap-dp-profile.json', \
        'I only know how to parse Camtrap DP 1.0 packages'

    deployments_file = None
    events_file = None
    media_file = None
    observations_file = None
    
    resources = datapackage['resources']
    for r in resources:
        if r['name'] == 'deployments':
            deployments_file = r['path']
        elif r['name'] == 'media':
            media_file = r['path']
        elif r['name'] == 'events':
            events_file = r['path']
        elif r['name'] == 'observations':
            observations_file = r['path']

    assert deployments_file is not None, 'No deployment file specified'
    assert events_file is not None, 'No events file specified'
    assert media_file is not None, 'No media file specified'
    assert observations_file is not None, 'No observation file specified'
    
    deployments_df = pd.read_csv(os.path.join(camtrap_dp_folder,deployments_file))
    events_df = pd.read_csv(os.path.join(camtrap_dp_folder,events_file))
    media_df = pd.read_csv(os.path.join(camtrap_dp_folder,media_file))
    observations_df = pd.read_csv(os.path.join(camtrap_dp_folder,observations_file))
    
    print('Read {} deployment lines'.format(len(deployments_df)))
    print('Read {} events lines'.format(len(events_df)))
    print('Read {} media lines'.format(len(media_df)))
    print('Read {} observation lines'.format(len(observations_df)))
    
    media_id_to_media_info = {}
    
    # i_row = 0; row = media_df.iloc[i_row]
    for i_row,row in media_df.iterrows():
        media_info = {}
        media_info['file_name'] = os.path.join(row['filePath'],row['fileName']).replace('\\','/')
        media_info['location'] = row['deploymentID']
        media_info['id'] = row['mediaID']
        media_info['datetime'] = row['timestamp']
        media_info['datetime'] = dateparser.parse(media_info['datetime'])
        media_info['frame_num'] = -1
        media_info['seq_num_frames'] = -1
        media_id_to_media_info[row['mediaID']] = media_info
        
    event_id_to_media_ids = defaultdict(list)
    
    # i_row = 0; row = events_df.iloc[i_row]
    for i_row,row in events_df.iterrows():
        media_id = row['mediaID']
        assert media_id in media_id_to_media_info
        event_id_to_media_ids[row['eventID']].append(media_id)
    
    event_id_to_category_names = defaultdict(set)
    
    # i_row = 0; row = observations_df.iloc[i_row]
    for i_row,row in observations_df.iterrows():
        
        if row['observationLevel'] != 'event':
            raise ValueError("I don't know how to parse image-level events yet")
            
        if row['observationType'] == 'blank':
            event_id_to_category_names[row['eventID']].add('empty')
        elif row['observationType'] == 'unknown':
            event_id_to_category_names[row['eventID']].add('unknown')
        elif row['observationType'] == 'human':
            assert row['scientificName'] == 'Homo sapiens'
            event_id_to_category_names[row['eventID']].add(row['scientificName'])
        else:
            assert row['observationType'] == 'animal'
            assert isinstance(row['scientificName'],str)
            event_id_to_category_names[row['eventID']].add(row['scientificName'])
    
    # Sort images within an event into frame numbers
    #
    # event_id = next(iter(event_id_to_media_ids))
    for event_id in event_id_to_media_ids.keys():
        media_ids_this_event = event_id_to_media_ids[event_id]
        media_info_this_event = [media_id_to_media_info[media_id] for media_id in media_ids_this_event]
        media_info_this_event = sorted(media_info_this_event, key=lambda x: x['datetime'])
        for i_media,media_info in enumerate(media_info_this_event):
            media_info['frame_num'] = i_media
            media_info['seq_num_frames'] = len(media_info_this_event)
            media_info['seq_id'] = event_id
            
    # Create category names
    category_name_to_category_id = {'empty':0}
    for event_id in event_id_to_category_names:
        category_names_this_event = event_id_to_category_names[event_id]
        for name in category_names_this_event:
            if name not in category_name_to_category_id:
                category_name_to_category_id[name] = len(category_name_to_category_id)
    
    # Move everything into COCO format
    images = list(media_id_to_media_info.values())
    
    categories = []
    for name in category_name_to_category_id:
        categories.append({'name':name,'id':category_name_to_category_id[name]})
    info = {'version':1.0,'description':datapackage['name']}
    
    # Create annotations
    annotations = []
    
    for event_id in event_id_to_media_ids.keys():
        i_ann = 0
        media_ids_this_event = event_id_to_media_ids[event_id]
        media_info_this_event = [media_id_to_media_info[media_id] for media_id in media_ids_this_event]
        categories_this_event = event_id_to_category_names[event_id]
        for im in media_info_this_event:
            for category_name in categories_this_event:
                ann = {}
                ann['id'] = event_id + '_' + str(i_ann)
                i_ann += 1
                ann['image_id'] = im['id']
                ann['category_id'] = category_name_to_category_id[category_name]
                ann['sequence_level_annotation'] = True
                annotations.append(ann)
    
    coco_data = {}
    coco_data['images'] = images
    coco_data['annotations'] = annotations
    coco_data['categories'] = categories
    coco_data['info'] = info
    
    for im in coco_data['images']:
        im['datetime'] = str(im['datetime'] )
        
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(coco_data,f,indent=1)
    
    return coco_data
            
    
#%% Interactive driver

if False:

    pass

    #%%
    
    camtrap_dp_folder = r'C:\temp\pilot2\pilot2'
    coco_file = os.path.join(camtrap_dp_folder,'test-coco.json')
    coco_data = camtrap_dp_to_coco(camtrap_dp_folder,
                                   output_file=coco_file)
    
    #%% Validate
    
    from megadetector.data_management.databases.integrity_check_json_db import \
        integrity_check_json_db, IntegrityCheckOptions
    
    options = IntegrityCheckOptions()
    
    options.baseDir = camtrap_dp_folder
    options.bCheckImageSizes = False
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = True
    options.iMaxNumImages = -1
    options.nThreads = 1
    options.verbose = True
    
    sortedCategories, data, errorInfo = integrity_check_json_db(coco_file,options)

    #%% Preview
    
    from megadetector.visualization.visualize_db import DbVizOptions, visualize_db
    
    options = DbVizOptions()
    options.parallelize_rendering = True
    options.parallelize_rendering_with_threads = True
    options.parallelize_rendering_n_cores = 10
    
    preview_dir = r'c:\temp\camtrapdp-preview'
    htmlOutputFile,image_db = visualize_db(coco_file, preview_dir, camtrap_dp_folder, options=options)
    
    from megadetector.utils.path_utils import open_file
    open_file(htmlOutputFile)
    
    
#%% Command-line driver

# TODO
