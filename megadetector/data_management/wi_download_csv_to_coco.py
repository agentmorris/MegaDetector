"""

wi_download_csv_to_coco.py

Converts a .csv file from a Wildlife Insights project export to a COCO camera traps .json file.

Currently assumes that common names are unique identifiers, which is convenient but unreliable.

"""

#%% Imports and constants

import os
import json
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from megadetector.visualization import visualization_utils as vis_utils
from megadetector.utils.ct_utils import isnan

wi_extra_annotation_columns = \
    ('is_blank','identified_by','wi_taxon_id','class','order','family','genus','species','uncertainty',
             'number_of_objects','age','sex','animal_recognizable','individual_id','individual_animal_notes',
             'behavior','highlighted','markings')

wi_extra_image_columns = ('project_id','deployment_id')

def _make_location_id(project_id,deployment_id):    
    return 'project_' + str(project_id) + '_deployment_' + deployment_id

default_category_remappings = {
    'Homo Species':'Human',
    'Human-Camera Trapper':'Human',
    'No CV Result':'Unknown'
}


#%% Main function

def wi_download_csv_to_coco(csv_file_in,
                            coco_file_out=None,
                            image_folder=None,
                            validate_images=False,
                            gs_prefix=None,
                            verbose=True,
                            category_remappings=default_category_remappings):
    """
    Converts a .csv file from a Wildlife Insights project export to a COCO 
    Camera Traps .json file.
    
    Args:
        csv_file_in (str): the downloaded .csv file we should convert to COCO
        coco_file_out (str, optional): the .json file we should write; if [coco_file_out] is None, 
            uses [csv_file_in].json
        image_folder (str, optional): the folder where images live, only relevant if 
            [validate_images] is True
        validate_images (bool, optional): whether to check images for corruption and load
            image sizes; if this is True, [image_folder] must be a valid folder
        gs_prefix (str, optional): a string to remove from GS URLs to convert to path names... 
            for example, if your gs:// URLs look like:
    
            `gs://11234134_xyz/deployment/55554/dfadfasdfs.jpg`
    
            ...and you specify gs_prefix='11234134_xyz/deployment/', the filenames in
            the .json file will look like:
        
            `55554/dfadfasdfs.jpg`
        verbose (bool, optional): enable additional debug console output
        category_remappings (dict, optional): str --> str dict that maps any number of
            WI category names to output category names; for example defaults to mapping
            "Homo Species" to "Human", but leaves 99.99% of categories unchanged.        
            
    Returns: 
        dict: COCO-formatted data, identical to what's written to [coco_file_out]
    """
    
    ##%% Create COCO dictionaries
    
    category_name_to_id = {}
    category_name_to_id['empty'] = 0
    
    df = pd.read_csv(csv_file_in)
    
    print('Read {} rows from {}'.format(len(df),csv_file_in))
    
    image_id_to_image = {}
    image_id_to_annotations = defaultdict(list)
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
        
        image_id = row['image_id']
        
        if image_id not in image_id_to_image:
            
            im = {}
            image_id_to_image[image_id] = im
            
            im['id'] = image_id
            
            gs_url = row['location']
            assert gs_url.startswith('gs://')
            
            file_name = gs_url.replace('gs://','')
            if gs_prefix is not None:
                file_name = file_name.replace(gs_prefix,'')
                
            location_id = _make_location_id(row['project_id'],row['deployment_id'])
            im['file_name'] = file_name
            im['location'] = location_id
            im['datetime'] = row['timestamp']
            
            im['wi_image_info'] = {}
            for s in wi_extra_image_columns:
                im['wi_image_info'][s] = str(row[s])
            
        else:
            
            im = image_id_to_image[image_id]
            assert im['datetime'] == row['timestamp']
            location_id = _make_location_id(row['project_id'],row['deployment_id'])
            assert im['location'] == location_id
            
        category_name = row['common_name']
        if category_remappings is not None and category_name in category_remappings:
            category_name = category_remappings[category_name]
            
        if category_name == 'Blank':
            category_name = 'empty'
            assert row['is_blank'] == 1
        else:
            assert row['is_blank'] == 0
        assert isinstance(category_name,str)
        if category_name in category_name_to_id:
            category_id = category_name_to_id[category_name]
        else:
            category_id = len(category_name_to_id)
            category_name_to_id[category_name] = category_id
        
        ann = {}
        ann['image_id'] = image_id
        annotations_this_image = image_id_to_annotations[image_id]
        annotation_number = len(annotations_this_image)
        ann['id'] = image_id + '_' + str(annotation_number).zfill(2)        
        ann['category_id'] = category_id
        annotations_this_image.append(ann)
        
        extra_info = {}
        for s in wi_extra_annotation_columns:            
            v = row[s]
            if not isnan(v):
                extra_info[s] = v
        ann['wi_extra_info'] = extra_info
        
    # ...for each row
    
    images = list(image_id_to_image.values())
    categories = []
    for category_name in category_name_to_id:
        category_id = category_name_to_id[category_name]
        categories.append({'id':category_id,'name':category_name})
    annotations = []
    for image_id in image_id_to_annotations:
        annotations_this_image = image_id_to_annotations[image_id]
        for ann in annotations_this_image:
            annotations.append(ann)
    info = {'version':'1.00','description':'converted from WI export'}
    info['source_file'] = csv_file_in
    coco_data = {}
    coco_data['info'] = info
    coco_data['images'] = images
    coco_data['annotations'] = annotations
    coco_data['categories'] = categories
    
    
    ##%% Validate images, add sizes        
    
    if validate_images:
        
        print('Validating images')
        
        assert os.path.isdir(image_folder), \
            'Must specify a valid image folder if you specify validate_images=True'
        
        # TODO: trivially parallelizable
        #    
        # im = images[0]
        for im in tqdm(images):
            file_name_relative = im['file_name']
            file_name_abs = os.path.join(image_folder,file_name_relative)
            assert os.path.isfile(file_name_abs)
                
            im['corrupt'] = False
            try:
                pil_im = vis_utils.load_image(file_name_abs)
            except Exception:
                im['corrupt'] = True
            if not im['corrupt']:
                im['width'] = pil_im.width
                im['height'] = pil_im.height
    
    
    ##%% Write output json
        
    if coco_file_out is None:        
        coco_file_out = csv_file_in + '.json'
        
    with open(coco_file_out,'w') as f:
        json.dump(coco_data,f,indent=1)


    ##%% Validate output
    
    from megadetector.data_management.databases.integrity_check_json_db import \
        IntegrityCheckOptions,integrity_check_json_db
    options = IntegrityCheckOptions()
    options.baseDir = image_folder
    options.bCheckImageExistence = True
    options.verbose = verbose
    _ = integrity_check_json_db(coco_file_out,options)
    
    return coco_data

# ...def wi_download_csv_to_coco(...)    


#%% Interactive driver

if False:
    
    #%%
    
    base_folder = r'a/b/c'
    csv_file_in = os.path.join(base_folder,'images.csv')
    coco_file_out = None
    gs_prefix = 'a_b_c_main/'
    image_folder = os.path.join(base_folder,'images')
    validate_images = False
    verbose = True
    category_remappings = default_category_remappings


#%% Command-line driver

# TODO
