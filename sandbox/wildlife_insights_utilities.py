"""

wildlife_insights_utilities.py

Functions related to working with the WI insights platform, specifically for:
    
* Retrieving images based on .csv downloads
* Pushing results to the ProcessCVResponse() API (requires an API key)
* Working with WI taxonomy records and geofencing data

"""

#%% Imports and constants

import os
import requests
import json

import numpy as np
import pandas as pd

from collections import defaultdict
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from tqdm import tqdm

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.path_utils import recursive_file_list, find_images

md_category_id_to_name = {'1':'animal','2':'person','3':'vehicle'}
md_category_name_to_id = invert_dictionary(md_category_id_to_name)

# Only used when pushing results directly to the platform via the API; any detections we want 
# to show in the UI should have at least this confidence value.
min_md_output_confidence = 0.25

# Fields expected to be present in a valid WI result
wi_result_fields = ['wi_taxon_id','class','order','family','genus','species','common_name']


#%% Miscellaneous WI support functions

def is_valid_prediction_string(s):
    """
    Prediction strings look like:
    
    '90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent'
    
    Args:
        s (str): the string to be tested for validity
        
    Returns:
        bool: True if this looks more or less like a WI prediction string
    """
    
    return isinstance(s,str) and (len(s.split(';')) == 7) and (s == s.lower())


def wi_result_to_prediction_string(r):
    """
    Converts the dict [r] - typically loaded from a row in a downloaded .csv file - to
    a valid prediction string, e.g.:
        
    1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal
    90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent
    
    Args:
        r (dict): dict containing WI prediction information, with at least the fields
            specified in wi_result_fields.
    
    Returns:
        str: the result in [r], as a semicolon-delimited prediction string
    """
    
    values = []
    for field in wi_result_fields:
        if isinstance(r[field],str):
            values.append(r[field].lower())
        else:
            assert isinstance(r[field],float) and np.isnan(r[field])
            values.append('')
    s = ';'.join(values)
    assert is_valid_prediction_string(s)
    return s
        

def compare_values(v0,v1):
    """
    Utility function for comparing two values when we want to return True if both
    values are NaN.
    
    Args:
        v0 (object): the first value to compare
        v1 (object): the second value to compare
        
    Returns:
        bool: True if v0 == v1, or if both v0 and v1 are NaN
    """
    if isinstance(v0,float) and isinstance(v1,float) and np.isnan(v0) and np.isnan(v1):
        return True
    return v0 == v1


def record_is_unidentified(record):
    """
    A record is considered "unidentified" if the "identified by" field is either NaN or "computer vision"
    
    Args:
        record (dict): dict representing a WI result loaded from a .csv file, with at least the 
            field "identified_by"
    
    Returns:
        bool: True if the "identified_by" field is either NaN or a string indicating that this 
        record has not yet been human-reviewed.
    """    
    
    identified_by = record['identified_by']
    assert isinstance(identified_by,float) or isinstance(identified_by,str)
    if isinstance(identified_by,float):
        assert np.isnan(identified_by)
        return True
    else:
        return identified_by == 'Computer vision'


def record_lists_are_identical(records_0,records_1,verbose=False):
    """
    Takes two lists of records in the form returned by read_images_from_download_bundle and
    determines whether they are the same.
    
    Args:
        records_0 (list of dict): the first list of records to compare
        records_1 (list of dict): the second list of records to compare
        verbose (bool, optional): enable additional debug output
        
    Returns:
        bool: True if the two lists are identical
    """
    
    if len(records_0) != len(records_1):
        return False    
    
    # i_record = 0; record_0 = records_0[i_record]
    for i_record,record_0 in enumerate(records_0):
        record_1 = records_1[i_record]
        assert set(record_0.keys()) == set(record_1.keys())
        for k in record_0.keys():
            if not compare_values(record_0[k],record_1[k]):
                if verbose:
                    print('Image ID: {} ({})\nRecord 0/{}: {}\nRecord 1/{}: {}'.format(
                        record_0['image_id'],record_1['image_id'],
                        k,record_0[k],k,record_1[k]))
                return False
            
    return True
    
    
#%% Functions for managing WI downloads

def read_sequences_from_download_bundle(download_folder):
    """
    Reads sequences.csv from [download_folder], returning a list of dicts.  This is a
    thin wrapper around pd.read_csv, it's just here for future-proofing.
    
    Args:
        download_folder (str): a folder containing exactly one file called sequences.csv, typically
            representing a Wildlife Insights download bundle.
    
    Returns:
        list of dict: a direct conversion of the .csv file to a list of dicts
    """

    print('Reading sequences from {}'.format(download_folder))
    
    sequence_list_files = os.listdir(download_folder)
    sequence_list_files = \
        [fn for fn in sequence_list_files if fn == 'sequences.csv']
    assert len(sequence_list_files) == 1, \
        'Could not find sequences.csv in {}'.format(download_folder)
    
    sequence_list_file = os.path.join(download_folder,sequence_list_files[0])

    df = pd.read_csv(sequence_list_file)
    sequence_records = df.to_dict('records')
    return sequence_records
    
    
def read_images_from_download_bundle(download_folder):
    """
    Reads all images_...csv files from [download_folder], returns a dict mapping image IDs
    to a list of dicts that describe each image.  It's a list of dicts rather than a single dict
    because images may appear more than once.
    
    Args:
        download_folder (str): a folder containing one or more images_....csv files, typically
            representing a Wildlife Insights download bundle.
    
    Returns:
        dict: Maps image GUIDs to dicts with at least the following fields:
            * project_id (int)
            * deployment_id (str)
            * image_id (str, should match the key)
            * filename (str, the filename without path at the time of upload)
            * location (str, starting with gs://)
            
        May also contain clasification fields: wi_taxon_id (str), species, etc.        
    """
    
    print('Reading images from {}'.format(download_folder))
    
    ##%% Find lists of images
    
    image_list_files = os.listdir(download_folder)
    image_list_files = \
        [fn for fn in image_list_files if fn.startswith('images_') and fn.endswith('.csv')]
    image_list_files = \
        [os.path.join(download_folder,fn) for fn in image_list_files]
    print('Found {} image list files'.format(len(image_list_files)))


    ##%% Read lists of images by deployment

    image_id_to_image_records = defaultdict(list)
    
    # image_list_file = image_list_files[0]
    for image_list_file in image_list_files:
    
        print('Reading images from list file {}'.format(
            os.path.basename(image_list_file)))
        
        df = pd.read_csv(image_list_file)
        
        # i_row = 0; row = df.iloc[i_row]
        for i_row,row in tqdm(df.iterrows(),total=len(df)):
            
            row_dict = row.to_dict()
            image_id = row_dict['image_id']            
            image_id_to_image_records[image_id].append(row_dict)
            
        # ...for each image
        
    # ...for each list file    

    deployment_ids = set()
    for image_id in image_id_to_image_records:
        image_records = image_id_to_image_records[image_id]
        for image_record in image_records:
            deployment_ids.add(image_record['deployment_id'])
        
    print('Found {} rows in {} deployments'.format(
        len(image_id_to_image_records),
        len(deployment_ids)))

    return image_id_to_image_records


def find_images_in_identify_tab(download_folder_with_identify,download_folder_excluding_identify):
    """
    Based on extracted download packages with and without the "exclude images in 'identify' tab 
    checkbox" checked, figure out which images are in the identify tab.  Returns a list of dicts (one
    per image).
    
    Args:
        download_folder_with_identify (str): the folder containing the download bundle that
            includes images from the "identify" tab
        download_folder_excluding_identify (str): the folder containing the download bundle that
            excludes images from the "identify" tab
        
    Returns:
        list of dict: list of image records that are present in the identify tab
    """
    
    ##%% Read data (~30 seconds)
    
    image_id_to_image_records_with_identify = \
        read_images_from_download_bundle(download_folder_with_identify)
    image_id_to_image_records_excluding_identify = \
        read_images_from_download_bundle(download_folder_excluding_identify)
    
    
    ##%% Find images that have not been identified
    
    all_image_ids_with_identify = set(image_id_to_image_records_with_identify.keys())
    all_image_ids_excluding_identify = set(image_id_to_image_records_excluding_identify.keys())
    
    image_ids_in_identify_tab = all_image_ids_with_identify.difference(all_image_ids_excluding_identify)
    
    assert len(image_ids_in_identify_tab) == \
        len(all_image_ids_with_identify) - len(all_image_ids_excluding_identify)
        
    print('Found {} images with identify, {} in identify tab, {} excluding'.format(
        len(all_image_ids_with_identify), 
        len(image_ids_in_identify_tab),
        len(all_image_ids_excluding_identify)))
    
    image_records_in_identify_tab = []
    deployment_ids_for_downloaded_images = set()    
    
    for image_id in image_ids_in_identify_tab:
        image_records_this_image = image_id_to_image_records_with_identify[image_id]
        assert len(image_records_this_image) > 0
        image_records_in_identify_tab.extend(image_records_this_image)
        for image_record in image_records_this_image:
            deployment_ids_for_downloaded_images.add(image_record['deployment_id'])
        
    print('Found {} records for {} unique images in {} deployments'.format(
        len(image_records_in_identify_tab),
        len(image_ids_in_identify_tab),
        len(deployment_ids_for_downloaded_images)))

    return image_records_in_identify_tab

# ...def find_images_in_identify_tab(...)


def write_download_commands(image_records_to_download,
                            download_dir_base,
                            force_download=False,
                            n_download_workers=25,
                            download_command_file_base=None):
    """
    Given a list of dicts with at least the field 'location' (a gs:// URL), prepare a set of "gcloud 
    storage" commands to download images, and write those to a series of .sh scripts, along with one
    .sh script that runs all the others and blocks.
    
    gcloud commands will use relative paths.
    
    image_records_to_download can also be a dict mapping IDs to lists of records.
    
    Args:
        image_records_to_download (list of dict): list of dicts with at least the field 'location'
        download_dir_base (str): local destination folder
        force_download (bool, optional): include gs commands even if the target file exists
        n_download_workers (int, optional): number of scripts to write (that's our hacky way
            of controlling parallelization)
        download_command_file (str, optional): path of the .sh script we should write, defaults
            to "download_wi_images.sh" in the destination folder
    """

    if isinstance(image_records_to_download,dict):
        
        all_image_records = []
        for k in image_records_to_download:
            records_this_image = image_records_to_download[k]
            all_image_records.extend(records_this_image)
        return write_download_commands(all_image_records, 
                                       download_dir_base=download_dir_base,
                                       force_download=force_download,
                                       n_download_workers=n_download_workers,
                                       download_command_file_base=download_command_file_base)
    
    ##%% Make list of gcloud storage commands
    
    if download_command_file_base is None:
        download_command_file_base = os.path.join(download_dir_base,'download_wi_images.sh')
    
    commands = []
    skipped_urls = []
    downloaded_urls = set()
    
    # image_record = image_records_to_download[0]
    for image_record in tqdm(image_records_to_download):
        
        url = image_record['location']
        if url in downloaded_urls:
            continue
        
        assert url.startswith('gs://')
        
        relative_path = url.replace('gs://','')
        abs_path = os.path.join(download_dir_base,relative_path)
        
        # Skip files that already exist
        if (not force_download) and (os.path.isfile(abs_path)):
            skipped_urls.append(url)
            continue
        
        # command = 'gsutil cp "{}" "./{}"'.format(url,relative_path)
        command = 'gcloud storage cp --no-clobber "{}" "./{}"'.format(url,relative_path)
        commands.append(command)
    
    print('Generated {} commands for {} image records'.format(
        len(commands),len(image_records_to_download)))
    
    print('Skipped {} URLs'.format(len(skipped_urls)))
    
    
    ##%% Write those commands out to n .sh files
    
    commands_by_script = split_list_into_n_chunks(commands,n_download_workers)
    
    local_download_commands = []
    
    output_dir = os.path.dirname(download_command_file_base)
    os.makedirs(output_dir,exist_ok=True)
    
    # Write out the download script for each chunk
    # i_script = 0
    for i_script in range(0,n_download_workers):
        download_command_file = insert_before_extension(download_command_file_base,str(i_script).zfill(2))
        local_download_commands.append(os.path.basename(download_command_file))
        with open(download_command_file,'w',newline='\n') as f:
            for command in commands_by_script[i_script]:
                f.write(command + '\n')
    
    # Write out the main download script
    with open(download_command_file_base,'w',newline='\n') as f:
        for local_download_command in local_download_commands:
            f.write('./' + local_download_command + ' &\n')
        f.write('wait\n')
        f.write('echo done\n')

# ...def write_download_commands(...)


#%% Functions and constants related to pushing results to the DB

# Sample payload for validation

sample_update_payload = {
    
    "predictions": [
        {
          "project_id": "1234",
          "ignore_data_file_checks": True,
          "prediction": "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
          "prediction_score": 0.81218224763870239,
            "classifications": {
                "classes": [
                    "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
                    "b1352069-a39c-4a84-a949-60044271c0c1;aves;;;;;bird",
                    "90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent",
                    "f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal",
                    "ac068717-6079-4aec-a5ab-99e8d14da40b;mammalia;rodentia;sciuridae;dremomys;rufigenis;red-cheeked squirrel"
                ],
                "scores": [
                    0.81218224763870239,
                    0.1096673980355263,
                    0.02707692421972752,
                    0.00771023565903306,
                    0.0049269795417785636
                ]
            },
            "detections": [
                {
                    "category": "1",
                    "label": "animal",
                    "conf": 0.181,
                    "bbox": [
                        0.02421,
                        0.35823999999999989,
                        0.051560000000000009,
                        0.070826666666666746
                    ]
                }
            ],
            "model_version": "3.1.2",
            "prediction_source": "manual_update",
            "data_file_id": "2ea1d2b2-7f84-43f9-af1f-8be0e69c7015"
        }
    ]
}

blank_prediction_string = 'f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank'
no_cv_result_prediction_string = 'f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result'
rodent_prediction_string = '90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent'
mammal_prediction_string = 'f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal'
animal_prediction_string = '1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal'
human_prediction_string = '990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human'

process_cv_response_url = 'https://placeholder'


def prepare_data_update_auth_headers(auth_token_file):
    """
    auth_token_file should be a single-line .txt file containing a write-enabled
    API token.
    """

    with open(auth_token_file,'r') as f:
        auth_token = f.read()
            
    headers = {
        'Authorization': 'Bearer ' + auth_token,
        'Content-Type': 'application/json'
    }
    
    return headers


def push_results_for_images(payload,
                            headers,
                            url=process_cv_response_url,
                            verbose=False):
    """
    Push results for one or more images represented in [payload] to the process_cv_response API.
    """
    
    if verbose:
        print('Sending header {} to URL {}'.format(
            headers,url))
        
    response = requests.post(url, headers=headers, json=payload)

    # Check the response status code
    if response.status_code in (200,201):  
        if verbose:
            print('Successfully pushed results for {} images'.format(len(payload['predictions'])))
            print(response.headers)
            print(str(response))
    else:
        print(f'Error: {response.status_code} {response.text}')
        
    return response.status_code


def parallel_push_results_for_images(payloads,
                                     headers,
                                     url=process_cv_response_url,
                                     verbose=False,
                                     pool_type='thread',
                                     n_workers=10):
    """
    Push results for the list of payloads in [payloads] to the process_cv_response API, parallelized
    over multiple workers.
    """
    
    if n_workers == 1:
        
        results = []
        for payload in payloads:
            results.append(push_results_for_images(payload,
                                                   headers=headers,
                                                   url=url,
                                                   verbose=verbose))
        return results
    
    else:
        
        assert pool_type in ('thread','process')
        
        if pool_type == 'thread':
            pool_string = 'thread'
            pool = ThreadPool(n_workers)
        else:
            pool_string = 'process'
            pool = Pool(n_workers)
    
        print('Created a {} pool of {} workers'.format(
            pool_string,n_workers))
        
        results = list(tqdm(pool.imap(
            partial(push_results_for_images,headers=headers,url=url,verbose=verbose),payloads), 
            total=len(payloads)))
        
        assert len(results) == len(payloads)
        return results


def generate_payload_with_replacement_detections(wi_result,
                                                 detections,
                                                 prediction_score=0.9,
                                                 model_version='3.1.2',
                                                 prediction_source='manual_update'):
    """
    Generate a payload for a single image that keeps the classifications from 
    [wi_result], but replaces the detections with the MD-formatted list [detections].
    """
    
    payload_detections = []
    
    # detection = detections[0]    
    for detection in detections:
        detection_out = detection.copy()
        detection_out['label'] = md_category_id_to_name[detection['category']]
        if detection_out['conf'] < min_md_output_confidence:
            detection_out['conf'] = min_md_output_confidence
        payload_detections.append(detection_out)
    
    prediction_string = wi_result_to_prediction_string(wi_result)
    
    prediction = {}
    prediction['ignore_data_file_checks']  = True
    prediction['prediction'] = prediction_string
    prediction['prediction_score'] = prediction_score
    
    classifications = {}
    classifications['classes'] = [prediction_string]
    classifications['scores'] = [prediction_score]
    
    prediction['classifications'] = classifications
    prediction['detections'] = payload_detections
    prediction['model_version'] = model_version
    prediction['prediction_source'] = prediction_source
    prediction['data_file_id'] = wi_result['image_id']
    prediction['project_id'] = str(wi_result['project_id'])
    payload = {}
    payload['predictions'] = [prediction]
    
    return payload


def generate_blank_prediction_payload(data_file_id,
                                      project_id,
                                      blank_confidence=0.9,
                                      model_version='3.1.2',
                                      prediction_source='manual_update'):
    """
    Generate a payload that will set a single image to the blank classification, with
    no detections.
    """
    
    prediction = {}
    prediction['ignore_data_file_checks']  = True
    prediction['prediction'] = blank_prediction_string
    prediction['prediction_score'] = blank_confidence
    prediction['classifications'] = {}
    prediction['classifications']['classes'] = [blank_prediction_string]
    prediction['classifications']['scores'] = [blank_confidence]
    prediction['detections'] = []
    prediction['model_version'] = model_version
    prediction['prediction_source'] = prediction_source
    prediction['data_file_id'] = data_file_id
    prediction['project_id'] = project_id
    payload = {}
    payload['predictions'] = [prediction]
    
    return payload


def generate_no_cv_result_payload(data_file_id,
                                  project_id,
                                  no_cv_confidence=0.9,
                                  model_version='3.1.2',
                                  prediction_source='manual_update'):
    """
    Generate a payload that will set a single image to the blank classification, with
    no detections.
    """
    
    prediction = {}
    prediction['ignore_data_file_checks']  = True
    prediction['prediction'] = no_cv_result_prediction_string
    prediction['prediction_score'] = no_cv_confidence
    prediction['classifications'] = {}
    prediction['classifications']['classes'] = [no_cv_result_prediction_string]
    prediction['classifications']['scores'] = [no_cv_confidence]
    prediction['detections'] = []
    prediction['model_version'] = model_version
    prediction['prediction_source'] = prediction_source
    prediction['data_file_id'] = data_file_id
    prediction['project_id'] = project_id
    payload = {}
    payload['predictions'] = [prediction]
    
    return payload


def generate_payload_for_prediction_string(data_file_id,
                                           project_id,
                                           prediction_string,
                                           prediction_confidence=0.8,
                                           detections=None):
    """
    Generate a payload that will set a single image to a particular prediction, optionally
    including detections.    
    """
    
    assert is_valid_prediction_string(prediction_string), \
        'Invalid prediction string: {}'.format(prediction_string)
        
    payload_detections = []
    
    if detections is not None:
        # detection = detections[0]    
        for detection in detections:
            detection_out = detection.copy()
            detection_out['label'] = md_category_id_to_name[detection['category']]
            if detection_out['conf'] < min_md_output_confidence:
                detection_out['conf'] = min_md_output_confidence
            payload_detections.append(detection_out)
        
    prediction = {}
    prediction['ignore_data_file_checks']  = True
    prediction['prediction'] = prediction_string
    prediction['prediction_score'] = prediction_confidence
    prediction['classifications'] = {}
    prediction['classifications']['classes'] = [prediction_string]
    prediction['classifications']['scores'] = [prediction_confidence]
    prediction['detections'] = payload_detections
    prediction['model_version'] = '3.1.2'
    prediction['prediction_source'] = 'manual_update'
    prediction['data_file_id'] = data_file_id
    prediction['project_id'] = project_id
    
    payload = {}
    payload['predictions'] = [prediction]
    
    return payload


def validate_payload(payload):
    """
    Verifies that the dict [payload] is compatible with the ProcessCVResponse() API.  Throws an
    error if [payload] is invalid.
    """
    
    assert isinstance(payload,dict)
    assert len(payload.keys()) == 1 and 'predictions' in payload
    
    # prediction = payload['predictions'][0]
    for prediction in payload['predictions']:
        
        assert 'project_id' in prediction
        if not isinstance(prediction['project_id'],int):
            _ = int(prediction['project_id'])
        assert 'ignore_data_file_checks' in prediction and \
            isinstance(prediction['ignore_data_file_checks'],bool)
        assert 'prediction' in prediction and \
            isinstance(prediction['prediction'],str) and \
            len(prediction['prediction'].split(';')) == 7
        assert 'prediction_score' in prediction and \
            isinstance(prediction['prediction_score'],float)
        assert 'model_version' in prediction and \
            isinstance(prediction['model_version'],str)
        assert 'data_file_id' in prediction and \
            isinstance(prediction['data_file_id'],str) and \
            len(prediction['data_file_id']) == 36
        assert 'classifications' in prediction and \
            isinstance(prediction['classifications'],dict)
        classifications = prediction['classifications']
        assert 'classes' in classifications and isinstance(classifications['classes'],list)
        assert 'scores' in classifications and isinstance(classifications['scores'],list)
        assert len(classifications['classes']) == len(classifications['scores'])
        for c in classifications['classes']:
            assert is_valid_prediction_string(c)            
        for score in classifications['scores']:
            assert isinstance(score,float) and score >= 0 and score <= 1.0
        assert 'detections' in prediction and isinstance(prediction['detections'],list)
        
        for detection in prediction['detections']:
            
            assert isinstance(detection,dict)
            assert 'category' in detection and detection['category'] in ('1','2','3')
            assert 'label' in detection and detection['label'] in ('animal','person','vehicle')
            assert 'conf' in detection and \
                isinstance(detection['conf'],float) and \
                detection['conf'] >= 0 and detection['conf'] <= 1.0
            assert 'bbox' in detection and \
                isinstance(detection['bbox'],list) and \
                len(detection['bbox']) == 4
                
         # ...for each detection
         
    # ...for each prediction
    
# ...def validate_payload(...)


#%% Validate constants

# ...at the time this module gets loaded.

blank_payload = generate_blank_prediction_payload('70ede9c6-d056-4dd1-9a0b-3098d8113e0e','1234')
validate_payload(sample_update_payload)
validate_payload(blank_payload)


#%% Functions and constants related to working with batch predictions

def get_kingdom(prediction_string):
    """
    Return the kingdom field from a WI prediction string
    
    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format
        
    Returns:
        str: the kingdom field from the input string
    """
    tokens = prediction_string.split(';')
    return tokens[1]


def is_human_classification(prediction_string):
    """
    Determines whether the input string represents a human classification, which includes a variety
    of common names (hiker, person, etc.)
    
    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format
        
    Returns:
        bool: whether this string corresponds to a human category
    """
    return prediction_string == human_prediction_string or 'homo;sapiens' in prediction_string
    

def is_animal_classification(prediction_string):
    """
    Determines whether the input string represents an animal classification, which excludes, e.g.,
    humans, blanks, vehicles, unknowns
    
    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format
        
    Returns:
        bool: whether this string corresponds to an animal category
    """
    
    if prediction_string == animal_prediction_string:
        return True
    if prediction_string == human_prediction_string or 'homo;sapiens' in prediction_string:
        return False
    if prediction_string == blank_prediction_string:
        return False
    if prediction_string == no_cv_result_prediction_string:
        return False
    if len(get_kingdom(prediction_string)) == 0:
        return False
    return True
    

def generate_md_formatted_results_from_vertex_ai_results(image_folder,json_file):
    """
    Convert results in the WI .json format to MD/Timelapse format.
    """
    ##%% Read predictions
    
    assert isinstance(json_file,str)
    assert os.path.isfile(json_file)
    with open(json_file,'r') as f:
        filepath_to_prediction = json.load(f)
        
    print('\nRead {} predictions'.format(len(filepath_to_prediction)))
    
    
    ##%% Enumerate image files
        
    image_files = find_images(image_folder,return_relative_paths=True,recursive=True)
    print('Enumerated {} images'.format(len(image_files)))    
    assert len(image_files) == len(filepath_to_prediction)
            

    ##%% Create MD results
    
    detection_categories = {"1": "animal", "2": "person", "3": "vehicle" }
    classification_category_name_to_id = {}
    
    images = []
    
    image_folder = image_folder.replace('\\','/')
    if image_folder.endswith('/'):
        image_folder = image_folder[0:-1]
        
    # i_image = 0; image_fn_relative = image_files[i_image]
    for i_image,image_fn_relative in enumerate(image_files):
        
        filepath = image_folder + '/' + image_fn_relative
        prediction = filepath_to_prediction[filepath]
        
        im = {}
        im['file'] = image_fn_relative
        
        ## Process detections
        
        im['detections'] = []        
        detection_info = prediction['detection']
        n_detections = len(detection_info['detection_scores'])
        assert n_detections == len(detection_info['detection_classes'])
        assert n_detections == len(detection_info['detection_boxes'])
        
        # i_det = 0
        for i_det in range(0,n_detections):
            det = {}
            det['conf'] = detection_info['detection_scores'][i_det]            
            box = detection_info['detection_boxes'][i_det]
            x = box[1]
            y = box[0]
            w = box[3] - box[1]
            h = box[2] - box[0]
            det['bbox'] = [x,y,w,h]
            
            prediction_string = detection_info['detection_classes'][i_det]
            if is_animal_classification(prediction_string):
                category_id = '1'
            elif is_human_classification(prediction_string):
                category_id = '2'
            else:
                # Not trying to do something very elegant here
                category_id = '1'
            det['category'] = category_id        
            im['detections'].append(det)
        
        
        ## Process classifications
        
        classification_info = prediction['classifier']        
        classification_classes = classification_info['classifier_classes']
        classification_scores = classification_info['classifier_scores']
        assert len(classification_classes) > 0
        assert len(classification_classes) == len(classification_scores)
        
        prediction_category_string = prediction['data'][0]['tag']
        prediction_score = prediction['data'][0]['value']
        
        category_name = prediction_category_string.split(';')[-1]
        if category_name not in classification_category_name_to_id:
            classification_category_name_to_id[category_name] = str(len(classification_category_name_to_id))
        classification_category_id = classification_category_name_to_id[category_name]
        classification_conf = prediction_score
        
        # Create a fake detection for this case
        if len(im['detections']) == 0:            
            det = {}
            det['conf'] = prediction_score
            det['bbox'] = [0,0,1,1]
            det['category'] = '1'
            im['detections'] = [det]
            
    
        # Naively assign any animal classification category to all animal detections, don't do anything to human/vehicle boxes
        for det in im['detections']:
            if is_animal_classification(prediction_category_string) and det['category'] == '1':
                det['classifications'] = [[classification_category_id,classification_conf]]
        
        images.append(im)
        
    classification_categories = {}
    for category_name in classification_category_name_to_id:
        category_id = classification_category_name_to_id[category_name]
        classification_categories[category_id] = category_name
        
    info = {}
    info['format_version'] = 1.4
    info['detector'] = 'md_v5a.0.0.pt'
    info['classifier_metadata'] = 'wi_vertex_ai'
    
    d = {}
    d['images'] = images
    d['classification_categories'] = classification_categories
    d['detection_categories'] = detection_categories
    d['info'] = info
    
    return d


def generate_md_formatted_results_from_batch_jsonl(image_folder,
                                                   jsonl_folder,
                                                   bucket_prefix=''):
    """
    Given a folder of jsonl files produced by the WI batch API, or a single .json file
    generated locallly, generate MD-formatted results.
    
    If the filenames in the .json file have a prefx that should be appended to each
    relative image path (either a bucket prefix or an absolute folder prefix), provide
    that as [bucket_prefix].  Currently this is literally prepended as a string, so it's
    the responsbility of the caller to make sure that the / vs. \\ convention in the prefix
    is consistent with whatever is in the results file.
    """
    
    ##%% Read predictions
    
    assert isinstance(jsonl_folder,str)
    
    if os.path.isdir(jsonl_folder):
        
        # Enumerate jsonl files
        #
        # Filenames look like:
        #
        # prediction.results-00089-of-00090
        jsonl_files = recursive_file_list(jsonl_folder,recursive=False,return_relative_paths=False)
        jsonl_files = [fn for fn in jsonl_files if '.results' in fn]
        
        filepath_to_prediction = {}
        # jsonl_fn = jsonl_files[0]
        for jsonl_fn in tqdm(jsonl_files):
            with open(jsonl_fn,'r') as f:
                lines = f.readlines()
            # line = lines[0]
            for line in lines:
                d = json.loads(line)
                instance = d['instance']
                assert 'filepath' in instance
                assert instance['filepath'] == d['prediction']['filepath']            
                filepath_to_prediction[instance['filepath']] = d['prediction']
                
    elif os.path.isfile(jsonl_folder):
        
        with open(jsonl_folder,'r') as f:
            d = json.load(f)
        assert isinstance(d,dict) and 'predictions' in d, \
            '{} does not appear to be a valid predictions file'.format(jsonl_folder)
            
        filepath_to_prediction = {}
        
        predictions = d['predictions']
        assert isinstance(predictions,list)
        
        # prediction = predictions[0]
        for prediction in predictions:
            filepath_to_prediction[prediction['filepath'].replace('\\','/')] = prediction
    
    else:
        
        raise ValueError('Could not find prediction results at {}'.format(jsonl_folder))
        
    print('\nRead {} predictions'.format(len(filepath_to_prediction)))
    
    
    ##%% Enumerate image files
        
    image_files = find_images(image_folder,return_relative_paths=True,recursive=True)
    print('Enumerated {} images'.format(len(image_files)))    
    assert len(image_files) == len(filepath_to_prediction)
            

    ##%% Create MD results
    
    detection_categories = {"1": "animal", "2": "person", "3": "vehicle" }
    classification_category_name_to_id = {}
    
    images = []
    
    # image_fn_relative = image_files[0]
    for i_image,image_fn_relative in enumerate(image_files):
        
        filepath = (bucket_prefix + image_fn_relative).replace('\\','/')
        prediction = filepath_to_prediction[filepath]
        im = {}
        im['file'] = image_fn_relative
        
        if 'detections' not in prediction:
            assert ('failures' in prediction) and (len(prediction['failures']) > 0)
            im['failure'] = str(prediction['failures'])
            images.append(im)
            print('Failure for image {}'.format(image_fn_relative))
            continue
                
        im['detections'] = prediction['detections']
        for det in im['detections']:
            for s in ['category','conf','bbox']:
                assert s in det
        
        # Not using these for now
        classifications = prediction['classifications']
        classification_classes = classifications['classes']
        classification_scores = classifications['scores']
        assert len(classification_classes) > 0
        assert len(classification_classes) == len(classification_scores)
        
        prediction_category_string = prediction['prediction']
        
        category_name = prediction_category_string.split(';')[-1]
        if category_name not in classification_category_name_to_id:
            classification_category_name_to_id[category_name] = str(len(classification_category_name_to_id))
        classification_category_id = classification_category_name_to_id[category_name]
        classification_conf = prediction['prediction_score']
        
        if len(im['detections']) > 0:
    
            """
            blank_prediction_string = 'f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank'
            no_cv_result_prediction_string = 'f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result'
            rodent_prediction_string = '90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent'
            mammal_prediction_string = 'f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal'
            animal_prediction_string = '1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal'
            human_prediction_string = '990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human'
            """
    
            # Naively assign any animal classification category to all animal detections, don't do anything to human/vehicle boxes
            for det in im['detections']:
                if is_animal_classification(prediction_category_string) and det['category'] == '1':
                    det['classifications'] = [[classification_category_id,classification_conf]]
            
        images.append(im)
    
    # ...for each image
    
    classification_categories = {}
    for category_name in classification_category_name_to_id:
        category_id = classification_category_name_to_id[category_name]
        classification_categories[category_id] = category_name
        
    info = {}
    info['format_version'] = 1.4
    info['detector'] = 'md_v5a.0.0.pt'
    # info['classifier_metadata'] = 'wi_2024.10.12'
    
    d = {}
    d['images'] = images
    d['classification_categories'] = classification_categories
    d['detection_categories'] = detection_categories
    d['info'] = info
    
    return d

    
#%% Functions related to geofencing and taxonomy mapping

import codecs

# This maps a taxonomy string (e.g. mammalia;cetartiodactyla;cervidae;odocoileus;virginianus) to 
# a dict with keys taxon_id, common_name, kingdom, phylum, class, order, family, genus, species
taxonomy_string_to_taxonomy_info = None
binomial_name_to_taxonomy_info = None
common_name_to_taxonomy_info = None

def taxonomy_info_to_taxonomy_string(taxonomy_info):
    
    return taxonomy_info['class'] + ';' + \
        taxonomy_info['order'] + ';'  + \
        taxonomy_info['family'] + ';'  + \
        taxonomy_info['genus'] + ';'  + \
        taxonomy_info['species']
        
def initialize_taxonomy_info(taxonomy_file,force_init=False,encoding='cp1252'):
    
    global taxonomy_string_to_taxonomy_info 
    global binomial_name_to_taxonomy_info
    global common_name_to_taxonomy_info
    
    if (taxonomy_string_to_taxonomy_info is not None) and (not force_init):
        return
    
    """
    Taxonomy keys are taxonomy strings, e.g.:
    
    'mammalia;cetartiodactyla;cervidae;odocoileus;virginianus'
    
    Taxonomy values are extended strings w/Taxon IDs and common names, e.g.:
        
    '5c7ce479-8a45-40b3-ae21-7c97dfae22f5;mammalia;cetartiodactyla;cervidae;odocoileus;virginianus;white-tailed deer'
    """
    
    with open(taxonomy_file,encoding=encoding,errors='ignore') as f:
        taxonomy_table = json.load(f,strict=False)    
    
    # Right now I'm punting on some unusual-character issues, but here is some scrap that
    # might help address this in the future
    if False:
        with codecs.open(taxonomy_file,'r',encoding=encoding,errors='ignore') as f:
            s = f.read()        
        import unicodedata
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')    
        taxonomy_table = json.loads(s,strict=False)
        
    taxonomy_string_to_taxonomy_info = {}
    binomial_name_to_taxonomy_info = {}
    common_name_to_taxonomy_info = {}
    
    # taxonomy_string = next(iter(taxonomy_table.keys()))
    for taxonomy_string in taxonomy_table.keys():
        
        taxonomy_string = taxonomy_string.lower()
        
        taxon_info = {}
        extended_string = taxonomy_table[taxonomy_string]
        tokens = extended_string.split(';')
        assert len(tokens) == 7
        taxon_info['taxon_id'] = tokens[0]
        assert len(taxon_info['taxon_id']) == 36
        taxon_info['kingdom'] = 'animal'
        taxon_info['phylum'] = 'chordata'
        taxon_info['class'] = tokens[1]
        taxon_info['order'] = tokens[2]
        taxon_info['family'] = tokens[3]
        taxon_info['genus'] = tokens[4]
        taxon_info['species'] = tokens[5]
        taxon_info['common_name'] = tokens[6]
        
        if taxon_info['common_name'] != '':
            common_name_to_taxonomy_info[taxon_info['common_name']] = taxon_info
            
        taxonomy_string_to_taxonomy_info[taxonomy_string] = taxon_info
        if tokens[4] == '' or tokens[5] == '':
            # print('Warning: no binomial name for {}'.format(taxonomy_string))
            pass
        else:
            binomial_name = tokens[4].strip() + ' ' + tokens[5].strip()
            binomial_name_to_taxonomy_info[binomial_name] = taxon_info
        
# ...def initialize_taxonomy_info(...)


#%% Geofencing functions

taxonomy_string_to_geofencing_rules = None

# Maps lower-case countries to upper-case country codes
country_to_country_code = None

# ...and vice-versa
country_code_to_country = None

def initialize_geofencing(geofencing_file,country_code_file,force_init=False):
    
    global taxonomy_string_to_geofencing_rules
    global country_to_country_code
    global country_code_to_country
    
    if (country_to_country_code is not None) and \
        (country_code_to_country is not None) and \
        (taxonomy_string_to_geofencing_rules is not None) and \
        (not force_init):
        return
    
    country_code_df = pd.read_csv(country_code_file)
    country_to_country_code = {}
    country_code_to_country = {}
    for i_row,row in country_code_df.iterrows():
        country_to_country_code[row['name'].lower()] = row['alpha-3'].upper()
        country_code_to_country[row['alpha-3'].upper()] = row['name'].lower()
                    
    with open(geofencing_file,'r',encoding='utf-8') as f:
        taxonomy_string_to_geofencing_rules = json.load(f)
    
    """
    Geofencing keys are taxonomy strings, e.g.:
      
    'mammalia;cetartiodactyla;cervidae;odocoileus;virginianus'
    
    Geofencing values are tables mapping allow/block to country codes, optionally including region/state codes, e.g.:
        
    {'allow': {
      'ALA': [],
      'ARG': [],
      ...
      'SUR': [],
      'TTO': [],
      'USA': ['AL',
       'AR',
       'AZ',
       ...
    }
    """        
        
    # Validate
    
    # species_string = next(iter(taxonomy_string_to_geofencing_rules.keys()))
    for species_string in taxonomy_string_to_geofencing_rules.keys():
        
        species_rules = taxonomy_string_to_geofencing_rules[species_string]
        
        # Every country should *either* have allow rules or block rules, no countries 
        # currently have both        
        assert len(species_rules.keys()) == 1
        rule_type = list(species_rules.keys())[0]
        assert rule_type in ('allow','block')

        all_country_rules_this_species = species_rules[rule_type]
        for country_code in all_country_rules_this_species.keys():
            
            assert country_code in country_code_to_country
        
            region_rules = all_country_rules_this_species[country_code]
            
            # Right now we only have regional rules for the USA; these may be part of 
            # allow or block rules.
            if len(region_rules) > 0:
                assert country_code == 'USA'
    
    # ...for each species
    
# ...def initialize_geofencing(...)
    

def species_allowed_in_country(species,country,state=None,return_status=False):
    """
    Species can be a common name, a binomial name, or a species string
    
    Country can be a country name or a three-letter code
    
    State should be a two-letter code
    """

    assert taxonomy_string_to_geofencing_rules is not None, \
        'Initialize geofencing prior to species lookup'
    assert taxonomy_string_to_taxonomy_info is not None, \
        'Initialize taxonomy lookup prior to species lookup'
        
    # species = 'mammalia;cetartiodactyla;cervidae;odocoileus;virginianus'
    # species = 'didelphis marsupialis'
    # country = 'Guatemala'
    
    # species = 'common opossum'
    
    species = species.lower()
    
    # Turn "species" into a taxonomy string

    # If this is already a taxonomy string...    
    if len(species.split(';')) == 5:
        pass
    # If this is a binomial name...
    elif len(species.split(' ')) == 2 and (species in binomial_name_to_taxonomy_info):
        taxonomy_info = binomial_name_to_taxonomy_info[species]
        taxonomy_string = taxonomy_info_to_taxonomy_string(taxonomy_info)        
    # If this is a common name...
    elif species in common_name_to_taxonomy_info:
        taxonomy_info = common_name_to_taxonomy_info[species]
        taxonomy_string = taxonomy_info_to_taxonomy_string(taxonomy_info)        
    else:
        raise ValueError('Could not find taxonomic information for {}'.format(species))
        
    
    # Normalize [state]
    
    if state is not None:
        state = state.upper()
        assert len(state) == 2
        
    # Turn "country" into a country code
    
    if len(country) == 3:
        assert country.upper() in country_code_to_country        
        country = country.upper()
    else:
        assert country.lower() in country_to_country_code
        country = country_to_country_code[country.lower()]
        
    country_code = country.upper()
        
    # Species with no rules are allowed everywhere
    if taxonomy_string not in taxonomy_string_to_geofencing_rules:
        status = 'allow_by_default'
        if return_status:
            return status
        else:
            return True
    
    geofencing_rules_this_species = taxonomy_string_to_geofencing_rules[taxonomy_string]
    allowed_countries = []
    blocked_countries = []
    
    assert len(geofencing_rules_this_species.keys()) == 1
    rule_type = list(geofencing_rules_this_species.keys())[0]
    assert rule_type in ('allow','block')
    
    if rule_type == 'allow':    
        allowed_countries = list(geofencing_rules_this_species['allow'])
    else:
        assert rule_type == 'block'
        blocked_countries = list(geofencing_rules_this_species['block'])
    
    status = None
    if country_code in blocked_countries:
        status = 'blocked'
    elif country_code in allowed_countries:
        status = 'allowed'
    else:
        # The convention is that if allow rules exist, any country not on that list
        # is blocked.
        assert len(allowed_countries) > 0
        return 'not_on_country_allow_list'
    
    # Now let's see whether we have to deal with any regional rules
    if state is None:
        
        # If state rules are provided, we need to have a state
        if country_code == 'USA':
            state_list = geofencing_rules_this_species[rule_type][country_code]
            if len(state_list) > 0:
                raise ValueError('Cannot determine status for a species with state-level rules with no state information')
                
    else:
        
        # Right now state-level rules only exist for the US
        assert country_code == 'USA'
        state_list = geofencing_rules_this_species[rule_type][country_code]
        
        if state in state_list:
            # If the state is on the list, do what the list says
            if rule_type == 'allow':
                status = 'allow_on_state_allow_list'
            else:
                status = 'block_on_state_block_list'
        else:
            # If the state is not on the list, do the opposite of what the list says
            if rule_type == 'allow':
                status = 'block_not_on_state_allow_list'
            else:
                status = 'allow_not_on_state_block_list'
        
    if return_status:
        return status
    else:
        if status.startswith('allow'):
            return True
        else:
            assert status.startswith('block')
            return False

# ...def species_allowed_in_country(...)


#%% Interactive driver(s)

if False:
    
    pass

    #%% Geofencing tests
    
    geofencing_file = r'g:\temp\geofence_mapping.json'
    country_code_file = r'G:/temp/country-codes.csv'    
    encoding = 'cp1252'; taxonomy_file = r'g:\temp\taxonomy_mapping-' + encoding + '.json'
    
    initialize_taxonomy_info(taxonomy_file, force_init=True, encoding=encoding)    
    initialize_geofencing(geofencing_file, country_code_file, force_init=True)
    
    species = 'didelphis marsupialis'
    print(binomial_name_to_taxonomy_info[species])
    country = 'Guatemala'
    assert species_allowed_in_country(species, country)
    
    species = 'virginia opossum'
    print(common_name_to_taxonomy_info[species])
    country = 'USA'
    assert species_allowed_in_country(species, country)
    
    
    #%% Test several species
    
    geofencing_file = r'g:\temp\geofence_mapping.json'
    country_code_file = r'G:/temp/country-codes.csv'    
    encoding = 'cp1252'; taxonomy_file = r'g:\temp\taxonomy_mapping-' + encoding + '.json'
    
    initialize_taxonomy_info(taxonomy_file, force_init=True, encoding=encoding)    
    initialize_geofencing(geofencing_file, country_code_file, force_init=True)
    
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
            
        
    #%% Test driver
    
    image_folder = r'g:\temp\md-test-images'
    bucket_prefix = '/mnt/g/temp/md-test-images/'
    jsonl_folder = r'g:\temp\predictions.json'
        
    md_formatted_results = generate_md_formatted_results_from_batch_jsonl(image_folder,
                                                                          jsonl_folder,
                                                                          bucket_prefix)
    
    md_formatted_output_file = os.path.join(image_folder,'md_formatted_results.json')
    with open(md_formatted_output_file,'w') as f:
        json.dump(md_formatted_results,f,indent=1)
        
    image_files_in_results = set()
    for im in md_formatted_results['images']:
        assert im['file'] not in image_files_in_results
        image_files_in_results.add(im['file'])
        
        
    #%% Preview
    
    from megadetector.postprocessing.postprocess_batch_results import \
        PostProcessingOptions, process_batch_results
    from megadetector.utils import path_utils
    
    render_animals_only = False
    
    options = PostProcessingOptions()
    options.image_base_dir = image_folder
    options.include_almost_detections = True
    options.num_images_to_sample = None
    options.confidence_threshold = 0.2
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    options.sample_seed = 0
    options.max_figures_per_html_file = 5000
    
    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = 10
    options.parallelize_rendering_with_threads = True
    options.sort_classification_results_by_count = True
    
    if render_animals_only:
        # Omit some pages from the output, useful when animals are rare
        options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                          'detections_person_vehicle','non_detections']    
    
    output_base = r'g:\temp\preview'
    if render_animals_only:
        output_base = output_base + '_render_animals_only'
    os.makedirs(output_base, exist_ok=True)
    
    print('Processing post-RDE to {}'.format(output_base))
    
    options.md_results_file = md_formatted_output_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    
    path_utils.open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
    # import clipboard; clipboard.copy(html_output_file)
