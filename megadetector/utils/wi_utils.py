"""

wi_utils.py

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

from copy import deepcopy
from collections import defaultdict
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from tqdm import tqdm

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.utils.path_utils import find_images
from megadetector.postprocessing.validate_batch_results import \
    validate_batch_results, ValidateBatchResultsOptions

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
    Determine whether [s] is a valid WI prediction string.  Prediction strings look like:
    
    '90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent'
    
    Args:
        s (str): the string to be tested for validity
        
    Returns:
        bool: True if this looks more or less like a WI prediction string
    """
    
    return isinstance(s,str) and (len(s.split(';')) == 7) and (s == s.lower())


def wi_result_to_prediction_string(r):
    """
    Convert the dict [r] - typically loaded from a row in a downloaded .csv file - to
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
    Reads all images.csv files from [download_folder], returns a dict mapping image IDs
    to a list of dicts that describe each image.  It's a list of dicts rather than a single dict
    because images may appear more than once.
    
    Args:
        download_folder (str): a folder containing one or more images.csv files, typically
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
    Read the authorization token from a text file and prepare http headers.
    
    Args:        
        auth_token_file (str): a single-line text file containing a write-enabled
        API token.
        
    Returns:
        dict: http headers, with fields 'Authorization' and 'Content-Type'
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
    Push results for one or more images represented in [payload] to the 
    process_cv_response API, to write to the WI DB.
    
    Args:
        payload (dict): payload to upload to the API
        headers (dict): authorization headers, see prepare_data_update_auth_headers
        url (str, optional): API URL
        verbose (bool, optional): enable additional debug output            
        
    Return:
        int: response status code
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
    Push results for the list of payloads in [payloads] to the process_cv_response API, 
    parallelized over multiple workers.
    
    Args:
        payloads (list of dict): payloads to upload to the API
        headers (dict): authorization headers, see prepare_data_update_auth_headers
        url (str, optional): API URL
        verbose (bool, optional): enable additional debug output
        pool_type (str, optional): 'thread' or 'process'
        n_workers (int, optional): number of parallel workers
        
    Returns:
        list of int: list of http response codes, one per payload
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
    
    Args:
        wi_result (dict): dict representing a WI prediction result, with at least the 
            fields in the constant wi_result_fields
        detections (list): list of WI-formatted detection dicts (with fields ['conf'] and ['category'])
        prediction_score (float, optional): confidence value to use for the combined prediction
        model_version (str, optional): model version string to include in the payload
        prediction_source (str, optional): prediction source string to include in the payload
        
    Returns:
        dict: dictionary suitable for uploading via push_results_for_images
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
    no detections.  Suitable for upload via push_results_for_images.
    
    Args:
        data_file_id (str): unique identifier for this image used in the WI DB
        project_id (int): WI project ID
        blank_confidence (float, optional): confidence value to associate with this
            prediction
        model_version (str, optional): model version string to include in the payload
        prediction_source (str, optional): prediction source string to include in the payload
        
    Returns:
        dict: dictionary suitable for uploading via push_results_for_images
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
    no detections.  Suitable for uploading via push_results_for_images.
    
    Args:
        data_file_id (str): unique identifier for this image used in the WI DB
        project_id (int): WI project ID
        no_cv_confidence (float, optional): confidence value to associate with this
            prediction        
        model_version (str, optional): model version string to include in the payload
        prediction_source (str, optional): prediction source string to include in the payload
    
    Returns:
        dict: dictionary suitable for uploading via push_results_for_images
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
                                           detections=None,
                                           model_version='3.1.2',
                                           prediction_source='manual_update'):
    """
    Generate a payload that will set a single image to a particular prediction, optionally
    including detections.  Suitable for uploading via push_results_for_images.
    
    Args:
        data_file_id (str): unique identifier for this image used in the WI DB
        project_id (int): WI project ID
        prediction_string (str): WI-formatted prediction string to include in the payload
        prediction_confidence (float, optional): confidence value to associate with this
            prediction        
        detections (list, optional): list of MD-formatted detection dicts, with fields
            ['category'] and 'conf'
        model_version (str, optional): model version string to include in the payload
        prediction_source (str, optional): prediction source string to include in the payload
        
    
    Returns:
        dict: dictionary suitable for uploading via push_results_for_images
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
    prediction['model_version'] = model_version
    prediction['prediction_source'] = prediction_source
    prediction['data_file_id'] = data_file_id
    prediction['project_id'] = project_id
    
    payload = {}
    payload['predictions'] = [prediction]
    
    return payload


def validate_payload(payload):
    """
    Verifies that the dict [payload] is compatible with the ProcessCVResponse() API.  Throws an
    error if [payload] is invalid.
    
    Args:
        payload (dict): payload in the format expected by push_results_for_images.
    
    Returns:
        bool: successful validation; this is just future-proofing, currently never returns False
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
    
    return True

# ...def validate_payload(...)


#%% Validate constants

# This is executed at the time this module gets imported.

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


def generate_md_results_from_predictions_json(predictions_json_file,md_results_file,base_folder=None):
    """
    Generate an MD-formatted .json file from a predictions.json file.  Typically,
    MD results files use relative paths, and predictions.json files use absolute paths, so 
    this function optionally removes the leading string [base_folder] from all file names.
    
    Currently just applies the top classification category to every detection.  If the top classification 
    is "blank", writes an empty detection list.
    
    wi_to_md.py is a command-line driver for this function.
    
    Args:
        predictions_json_file (str): path to a predictions.json file
        md_results_file (str): path to which we should write an MD-formatted .json file
        base_folder (str, optional): leading string to remove from each path in the predictions.json file
    """
        
    # Read predictions file    
    with open(predictions_json_file,'r') as f:
        predictions = json.load(f)
    predictions = predictions['predictions']
    assert isinstance(predictions,list)
    
    from megadetector.utils.ct_utils import is_list_sorted
    
    detection_category_id_to_name = {}
    classification_category_name_to_id = {}
    
    # Keep track of detections that don't have an assigned detection category; these 
    # are fake detections we create for non-blank images with non-empty detection lists.
    # We need to go back later and give them a legitimate detection category ID.
    all_unknown_detections = []
    
    # Create the output images list
    images_out = []
    
    # im_in = predictions[0]
    for im_in in predictions:
        
        # blank_prediction_string
        im_out = {}
        
        fn = im_in['filepath']
        if base_folder is not None:
            if fn.startswith(base_folder):
                fn = fn.replace(base_folder,'',1)
        
        im_out['file'] = fn
        
        if 'failures' in im_in:
            
            im_out['failure'] = str(im_in['failures'])
            im_out['detections'] = None
            
        else:
            
            im_out['detections'] = []
            
            if 'detections' in im_in:
                
                if len(im_in['detections']) == 0:
                    im_out['detections'] = []
                else:
                    # det_in = im_in['detections'][0]
                    for det_in in im_in['detections']:
                        det_out = {}
                        if det_in['category'] in detection_category_id_to_name:
                            assert detection_category_id_to_name[det_in['category']] == det_in['label']
                        else:
                            detection_category_id_to_name[det_in['category']] = det_in['label']
                        det_out = {}
                        for s in ['category','conf','bbox']:
                            det_out[s] = det_in[s]
                        im_out['detections'].append(det_out)
                        
            # ...if detections are present
            
            class_to_assign = None
            class_confidence = None
            
            if 'classifications' in im_in:
                
                classifications = im_in['classifications']
                assert len(classifications['scores']) == len(classifications['classes'])
                assert is_list_sorted(classifications['scores'],reverse=True)
                class_to_assign = classifications['classes'][0]
                class_confidence = classifications['scores'][0]
                
            if 'prediction' in im_in:
                
                class_to_assign = im_in['prediction']
                class_confidence = im_in['prediction_score']
            
            if class_to_assign is not None:
                
                if class_to_assign == blank_prediction_string:
                
                    # This is a scenario that's not captured well by the MD format: a blank prediction
                    # with detections present.  But, for now, don't do anything special here, just making
                    # a note of this.
                    if len(im_out['detections']) > 0:
                        pass                    
                
                else:
                    
                    assert not class_to_assign.endswith('blank')
                    
                    # This is a scenario that's not captured well by the MD format: no detections present,
                    # but a non-blank prediction.  For now, create a fake detection to handle this prediction.
                    if len(im_out['detections']) == 0:
                    
                        print('Warning: creating fake detection for non-blank whole-image classification')
                        det_out = {}
                        all_unknown_detections.append(det_out)
                        
                        # We will change this to a string-int later
                        det_out['category'] = 'unknown'
                        det_out['conf'] = class_confidence
                        det_out['bbox'] = [0,0,1,1]
                        im_out['detections'].append(det_out)
                
                # ...if this is/isn't a blank classification
                
                # Attach that classification to each detection
                
                # Create a new category ID if necessary
                if class_to_assign in classification_category_name_to_id:
                    classification_category_id = classification_category_name_to_id[class_to_assign]
                else:
                    classification_category_id = str(len(classification_category_name_to_id))
                    classification_category_name_to_id[class_to_assign] = classification_category_id
                
                for det in im_out['detections']:
                    det['classifications'] = []
                    det['classifications'].append([classification_category_id,class_confidence])
                
            # ...if we have some type of classification for this image
        
        # ...if this is/isn't a failure
        
        images_out.append(im_out)
        
    # ...for each image
    
    # Fix the 'unknown' category
    
    if len(all_unknown_detections) > 0:
        
        max_detection_category_id = max([int(x) for x in detection_category_id_to_name.keys()])
        unknown_category_id = str(max_detection_category_id + 1)
        detection_category_id_to_name[unknown_category_id] = 'unknown'
        
        for det in all_unknown_detections:
            assert det['category'] == 'unknown'
            det['category'] = unknown_category_id
                    
    
    # Sort by filename
    
    images_out = sort_list_of_dicts_by_key(images_out,'file')
    
    # Prepare friendly classification names
    
    classification_category_descriptions = invert_dictionary(classification_category_name_to_id)
    classification_categories_out = {}
    for category_id in classification_category_descriptions.keys():
        category_name = classification_category_descriptions[category_id].split(';')[-1]
        classification_categories_out[category_id] = category_name
    
    # Prepare the output dict
    
    detection_categories_out = detection_category_id_to_name
    info = {}
    info['format_version'] = 1.4
    info['detector'] = 'converted_from_predictions_json'
    
    output_dict = {}
    output_dict['info'] = info
    output_dict['detection_categories'] = detection_categories_out
    output_dict['classification_categories'] = classification_categories_out
    output_dict['classification_category_descriptions'] = classification_category_descriptions
    output_dict['images'] = images_out
    
    with open(md_results_file,'w') as f:
        json.dump(output_dict,f,indent=1)
    
    validation_options = ValidateBatchResultsOptions()
    validation_options.raise_errors = True    
    _ = validate_batch_results(md_results_file, options=validation_options)
    
# ...def generate_md_results_from_predictions_json(...)
        

def generate_predictions_json_from_md_results(md_results_file,predictions_json_file,base_folder=None):
    """
    Generate a predictions.json file from the MD-formatted .json file [md_results_file].  Typically,
    MD results files use relative paths, and predictions.json files use absolute paths, so 
    this function optionally prepends [base_folder].  Does not handle classification results in
    MD format, since this is intended to prepare data for passing through the WI classifier.
    
    md_to_wi.py is a command-line driver for this function.
    
    Args:
        md_results_file (str): path to an MD-formatted .json file
        predictions_json_file (str): path to which we should write a predictions.json file
        base_folder (str, optional): folder name to prepend to each path in md_results_file,
            to convert relative paths to absolute paths.    
    """
        
    # Validate the input file
    validation_options = ValidateBatchResultsOptions()
    validation_options.raise_errors = True    
    validation_options.return_data = True
    md_results = validate_batch_results(md_results_file, options=validation_options)
    category_id_to_name = md_results['detection_categories']
    
    output_dict = {}
    output_dict['predictions'] = []
        
    # im = md_results['images'][0]
    for im in md_results['images']:
        
        prediction = {}
        fn = im['file']
        if base_folder is not None:
            fn = os.path.join(base_folder,fn)
        fn = fn.replace('\\','/')
        prediction['filepath'] = fn
        if 'failure' in im and im['failure'] is not None:
            prediction['failures'] = ['DETECTOR']
        else:
            assert 'detections' in im and im['detections'] is not None
            detections = []
            for det in im['detections']:
                output_det = deepcopy(det)
                output_det['label'] = category_id_to_name[det['category']]
                detections.append(output_det)
            
            # detections *must* be sorted in descending order by confidence
            detections = sort_list_of_dicts_by_key(detections,'conf', reverse=True)
            prediction['detections'] = detections
            
        assert len(prediction.keys()) >= 2
        output_dict['predictions'].append(prediction)
            
    # ...for each image
        
    os.makedirs(os.path.dirname(predictions_json_file),exist_ok=True)
    with open(predictions_json_file,'w') as f:
        json.dump(output_dict,f,indent=1)
        
# ...def generate_predictions_json_from_md_results(...)


def generate_instances_json_from_folder(folder,
                                        country=None,
                                        lat=None,
                                        lon=None,
                                        output_file=None,
                                        filename_replacements=None):
    """
    Generate an instances.json record that contains all images in [folder], optionally
    including location information, in a format suitable for run_model.py.  Optionally writes
    the results to [output_file].
    
    Args:
        folder (str): the folder to recursively search for images
        country (str, optional): a three-letter country code
        lat (float, optional): latitude to associate with all images
        lon (float, optional): longitude to associate with all images
        output_file (str, optional): .json file to which we should write instance records
        filename_replacements (dict, optional): str --> str dict indicating filename substrings
            that should be replaced with other strings.  Replacement occurs *after* converting
            backslashes to forward slashes.
        
    Returns:
        dict: dict with at least the field "instances"
    """
    
    assert os.path.isdir(folder)
        
    image_files_abs = find_images(folder,recursive=True,return_relative_paths=False)
    
    instances = []
    
    # image_fn_abs = image_files_abs[0]
    for image_fn_abs in image_files_abs:
        instance = {}
        instance['filepath'] = image_fn_abs.replace('\\','/')
        if filename_replacements is not None:
            for s in filename_replacements:
                instance['filepath'] = instance['filepath'].replace(s,filename_replacements[s])
        if country is not None:
            instance['country'] = country
        if lat is not None:
            assert lon is not None, 'Latitude provided without longitude'
            instance['latitude'] = lat
        if lon is not None:
            assert lat is not None, 'Longitude provided without latitude'
            instance['longitude'] = lon
        instances.append(instance)
    
    to_return = {'instances':instances}
    
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file),exist_ok=True)
        with open(output_file,'w') as f:
            json.dump(to_return,f,indent=1)
    
    return to_return
        
# ...def generate_instances_json_from_folder(...)


#%% Functions related to geofencing and taxonomy mapping

# This maps a taxonomy string (e.g. mammalia;cetartiodactyla;cervidae;odocoileus;virginianus) to 
# a dict with keys taxon_id, common_name, kingdom, phylum, class, order, family, genus, species
taxonomy_string_to_taxonomy_info = None
binomial_name_to_taxonomy_info = None
common_name_to_taxonomy_info = None

def taxonomy_info_to_taxonomy_string(taxonomy_info):
    """
    Convert a taxonomy record in dict format to a semicolon-delimited string
    
    Args:
        taxonomy_info (dict): dict in the format stored in, e.g., taxonomy_string_to_taxonomy_info
    
    Returns:
        str: string in the format used as keys in, e.g., taxonomy_string_to_taxonomy_info
    """
    return taxonomy_info['class'] + ';' + \
        taxonomy_info['order'] + ';'  + \
        taxonomy_info['family'] + ';'  + \
        taxonomy_info['genus'] + ';'  + \
        taxonomy_info['species']
        

def initialize_taxonomy_info(taxonomy_file,force_init=False,encoding='cp1252'):
    """
    Load WI taxonomy information from a .json file.  Stores information in the global
    dicts [taxonomy_string_to_taxonomy_info], [binomial_name_to_taxonomy_info], and
    [common_name_to_taxonomy_info].
    
    Args:
        taxonomy_file (str): .json file containing WI taxonomy information
        force_init (bool, optional): if the output dicts already exist, should we
            re-initialize anyway?
        encoding (str, optional): character encoding to use when opening the .json file
    """
    
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
        import codecs
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

# Dict mapping semicolon-delimited taxonomy strings to geofencing rules
taxonomy_string_to_geofencing_rules = None

# Maps lower-case country names to upper-case country codes
country_to_country_code = None

# Maps upper-case country codes to lower-case country names
country_code_to_country = None

def initialize_geofencing(geofencing_file,country_code_file,force_init=False):
    """
    Load geofencing information from a .json file, and country code mappings from
    a .csv file.  Stores results in the global tables [taxonomy_string_to_geofencing_rules],
    [country_to_country_code], and [country_code_to_country].
    
    Args:
        geofencing_file (str): .json file with geofencing rules
        country_code_file (str): .csv file with country code mappings
        force_init (bool, optional): if the output dicts already exist, should we
            re-initialize anyway?
    """
    global taxonomy_string_to_geofencing_rules
    global country_to_country_code
    global country_code_to_country
    
    if (country_to_country_code is not None) and \
        (country_code_to_country is not None) and \
        (taxonomy_string_to_geofencing_rules is not None) and \
        (not force_init):
        return
    
    # Read country code information
    country_code_df = pd.read_csv(country_code_file)
    country_to_country_code = {}
    country_code_to_country = {}
    for i_row,row in country_code_df.iterrows():
        country_to_country_code[row['name'].lower()] = row['alpha-3'].upper()
        country_code_to_country[row['alpha-3'].upper()] = row['name'].lower()
                    
    # Read geofencing information
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
    Determines whether [species] is allowed in [country], according to 
    already-initialized geofencing rules.
    
    Args:
        species (str): can be a common name, a binomial name, or a species string
        country (str): country name or three-letter code
        state (str, optional): two-letter US state code
        return_status (bool, optional): by default, this function returns a bool;
            if you want to know *why* [species] is allowed/not allowed, settings
            return_status to True will return additional information.        
    
    Returns:
        bool or str: typically returns True if [species] is allowed in [country], else
        False.  Returns a more detailed string if return_status is set.
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

    #%% instances.json generation test
    
    from megadetector.utils.wi_utils import generate_instances_json_from_folder # noqa
    
    instances_file = r'g:\temp\water-hole\instances.json'
    
    _ = generate_instances_json_from_folder(folder=r'g:\temp\water-hole',
                                            country='NAM',
                                            lat=None,
                                            lon=None,
                                            output_file=instances_file,
                                            filename_replacements={'g:/temp':'/mnt/g/temp'})

    # from megadetector.utils.path_utils import open_file; open_file(instances_file)


    #%% MD --> prediction conversion test
    
    from megadetector.utils.wi_utils import generate_predictions_json_from_md_results # noqa
    md_results_file = r'G:\temp\md-test-images\mdv5a.relpaths.json'
    predictions_json_file = r'\\wsl$\Ubuntu\home\dmorris\tmp\speciesnet-tests\mdv5a.abspaths.predictions-format.json'
    generate_predictions_json_from_md_results(md_results_file,predictions_json_file,base_folder=
                                              '/home/dmorris/tmp/md-test-images/')
    
    from megadetector.utils.wi_utils import generate_predictions_json_from_md_results # noqa
    md_results_file = r"G:\temp\water-hole\md_results.json"
    predictions_json_file = r"G:\temp\water-hole\md_results-prediction_format.json"
    generate_predictions_json_from_md_results(md_results_file,predictions_json_file,base_folder=
                                              '/mnt/g/temp/water-hole')    
    
    
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
            
        
    #%% Test conversion from predictons.json to MD format
    
    import os # noqa
    from megadetector.utils.wi_utils import generate_md_results_from_predictions_json # noqa
    
    # detector_source = 'speciesnet'
    detector_source = 'md'
    
    if False:
        image_folder = r'g:\temp\md-test-images'
        base_folder = '/home/dmorris/tmp/md-test-images/'
        if detector_source == 'speciesnet':    
            predictions_json_file = r"\\wsl$\Ubuntu\home\dmorris\tmp\speciesnet-tests\ensemble-output.json"
            md_results_file = r"\\wsl$\Ubuntu\home\dmorris\tmp\speciesnet-tests\ensemble-output-md-format.json"
        else:
            assert detector_source == 'md'
            predictions_json_file = r"\\wsl$\Ubuntu\home\dmorris\tmp\speciesnet-tests\ensemble-output-from-md-results.json"
            md_results_file = r"\\wsl$\Ubuntu\home\dmorris\tmp\speciesnet-tests\ensemble-output-md-format-from-md-results.json"        
    else:
        image_folder = r'g:\temp\water-hole'
        base_folder = '/mnt/g/temp/water-hole/'
        if detector_source == 'speciesnet':    
            predictions_json_file = r'g:\temp\water-hole\ensemble-output.json'
            md_results_file = r'g:\temp\water-hole\ensemble-output.md_format.json'
        else:
            assert detector_source == 'md'
            predictions_json_file = r'g:\temp\water-hole\ensemble-output-md.json'
            md_results_file = r'g:\temp\water-hole\ensemble-output-md.md_format.json'
    
    generate_md_results_from_predictions_json(predictions_json_file=predictions_json_file,
                                              md_results_file=md_results_file,
                                              base_folder=base_folder)
    
    # from megadetector.utils.path_utils import open_file; open_file(md_results_file)
    
    assert os.path.isdir(image_folder)
    
   
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
    
    output_base = r'g:\temp\preview' + '_' + detector_source
    if render_animals_only:
        output_base = output_base + '_render_animals_only'
    os.makedirs(output_base, exist_ok=True)
    
    print('Writing preview to {}'.format(output_base))
    
    options.md_results_file = md_results_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    
    path_utils.open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
    # import clipboard; clipboard.copy(html_output_file)
