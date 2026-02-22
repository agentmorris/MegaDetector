"""

wi_platform_utils.py

Utility functions for working with the Wildlife Insights platform, specifically:

* Retrieving images based on .csv downloads
* Pushing results to the ProcessCVResponse() API (requires an API key)

"""

#%% Imports

import os
import requests

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from multiprocessing.pool import Pool, ThreadPool
from functools import partial

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.path_utils import make_executable
from megadetector.utils.path_utils import path_join

from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import compare_values_nan_equal

from megadetector.utils.string_utils import is_int

from megadetector.utils.wi_taxonomy_utils import is_valid_prediction_string
from megadetector.utils.wi_taxonomy_utils import no_cv_result_prediction_string
from megadetector.utils.wi_taxonomy_utils import blank_prediction_string

from megadetector.detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP

# Only used when pushing results directly to the platform via the API; any detections we want
# to show in the UI should have at least this confidence value.
min_md_output_confidence = 0.25

md_category_id_to_name = DEFAULT_DETECTOR_LABEL_MAP
md_category_name_to_id = invert_dictionary(md_category_id_to_name)

# Fields expected to be present in a valid WI result
wi_result_fields = ['wi_taxon_id','class','order','family','genus','species','common_name']


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

    sequence_list_file = path_join(download_folder,sequence_list_files[0])

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
            representing a Wildlife Insights download bundle.  If this is a single .csv
            file, reads just that file.


    Returns:
        dict: Maps image GUIDs to dicts with at least the following fields:
            * project_id (int)
            * deployment_id (str)
            * image_id (str, should match the key)
            * filename (str, the filename without path at the time of upload)
            * location (str, starting with gs://)

        May also contain classification fields: wi_taxon_id (str), species, etc.
    """

    print('Reading images from {}'.format(download_folder))

    ##%% Find lists of images

    # If the caller supplied a single file
    if os.path.isfile(download_folder):

        image_list_files = [download_folder]

    else:
        assert os.path.isdir(download_folder), \
            'Could not find folder {}'.format(download_folder)

        image_list_files = os.listdir(download_folder)
        image_list_files = \
            [fn for fn in image_list_files if fn.startswith('images_') and fn.endswith('.csv')]
        image_list_files = \
            [path_join(download_folder,fn) for fn in image_list_files]
        image_list_files = sorted(image_list_files)
        print('Found {} image list files'.format(len(image_list_files)))


    ##%% Read lists of images by deployment

    image_id_to_image_records = defaultdict(list)

    # i_file = 0; image_list_file = image_list_files[i_file]
    for i_file,image_list_file in enumerate(image_list_files):

        print('Reading images from list file {} of {} ({})'.format(
            i_file,
            len(image_list_files),
            os.path.basename(image_list_file)))

        df = pd.read_csv(image_list_file,low_memory=False)

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

# ...def read_images_from_download_bundle(...)


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


def write_prefix_download_command(image_records,
                                  download_dir_base,
                                  force_download=False,
                                  download_command_file=None):
    """
    Write a .sh script to download all images (using gcloud) from the longest common URL
    prefix in the images represented in [image_records].

    Args:
        image_records (list of dict): list of dicts with at least the field 'location'.
            Can also be a dict whose values are lists of record dicts.
        download_dir_base (str): local destination folder
        force_download (bool, optional): overwrite existing files
        download_command_file (str, optional): path of the .sh script we should write, defaults
            to "download_wi_images_with_prefix.sh" in the destination folder.
    """

    ##%% Input validation

    # If a dict is provided, assume it maps image GUIDs to lists of records, flatten to a list
    if isinstance(image_records,dict):
        all_image_records = []
        for k in image_records:
            records_this_image = image_records[k]
            all_image_records.extend(records_this_image)
        image_records = all_image_records

    assert isinstance(image_records,list), \
        'Illegal image record list format {}'.format(type(image_records))
    assert isinstance(image_records[0],dict), \
        'Illegal image record format {}'.format(type(image_records[0]))

    urls = [r['location'] for r in image_records]

    # "urls" is a list of URLs starting with gs://.  Find the highest-level folder
    # that is common to all URLs in the list.  For example, if the list is:
    #
    # gs://a/b/c
    # gs://a/b/d
    #
    # The result should be:
    #
    # gs://a/b
    common_prefix = os.path.commonprefix(urls)

    # Remove the gs:// prefix if it's still there
    if common_prefix.startswith('gs://'):
        common_prefix = common_prefix[len('gs://'):]

    # Ensure the common prefix ends with a '/' if it's not empty
    if (len(common_prefix) > 0) and (not common_prefix.endswith('/')):
        common_prefix = os.path.dirname(common_prefix) + '/'

    print('Longest common prefix: {}'.format(common_prefix))

    if download_command_file is None:
        download_command_file = \
            path_join(download_dir_base,'download_wi_images_with_prefix.sh')

    os.makedirs(download_dir_base,exist_ok=True)

    with open(download_command_file,'w',newline='\n') as f:
        # The --no-clobber flag prevents overwriting existing files
        # The -r flag is for recursive download
        # The gs:// prefix is added back for the gcloud command
        no_clobber_string = ''
        if not force_download:
            no_clobber_string = '--no-clobber'

        cmd = 'gcloud storage cp -r {} "gs://{}" "{}"'.format(
            no_clobber_string,common_prefix,download_dir_base)
        print('Writing download command:\n{}'.format(cmd))
        f.write(cmd + '\n')

    print('Download script written to {}'.format(download_command_file))

# ...def write_prefix_download_command(...)


def url_to_relative_path(url,image_flattening='deployment'):
    """
    Convert a WI gs:// URL to a relative path.

    Args:
        url (str): the URL to convert to a relative path
        image_flattening (str, optional): if 'none', relative paths will be
            returned as the entire URL for each image, other than gs://.  Can be
            'guid' (just return [GUID].JPG) or 'deployment' (return
            [deployment]/[GUID].JPG).

    Returns:
        str: converted path
    """

    assert url.startswith('gs://'), 'Illegal URL {}'.format(url)

    relative_path = None

    if image_flattening == 'none':
        relative_path = url.replace('gs://','')
    elif image_flattening == 'guid':
        relative_path = url.split('/')[-1]
    else:
        assert image_flattening == 'deployment'
        tokens = url.split('/')
        found_deployment_id = False
        for i_token,token in enumerate(tokens):
            if token == 'deployment':
                assert i_token < (len(tokens)-1)
                deployment_id_string = tokens[i_token + 1]
                deployment_id_string = deployment_id_string.replace('_thumb','')
                assert is_int(deployment_id_string), \
                    'Illegal deployment ID {}'.format(deployment_id_string)
                image_id = url.split('/')[-1]
                relative_path = deployment_id_string + '/' + image_id
                found_deployment_id = True
                break

        # ...for each token

        assert found_deployment_id, \
            'Could not find deployment ID for url {}'.format(url)

    return relative_path

# ...def url_to_relative_path(...)


def write_download_commands(image_records,
                            download_dir_base,
                            force_download=False,
                            n_download_workers=25,
                            download_command_file_base=None,
                            image_flattening='deployment'):
    """
    Given a list of dicts with at least the field 'location' (a gs:// URL), prepare a set of "gcloud
    storage" commands to download images, and write those to a series of .sh scripts, along with one
    .sh script that runs all the others and blocks.

    gcloud commands will use relative paths.

    Args:
        image_records (list of dict): list of dicts with at least the field 'location'.
            Can also be a dict whose values are lists of record dicts.
        download_dir_base (str): local destination folder
        force_download (bool, optional): include gs commands even if the target file exists
        n_download_workers (int, optional): number of scripts to write (that's our hacky way
            of controlling parallelization)
        download_command_file_base (str, optional): path of the .sh script we should write, defaults
            to "download_wi_images.sh" in the destination folder.  Individual worker scripts will
            have a number added, e.g. download_wi_images_00.sh.
        image_flattening (str, optional): if 'none', relative paths will be preserved
            representing the entire URL for each image.  Can be 'guid' (just download to
            [GUID].JPG) or 'deployment' (download to [deployment]/[GUID].JPG).
    """

    ##%% Input validation

    # If a dict is provided, assume it maps image GUIDs to lists of records, flatten to a list
    if isinstance(image_records,dict):
        all_image_records = []
        for k in image_records:
            records_this_image = image_records[k]
            all_image_records.extend(records_this_image)
        image_records = all_image_records

    assert isinstance(image_records,list), \
        'Illegal image record list format {}'.format(type(image_records))
    assert isinstance(image_records[0],dict), \
        'Illegal image record format {}'.format(type(image_records[0]))


    ##%% Map URLs to relative paths

    # URLs look like:
    #
    # gs://145625555_2004881_2323_name__main/deployment/2241000/prod/directUpload/5fda0ddd-511e-46ca-95c1-302b3c71f8ea.JPG
    if image_flattening is None:
        image_flattening = 'none'
    image_flattening = image_flattening.lower().strip()

    assert image_flattening in ('none','guid','deployment'), \
        'Illegal image flattening strategy {}'.format(image_flattening)

    url_to_relative_path = {}

    for image_record in image_records:

        url = image_record['location']
        relative_path = url_to_relative_path(url=url,
                                              image_flattening=image_flattening)
        assert relative_path is not None

        if url in url_to_relative_path:
            assert url_to_relative_path[url] == relative_path, \
                'URL path mapping error'
        else:
            url_to_relative_path[url] = relative_path

    # ...for each image record


    ##%% Make list of gcloud storage commands

    if download_command_file_base is None:
        download_command_file_base = path_join(download_dir_base,'download_wi_images.sh')

    commands = []
    skipped_urls = []
    downloaded_urls = set()

    # image_record = image_records[0]
    for image_record in tqdm(image_records):

        url = image_record['location']
        if url in downloaded_urls:
            continue

        assert url.startswith('gs://'), 'Illegal URL {}'.format(url)

        relative_path = url_to_relative_path[url]
        abs_path = path_join(download_dir_base,relative_path)

        # Optionally skip files that already exist
        if (not force_download) and (os.path.isfile(abs_path)):
            skipped_urls.append(url)
            continue

        # command = 'gsutil cp "{}" "./{}"'.format(url,relative_path)
        command = 'gcloud storage cp --no-clobber "{}" "./{}"'.format(url,relative_path)
        commands.append(command)

    print('Generated {} commands for {} image records'.format(
        len(commands),len(image_records)))

    print('Skipped {} URLs'.format(len(skipped_urls)))


    ##%% Write those commands out to n .sh files

    commands_by_script = split_list_into_n_chunks(commands,n_download_workers)

    local_download_commands = []

    output_dir = os.path.dirname(download_command_file_base)
    os.makedirs(output_dir,exist_ok=True)

    # Write out the download script for each chunk
    # i_script = 0
    for i_script in range(0,n_download_workers):
        if len(commands_by_script[i_script]) == 0:
            continue
        download_command_file = insert_before_extension(download_command_file_base,str(i_script).zfill(2))
        local_download_commands.append(os.path.basename(download_command_file))
        with open(download_command_file,'w',newline='\n') as f:
            for command in commands_by_script[i_script]:
                f.write(command + '\n')
        make_executable(download_command_file,catch_exceptions=True)


    # Write out the main download script
    with open(download_command_file_base,'w',newline='\n') as f:
        for local_download_command in local_download_commands:
            f.write('./' + local_download_command + ' &\n')
        f.write('wait\n')
        f.write('echo done\n')
    make_executable(download_command_file_base,catch_exceptions=True)

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

        try:
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
        finally:
            pool.close()
            pool.join()
            print('Pool closed and joined for WI result uploads')

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


#%% Functions for working with WI results (from the API or from download bundles)

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
            if not compare_values_nan_equal(record_0[k],record_1[k]):
                if verbose:
                    print('Image ID: {} ({})\nRecord 0/{}: {}\nRecord 1/{}: {}'.format(
                        record_0['image_id'],record_1['image_id'],
                        k,record_0[k],k,record_1[k]))
                return False

    return True


#%% Validate constants

# This is executed at the time this module gets imported.

blank_payload = generate_blank_prediction_payload('70ede9c6-d056-4dd1-9a0b-3098d8113e0e','1234')
validate_payload(sample_update_payload)
validate_payload(blank_payload)

