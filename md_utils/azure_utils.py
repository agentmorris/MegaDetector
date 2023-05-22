########
#
# azure_utils.py
#
# Miscellaneous Azure Blob Storage utilities
#
# Requires azure-storage-blob>=12.4.0
#
########

import json
from md_utils import path_utils
from typing import Any, Iterable, List, Optional, Tuple, Union

from azure.storage.blob import BlobPrefix, ContainerClient

from md_utils import sas_blob_utils


def walk_container(container_client: ContainerClient,
                   max_depth: int = -1,
                   prefix: str = '',
                   store_folders: bool = True,
                   store_blobs: bool = True,
                   debug_max_items: int = -1) -> Tuple[List[str], List[str]]:
    """
    Recursively walk folders a Azure Blob Storage container.

    Based on:
    https://github.com/Azure/azure-sdk-for-python/blob/master/sdk/storage/azure-storage-blob/samples/blob_samples_walk_blob_hierarchy.py
    """
    
    depth = 1

    def walk_blob_hierarchy(prefix: str,
                            folders: Optional[List[str]] = None,
                            blobs: Optional[List[str]] = None
                            ) -> Tuple[List[str], List[str]]:
        if folders is None:
            folders = []
        if blobs is None:
            blobs = []

        nonlocal depth

        if 0 < max_depth < depth:
            return folders, blobs

        for item in container_client.walk_blobs(name_starts_with=prefix):
            short_name = item.name[len(prefix):]
            if isinstance(item, BlobPrefix):
                # print('F: ' + prefix + short_name)
                if store_folders:
                    folders.append(prefix + short_name)
                depth += 1
                walk_blob_hierarchy(item.name, folders=folders, blobs=blobs)
                if (debug_max_items > 0
                        and len(folders) + len(blobs) > debug_max_items):
                    return folders, blobs
                depth -= 1
            else:
                if store_blobs:
                    blobs.append(prefix + short_name)

        return folders, blobs

    folders, blobs = walk_blob_hierarchy(prefix=prefix)

    assert all(s.endswith('/') for s in folders)
    folders = [s.strip('/') for s in folders]

    return folders, blobs


def list_top_level_blob_folders(container_client: ContainerClient) -> List[str]:
    """
    List all top-level folders in a container.
    """
    
    top_level_folders, _ = walk_container(
        container_client, max_depth=1, store_blobs=False)
    return top_level_folders


def concatenate_json_lists(input_files: Iterable[str],
                           output_file: Optional[str] = None
                           ) -> List[Any]:
    """
    Given a list of JSON files that contain lists (typically string
    filenames), concatenates the lists into a single list and optionally
    writes out this list to a new output JSON file.
    """
    
    output_list = []
    for fn in input_files:
        with open(fn, 'r') as f:
            file_list = json.load(f)
        output_list.extend(file_list)
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(output_list, f, indent=1)
    return output_list


def upload_file_to_blob(account_name: str,
                        container_name: str,
                        local_path: str,
                        blob_name: str,
                        sas_token: str,
                        overwrite: bool=False) -> str:
    """
    Uploads a local file to Azure Blob Storage and returns the uploaded
    blob URI with SAS token.
    """
    
    container_uri = sas_blob_utils.build_azure_storage_uri(
        account=account_name, container=container_name, sas_token=sas_token)
    with open(local_path, 'rb') as data:
        return sas_blob_utils.upload_blob(
            container_uri=container_uri, blob_name=blob_name, data=data, 
            overwrite=overwrite)


def enumerate_blobs_to_file(
        output_file: str,
        account_name: str,
        container_name: str,
        sas_token: Optional[str] = None,
        blob_prefix: Optional[str] = None,
        blob_suffix: Optional[Union[str, Tuple[str]]] = None,
        rsearch: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: Optional[bool] = True
        ) -> List[str]:
    """
    Enumerates blobs in a container, and writes the blob names to an output
    file.

    Args:
        output_file: str, path to save list of files in container
            If ends in '.json', writes a JSON string. Otherwise, writes a
            newline-delimited list. Can be None, in which case this is just a 
            convenient wrapper for blob enumeration.
        account_name: str, Azure Storage account name
        container_name: str, Azure Blob Storage container name
        sas_token: optional str, container SAS token, leading ? will be removed if present.
        blob_prefix: optional str, returned results will only contain blob names
            to with this prefix
        blob_suffix: optional str or tuple of str, returned results will only
            contain blob names with this/these suffix(es). The blob names will
            be lowercased first before comparing with the suffix(es).
        rsearch: optional str, returned results will only contain blob names
            that match this regex. Can also be a list of regexes, in which case
            blobs matching *any* of the regex's will be returned.            
        limit: int, maximum # of blob names to list
            if None, then returns all blob names

    Returns: list of str, sorted blob names, of length limit or shorter.
    """
    
    if sas_token is not None and len(sas_token) > 9 and sas_token[0] == '?':
        sas_token = sas_token[1:]
        
    container_uri = sas_blob_utils.build_azure_storage_uri(
        account=account_name, container=container_name, sas_token=sas_token)
    
    matched_blobs = sas_blob_utils.list_blobs_in_container(
        container_uri=container_uri, blob_prefix=blob_prefix,
        blob_suffix=blob_suffix, rsearch=rsearch, limit=limit, verbose=verbose)
    
    if output_file is not None:
        path_utils.write_list_to_file(output_file, matched_blobs)
        
    return matched_blobs
