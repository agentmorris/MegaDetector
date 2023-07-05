########
#
# sas_blob_utils.py
#
# This module contains helper functions for dealing with Shared Access Signatures
# (SAS) tokens for Azure Blob Storage.
#
# The default Azure Storage SAS URI format is:
#
# https://<account>.blob.core.windows.net/<container>/<blob>?<sas_token>
#
# This module assumes azure-storage-blob version 12.5.
#
# Documentation for Azure Blob Storage:
# docs.microsoft.com/en-us/azure/developer/python/sdk/storage/storage-blob-readme
#
# Documentation for SAS:
# docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview
#
########

#%% Imports

from datetime import datetime, timedelta
import io
import re
from typing import (Any, AnyStr, Dict, IO, Iterable, List, Optional, Set, Tuple, Union)
from urllib import parse
import uuid

from tqdm import tqdm

from azure.storage.blob import (
    BlobClient,
    BlobProperties,
    ContainerClient,
    ContainerSasPermissions,
    generate_container_sas)


#%% URI management

def build_azure_storage_uri(
        account: str,
        container: Optional[str] = None,
        blob: Optional[str] = None,
        sas_token: Optional[str] = None,
        account_url_template: str = 'https://{account}.blob.core.windows.net'
        ) -> str:
    """
    Args:
        account: str, name of Azure Storage account
        container: optional str, name of Azure Blob Storage container
        blob: optional str, name of blob, not URL-escaped
            if blob is given, must also specify container
        sas_token: optional str, Shared Access Signature (SAS).  Leading ? 
            is removed if present.
        account_url_template: str, Python 3 string formatting template,
            contains '{account}' placeholder, defaults to default Azure
            Storage URL format. Set this value if using Azurite Azure Storage
            emulator.

    Returns: str, Azure storage URI
    """
    
    uri = account_url_template.format(account=account)
    if container is not None:
        uri = f'{uri}/{container}'
    if blob is not None:
        assert container is not None
        blob = parse.quote(blob)
        uri = f'{uri}/{blob}'
    if sas_token is not None:
        if sas_token[0] == '?':
            sas_token = sas_token[1:]
        uri = f'{uri}?{sas_token}'
    return uri


def _get_resource_reference(prefix: str) -> str:
    return '{}{}'.format(prefix, str(uuid.uuid4()).replace('-', ''))


def get_client_from_uri(container_uri: str) -> ContainerClient:
    """
    Gets a ContainerClient for the given container URI.
    """
    
    return ContainerClient.from_container_url(container_uri)


def get_account_from_uri(sas_uri: str) -> str:
    """
    Assumes that sas_uri points to Azure Blob Storage account hosted at
    a default Azure URI. Does not work for locally-emulated Azure Storage
    or Azure Storage hosted at custom endpoints.
    """
    
    url_parts = parse.urlsplit(sas_uri)
    loc = url_parts.netloc  # "<account>.blob.core.windows.net"
    return loc.split('.')[0]


def is_container_uri(sas_uri: str) -> bool:
    """
    Returns True if the signed resource field in the URI "sr" is a container "c"
    or a directory "d"
    """
    
    data = get_all_query_parts(sas_uri)
    if 'sr' not in data:
        return False

    if 'c' in data['sr'] or 'd' in data['sr']:
        return True
    else:
        return False

def is_blob_uri(sas_uri: str) -> bool:
    """
    Returns True if the signed resource field in the URI "sr" is a blob "b".
    """
    
    data = get_all_query_parts(sas_uri)
    if 'sr' not in data:
        return False

    if 'b' in data['sr']:
        return True
    else:
        return False


def get_container_from_uri(sas_uri: str, unquote: bool = True) -> str:
    """
    Gets the container name from a Azure Blob Storage URI.

    Assumes that sas_uri points to Azure Blob Storage account hosted at
    a default Azure URI. Does not work for locally-emulated Azure Storage
    or Azure Storage hosted at custom endpoints.

    Args:
        sas_uri: str, Azure blob storage URI, may include SAS token
        unquote: bool, whether to replace any %xx escapes by their
            single-character equivalent, default True

    Returns: str, container name

    Raises: ValueError, if sas_uri does not include a container
    """
    
    url_parts = parse.urlsplit(sas_uri)
    raw_path = url_parts.path.lstrip('/')  # remove leading "/" from path
    container = raw_path.split('/')[0]
    if container == '':
        raise ValueError('Given sas_uri does not include a container.')
    if unquote:
        container = parse.unquote(container)
    return container


def get_blob_from_uri(sas_uri: str, unquote: bool = True) -> str:
    """
    Return the path to the blob from the root container if this sas_uri
    is for an individual blob; otherwise returns None.

    Args:
        sas_uri: str, Azure blob storage URI, may include SAS token
        unquote: bool, whether to replace any %xx escapes by their
            single-character equivalent, default True

    Returns: str, blob name (path to the blob from the root container)

    Raises: ValueError, if sas_uri does not include a blob name
    """
    
    # Get the entire path with all slashes after the container
    url_parts = parse.urlsplit(sas_uri)
    raw_path = url_parts.path.lstrip('/')  # remove leading "/" from path
    parts = raw_path.split('/', maxsplit=1)
    if len(parts) < 2 or parts[1] == '':
        raise ValueError('Given sas_uri does not include a blob name')

    blob = parts[1]  # first item is an empty string
    if unquote:
        blob = parse.unquote(blob)
    return blob


def get_sas_token_from_uri(sas_uri: str) -> Optional[str]:
    """
    Get the query part of the SAS token that contains permissions, access
    times and signature.

    Args:
        sas_uri: str, Azure blob storage SAS token

    Returns: str, query part of the SAS token (without leading '?'),
        or None if URI has no token.
    """
    
    url_parts = parse.urlsplit(sas_uri)
    sas_token = url_parts.query or None  # None if query is empty string
    return sas_token


def get_resource_type_from_uri(sas_uri: str) -> Optional[str]:
    """
    Get the resource type pointed to by this SAS token.

    Args:
        sas_uri: str, Azure blob storage URI with SAS token

    Returns: A string (either 'blob' or 'container') or None.
    """
    
    url_parts = parse.urlsplit(sas_uri)
    data = parse.parse_qs(url_parts.query)
    if 'sr' in data:
        types = data['sr']
        if 'b' in types:
            return 'blob'
        elif 'c' in types:
            return 'container'
    return None


def get_endpoint_suffix(sas_uri):
    """
    Gets the endpoint at which the blob storage account is served.
    Args:
        sas_uri: str, Azure blob storage URI with SAS token

    Returns: A string, usually 'core.windows.net' or 'core.chinacloudapi.cn', to
        use for the `endpoint` argument in various blob storage SDK functions.
    """
    
    url_parts = parse.urlsplit(sas_uri)
    suffix = url_parts.netloc.split('.blob.')[1].split('/')[0]
    return suffix


def get_permissions_from_uri(sas_uri: str) -> Set[str]:
    """
    Get the permissions given by this SAS token.

    Args:
        sas_uri: str, Azure blob storage URI with SAS token

    Returns: A set containing some of 'read', 'write', 'delete' and 'list'.
        Empty set returned if no permission specified in sas_uri.
    """
    
    data = get_all_query_parts(sas_uri)
    permissions_set = set()
    if 'sp' in data:
        permissions = data['sp'][0]
        if 'r' in permissions:
            permissions_set.add('read')
        if 'w' in permissions:
            permissions_set.add('write')
        if 'd' in permissions:
            permissions_set.add('delete')
        if 'l' in permissions:
            permissions_set.add('list')
    return permissions_set


def get_all_query_parts(sas_uri: str) -> Dict[str, Any]:
    """
    Gets the SAS token parameters.
    """
    
    url_parts = parse.urlsplit(sas_uri)
    return parse.parse_qs(url_parts.query)


#%% Blob 

def check_blob_exists(sas_uri: str, blob_name: Optional[str] = None) -> bool:
    """
    Checks whether a given URI points to an actual blob.

    Assumes that sas_uri points to Azure Blob Storage account hosted at
    a default Azure URI. Does not work for locally-emulated Azure Storage
    or Azure Storage hosted at custom endpoints. In these cases, create a
    BlobClient using the default constructor, instead of from_blob_url(),
    and use the BlobClient.exists() method directly.

    Args:
        sas_uri: str, URI to a container or a blob
            if blob_name is given, sas_uri is treated as a container URI
            otherwise, sas_uri is treated as a blob URI
        blob_name: optional str, name of blob, not URL-escaped
            must be given if sas_uri is a URI to a container

    Returns: bool, whether the sas_uri given points to an existing blob
    """
    
    if blob_name is not None:
        sas_uri = build_blob_uri(
            container_uri=sas_uri, blob_name=blob_name)

    with BlobClient.from_blob_url(sas_uri) as blob_client:
        return blob_client.exists()


def upload_blob(container_uri: str, blob_name: str,
                data: Union[Iterable[AnyStr], IO[AnyStr]],
                overwrite: bool = False) -> str:
    """
    Creates a new blob of the given name from an IO stream.

    Args:
        container_uri: str, URI to a container, may include SAS token
        blob_name: str, name of blob to upload
        data: str, bytes, or IO stream
            if str, assumes utf-8 encoding
        overwrite: bool, whether to overwrite existing blob (if any)

    Returns: str, URL to blob, includes SAS token if container_uri has SAS token
    """
    
    account_url, container, sas_token = split_container_uri(container_uri)
    with BlobClient(account_url=account_url, container_name=container,
                    blob_name=blob_name, credential=sas_token) as blob_client:
        blob_client.upload_blob(data, overwrite=overwrite)
        return blob_client.url


def download_blob_to_stream(sas_uri: str) -> Tuple[io.BytesIO, BlobProperties]:
    """
    Downloads a blob to an IO stream.

    Args:
        sas_uri: str, URI to a blob

    Returns:
        output_stream: io.BytesIO, remember to close it when finished using
        blob_properties: BlobProperties

    Raises: azure.core.exceptions.ResourceNotFoundError, if sas_uri points
        to a non-existent blob
    """
    
    with BlobClient.from_blob_url(sas_uri) as blob_client:
        output_stream = io.BytesIO()
        blob_client.download_blob().readinto(output_stream)
        output_stream.seek(0)
        blob_properties = blob_client.get_blob_properties()
    return output_stream, blob_properties


def build_blob_uri(container_uri: str, blob_name: str) -> str:
    """
    Args:
        container_uri: str, URI to blob storage container
            <account_url>/<container>?<sas_token>
        blob_name: str, name of blob, not URL-escaped

    Returns: str, blob URI <account_url>/<container>/<blob_name>?<sas_token>,
        <blob_name> is URL-escaped
    """
    
    account_url, container, sas_token = split_container_uri(container_uri)

    blob_name = parse.quote(blob_name)
    blob_uri = f'{account_url}/{container}/{blob_name}'
    if sas_token is not None:
        blob_uri += f'?{sas_token}'
    return blob_uri


#%% Container

def list_blobs_in_container(
        container_uri: str,
        blob_prefix: Optional[str] = None,
        blob_suffix: Optional[Union[str, Tuple[str]]] = None,
        rsearch: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: Optional[bool] = True
) -> List[str]:
    """
    Get a sorted list of blob names in this container.

    Args:
        container_uri: str, URI to a container, may include SAS token
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

    Returns:
        sorted list of blob names, of length limit or shorter.
    """

    if verbose:
        print('Listing blobs')
        
    if (get_sas_token_from_uri(container_uri) is not None
            and get_resource_type_from_uri(container_uri) != 'container'):
        raise ValueError('The SAS token provided is not for a container.')

    if blob_prefix is not None and not isinstance(blob_prefix, str):
        raise ValueError('blob_prefix must be a str.')

    if (blob_suffix is not None
            and not isinstance(blob_suffix, str)
            and not isinstance(blob_suffix, tuple)):
        raise ValueError('blob_suffix must be a str or a tuple of strings')

    list_blobs = []
    with get_client_from_uri(container_uri) as container_client:
        generator = container_client.list_blobs(
            name_starts_with=blob_prefix)

        if blob_suffix is None and rsearch is None:
            list_blobs = [blob.name for blob in tqdm(generator,disable=(not verbose))]
            i = len(list_blobs)
        else:
            i = 0
            for blob in tqdm(generator,disable=(not verbose)):
                i += 1
                suffix_ok = (blob_suffix is None
                             or blob.name.lower().endswith(blob_suffix))
                regex_ok = False
                if rsearch is None:
                    regex_ok = True
                else:
                    if not isinstance(rsearch, list):
                        rsearch = [rsearch]
                    # Check whether this blob name matches *any* of our regex's
                    for expr in rsearch:
                        if re.search(expr, blob.name) is not None:
                            regex_ok = True
                            break
                if suffix_ok and regex_ok:
                    list_blobs.append(blob.name)
                    if limit is not None and len(list_blobs) == limit:
                        break

    if verbose:
        print(f'Enumerated {len(list_blobs)} matching blobs out of {i} total')
        
    return sorted(list_blobs)  # sort for determinism


def generate_writable_container_sas(account_name: str,
                                    account_key: str,
                                    container_name: str,
                                    access_duration_hrs: float,
                                    account_url: Optional[str] = None
                                    ) -> str:
    """
    Creates a container and returns a SAS URI with read/write/list
    permissions.

    Args:
        account_name: str, name of blob storage account
        account_key: str, account SAS token or account shared access key
        container_name: str, name of container to create, must not match an
            existing container in the given storage account
        access_duration_hrs: float
        account_url: str, optional, defaults to default Azure Storage URL

    Returns: str, URL to newly created container

    Raises: azure.core.exceptions.ResourceExistsError, if container already
        exists
    """
    
    if account_url is None:
        account_url = build_azure_storage_uri(account=account_name)
    with ContainerClient(account_url=account_url,
                         container_name=container_name,
                         credential=account_key) as container_client:
        container_client.create_container()

    permissions = ContainerSasPermissions(read=True, write=True, list=True)
    container_sas_token = generate_container_sas(
        account_name=account_name,
        container_name=container_name,
        account_key=account_key,
        permission=permissions,
        expiry=datetime.utcnow() + timedelta(hours=access_duration_hrs))

    return f'{account_url}/{container_name}?{container_sas_token}'


def split_container_uri(container_uri: str) -> Tuple[str, str, Optional[str]]:
    """
    Args:
        container_uri: str, URI to blob storage container
            <account_url>/<container>?<sas_token>

    Returns: account_url, container_name, sas_token
    """
    
    account_container = container_uri.split('?', maxsplit=1)[0]
    account_url, container_name = account_container.rsplit('/', maxsplit=1)
    sas_token = get_sas_token_from_uri(container_uri)
    return account_url, container_name, sas_token
