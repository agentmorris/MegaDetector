"""

url_utils.py

Frequently-used functions for downloading or manipulating URLs

"""

#%% Imports and constants

import os
import re
import urllib
import tempfile
import requests

from functools import partial
from tqdm import tqdm
from urllib.parse import urlparse
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

url_utils_temp_dir = None
max_path_len = 255


#%% Download functions

class DownloadProgressBar():
    """
    Progress updater based on the progressbar2 package.
    
    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """
    
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            # This is a pretty random import I'd rather not depend on outside of the
            # rare case where it's used, so importing locally
            # pip install progressbar2
            import progressbar            
            self.pbar = progressbar.ProgressBar(max_value=total_size)
            self.pbar.start()
            
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
            

def get_temp_folder(preferred_name='url_utils'):
    """
    Gets a temporary folder for use within this module.
    
    Args:
        preferred_name (str, optional): subfolder to use within the system temp folder
        
    Returns:
        str: the full path to the temporary subfolder
    """
    global url_utils_temp_dir
    
    if url_utils_temp_dir is None:
        url_utils_temp_dir = os.path.join(tempfile.gettempdir(),preferred_name)
        os.makedirs(url_utils_temp_dir,exist_ok=True)
        
    return url_utils_temp_dir
    
           
def download_url(url, 
                 destination_filename=None, 
                 progress_updater=None, 
                 force_download=False, 
                 verbose=True,
                 escape_spaces=True):
    """
    Downloads a URL to a file.  If no file is specified, creates a temporary file, 
    making a best effort to avoid filename collisions.
    
    Prints some diagnostic information and makes sure to omit SAS tokens from printouts.
    
    Args:
        url (str): the URL to download
        destination_filename (str, optional): the target filename; if None, will create
            a file in system temp space        
        progress_updater (object or bool, optional): can be "None", "False", "True", or a 
            specific callable object.  If None or False, no progress updated will be 
            displayed.  If True, a default progress bar will be created.
        force_download (bool, optional): download this file even if [destination_filename]
            exists.
        verbose (bool, optional): enable additional debug console output
        escape_spaces (bool, optional): replace ' ' with '%20'
        
    Returns:
        str: the filename to which [url] was downloaded, the same as [destination_filename]
        if [destination_filename] was not None
    """
    
    if progress_updater is not None and isinstance(progress_updater,bool):
        if not progress_updater:
            progress_updater = None
        else:
            progress_updater = DownloadProgressBar()
            
    url_no_sas = url.split('?')[0]
        
    if destination_filename is None:
        
        target_folder = get_temp_folder()
        url_without_sas = url.split('?', 1)[0]
        
        # This does not guarantee uniqueness, hence "semi-best-effort"
        url_as_filename = re.sub(r'\W+', '', url_without_sas)
        n_folder_chars = len(url_utils_temp_dir)
        if len(url_as_filename) + n_folder_chars > max_path_len:
            print('Warning: truncating filename target to {} characters'.format(max_path_len))
            url_as_filename = url_as_filename[-1*(max_path_len-n_folder_chars):]
        destination_filename = \
            os.path.join(target_folder,url_as_filename)
        
    if escape_spaces:
        url = url.replace(' ','%20')
        
    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url_no_sas)))
    else:
        if verbose:
            print('Downloading file {} to {}'.format(os.path.basename(url_no_sas),destination_filename),end='')
        target_dir = os.path.dirname(destination_filename)
        os.makedirs(target_dir,exist_ok=True)
        urllib.request.urlretrieve(url, destination_filename, progress_updater)  
        assert(os.path.isfile(destination_filename))
        nBytes = os.path.getsize(destination_filename)
        if verbose:
            print('...done, {} bytes.'.format(nBytes))
        
    return destination_filename


def download_relative_filename(url, output_base, verbose=False):
    """
    Download a URL to output_base, preserving relative path.  Path is relative to 
    the site, so:
        
        https://abc.com/xyz/123.txt
    
    ...will get downloaded to:
        
        output_base/xyz/123.txt        
        
    Args:
        url (str): the URL to download
        output_base (str): the base folder to which we should download this file
        verbose (bool, optional): enable additional debug console output
        
    Returns:
        str: the local destination filename
    """
    
    p = urlparse(url)
    # remove the leading '/'
    assert p.path.startswith('/'); relative_filename = p.path[1:]
    destination_filename = os.path.join(output_base,relative_filename)
    return download_url(url, destination_filename, verbose=verbose)


def _do_parallelized_download(download_info,overwrite=False,verbose=False):
    """
    Internal function for download parallelization.
    """
    
    url = download_info['url']
    target_file = download_info['target_file']
    result = {'status':'unknown','url':url,'target_file':target_file}
    
    if ((os.path.isfile(target_file)) and (not overwrite)):
        if verbose:
            print('Skipping existing file {}'.format(target_file))
        result['status'] = 'skipped'
        return result
    try:
        download_url(url=url, 
                     destination_filename=target_file,
                     verbose=verbose, 
                     force_download=overwrite)
    except Exception as e:
        print('Warning: error downloading URL {}: {}'.format(
            url,str(e)))     
        result['status'] = 'error: {}'.format(str(e))
        return result
    
    result['status'] = 'success'
    return result


def parallel_download_urls(url_to_target_file,verbose=False,overwrite=False,
                           n_workers=20,pool_type='thread'):
    """
    Downloads a list of URLs to local files.
    
    Catches exceptions and reports them in the returned "results" array.    
    
    Args:
        url_to_target_file: a dict mapping URLs to local filenames.
        verbose (bool, optional): enable additional debug console output
        overwrite (bool, optional): whether to overwrite existing local files
        n_workers (int, optional): number of concurrent workers, set to <=1 to disable
            parallelization
        pool_type (str, optional): worker type to use; should be 'thread' or 'process'
        
    Returns:
        list: list of dicts with keys:
            - 'url': the url this item refers to
            - 'status': 'skipped', 'success', or a string starting with 'error'
            - 'target_file': the local filename to which we downloaded (or tried to 
              download) this URL            
    """
    
    all_download_info = []
    
    print('Preparing download list')
    for url in tqdm(url_to_target_file):
        download_info = {}
        download_info['url'] = url
        download_info['target_file'] = url_to_target_file[url]
        all_download_info.append(download_info)
        
    print('Downloading {} images on {} workers'.format(
        len(all_download_info),n_workers))

    if n_workers <= 1:

        results = []
        
        for download_info in tqdm(all_download_info):        
            result = _do_parallelized_download(download_info,overwrite=overwrite,verbose=verbose)
            results.append(result)
        
    else:

        if pool_type == 'thread':
            pool = ThreadPool(n_workers)
        else:
            assert pool_type == 'process', 'Unsupported pool type {}'.format(pool_type)
            pool = Pool(n_workers)
        
        print('Starting a {} pool with {} workers'.format(pool_type,n_workers))
        
        results = list(tqdm(pool.imap(
            partial(_do_parallelized_download,overwrite=overwrite,verbose=verbose),
            all_download_info), total=len(all_download_info)))
                
    return results

    
def test_url(url, error_on_failure=True, timeout=None):
    """
    Tests the availability of [url], returning an http status code.
    
    Args:
        url (str): URL to test
        error_on_failure (bool, optional): whether to error (vs. just returning an
            error code) if accessing this URL fails
        timeout (int, optional): timeout in seconds to wait before considering this 
            access attempt to be a failure; see requests.head() for precise documentation
    
    Returns:
        int: http status code (200 for success)
    """
    
    # r = requests.get(url, stream=True, verify=True, timeout=timeout)
    r = requests.head(url, stream=True, verify=True, timeout=timeout)
    
    if error_on_failure and r.status_code != 200:        
        raise ValueError('Could not access {}: error {}'.format(url,r.status_code))
    return r.status_code
    

def test_urls(urls, error_on_failure=True, n_workers=1, pool_type='thread', timeout=None):
    """
    Verify that URLs are available (i.e., returns status 200).  By default,
    errors if any URL is unavailable.  
    
    Args:
        urls (list): list of URLs to test
        error_on_failure (bool, optional): whether to error (vs. just returning an
            error code) if accessing this URL fails
        n_workers (int, optional): number of concurrent workers, set to <=1 to disable
            parallelization
        pool_type (str, optional): worker type to use; should be 'thread' or 'process'
        timeout (int, optional): timeout in seconds to wait before considering this 
            access attempt to be a failure; see requests.head() for precise documentation
    
    Returns:
        list: a list of http status codes, the same length and order as [urls]
    """
        
    if n_workers <= 1:

        status_codes = []
        
        for url in tqdm(urls):
            
            r = requests.get(url, timeout=timeout)
            
            if error_on_failure and r.status_code != 200:        
                raise ValueError('Could not access {}: error {}'.format(url,r.status_code))
            status_codes.append(r.status_code)
                
    else:

        if pool_type == 'thread':
            pool = ThreadPool(n_workers)
        else:
            assert pool_type == 'process', 'Unsupported pool type {}'.format(pool_type)
            pool = Pool(n_workers)
        
        print('Starting a {} pool with {} workers'.format(pool_type,n_workers))
        
        status_codes = list(tqdm(pool.imap(
            partial(test_url,error_on_failure=error_on_failure,timeout=timeout),
            urls), total=len(urls)))
                
    return status_codes
