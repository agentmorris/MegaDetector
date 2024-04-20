########
#
# url_utils.py
#
# Frequently-used functions for downloading or manipulating URLs
#
########

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

    global url_utils_temp_dir
    
    if url_utils_temp_dir is None:
        url_utils_temp_dir = os.path.join(tempfile.gettempdir(),preferred_name)
        os.makedirs(url_utils_temp_dir,exist_ok=True)
        
    return url_utils_temp_dir
    
           
def download_url(url, destination_filename=None, progress_updater=None, 
                 force_download=False, verbose=True):
    """
    Download a URL to a file.  If no file is specified, creates a temporary file, 
    with a semi-best-effort to avoid filename collisions.
    
    Prints some diagnostic information and makes sure to omit SAS tokens from printouts.
    
    progress_updater can be "None", "True", or a specific callback.
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
    """
    
    p = urlparse(url)
    # remove the leading '/'
    assert p.path.startswith('/'); relative_filename = p.path[1:]
    destination_filename = os.path.join(output_base,relative_filename)
    download_url(url, destination_filename, verbose=verbose)

def _do_parallelized_download(download_info,overwrite=False,verbose=False):
    """
    Internal function for download parallelization.
    """
    
    url = download_info['url']
    target_file = download_info['target_file']
    result = {'status':'unknown','url':url,'target_file':target_file}
    
    if ((os.path.isfile(target_file)) and (not overwrite)):
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
    Download a list of URLs to local files.  url_to_target_file should
    be a dict mapping URLs to output files.  Catches exceptions and reports
    them in the returned "results" array.    
    """
    
    all_download_info = []
    for url in url_to_target_file:
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

    
def test_urls(urls, error_on_failure=True):
    """
    Verify that a list of URLs is available (returns status 200).  By default,
    errors if any URL is unavailable.  If error_on_failure is False, returns
    status codes for each URL.
    
    TODO: trivially parallelizable.
    """
    
    status_codes = []
    
    for url in tqdm(urls):
        
        r = requests.get(url)
        
        if error_on_failure and r.status_code != 200:        
            raise ValueError('Could not access {}: error {}'.format(url,r.status_code))
        status_codes.append(r.status_code)
        
    return status_codes

