"""

url_utils.py

Frequently-used functions for downloading, manipulating, or serving URLs

"""

#%% Imports and constants

import os
import re
import urllib
import urllib.request
import urllib.error
import requests
import shutil
import pytest
import socketserver
import threading
import http.server

from functools import partial
from tqdm import tqdm
from urllib.parse import urlparse
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

from megadetector.utils.ct_utils import make_test_folder
from megadetector.utils.ct_utils import make_temp_folder

max_path_len = 255


#%% Download functions

class DownloadProgressBar:
    """
    Progress updater based on the progressbar2 package.

    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """


    def __init__(self):

        self.pbar = None


    def __call__(self, block_num, block_size, total_size): # noqa

        if not self.pbar:
            try:
                import progressbar # type: ignore
                self.pbar = progressbar.ProgressBar(max_value=total_size)
                self.pbar.start()
            except ImportError:
                self.pbar = None
                # print("ProgressBar not available, install 'progressbar2' for visual progress.")

        if self.pbar:
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(downloaded)
            else:
                self.pbar.finish()


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

        target_folder = make_temp_folder(subfolder='url_utils',append_guid=False)
        url_without_sas = url.split('?', 1)[0]

        # This does not guarantee uniqueness, hence "semi-best-effort"
        url_as_filename = re.sub(r'\W+', '', url_without_sas)

        n_folder_chars = len(target_folder)

        if (len(url_as_filename) + n_folder_chars) >= max_path_len:
            print('Warning: truncating filename target to {} characters'.format(max_path_len))
            max_fn_len = max_path_len - (n_folder_chars + 1)
            url_as_filename = url_as_filename[-1 * max_fn_len:]
        destination_filename = \
            os.path.join(target_folder,url_as_filename)

    # ...if the destination filename wasn't specified

    if escape_spaces:
        url = url.replace(' ','%20')

    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url_no_sas)))
    else:
        if verbose:
            print('Downloading file {} to {}'.format(os.path.basename(url_no_sas),destination_filename),end='')
        target_dir = os.path.dirname(destination_filename)
        if len(target_dir) > 0:
            os.makedirs(target_dir,exist_ok=True)
        urllib.request.urlretrieve(url, destination_filename, progress_updater)
        assert(os.path.isfile(destination_filename))
        n_bytes = os.path.getsize(destination_filename)
        if verbose:
            print('...done, {} bytes.'.format(n_bytes))

    return destination_filename

# ...def download_url(...)


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

# ...def download_relative_filename(...)


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

# ...def _do_parallelized_download(...)


def parallel_download_urls(url_to_target_file,
                           verbose=False,
                           overwrite=False,
                           n_workers=20,
                           pool_type='thread'):
    """
    Downloads a list of URLs to local files.

    Catches exceptions and reports them in the returned "results" array.

    Args:
        url_to_target_file (dict): a dict mapping URLs to local filenames.
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

    if verbose:
        print('Preparing download list')
    for url in tqdm(url_to_target_file, disable=(not verbose)):
        download_info = {}
        download_info['url'] = url
        download_info['target_file'] = url_to_target_file[url]
        all_download_info.append(download_info)

    if verbose:
        print('Downloading {} images on {} workers'.format(
            len(all_download_info),n_workers))

    if n_workers <= 1:

        results = []

        for download_info in tqdm(all_download_info, disable=(not verbose)):
            result = _do_parallelized_download(download_info,overwrite=overwrite,verbose=verbose)
            results.append(result)

    else:

        pool = None

        try:
            if pool_type == 'thread':
                pool = ThreadPool(n_workers)
            else:
                assert pool_type == 'process', 'Unsupported pool type {}'.format(pool_type)
                pool = Pool(n_workers)

            if verbose:
                print('Starting a {} pool with {} workers'.format(pool_type,n_workers))

            results = list(tqdm(pool.imap(
                partial(_do_parallelized_download,overwrite=overwrite,verbose=verbose),
                all_download_info), total=len(all_download_info), disable=(not verbose)))

        finally:
            if pool:
                pool.close()
                pool.join()
                print("Pool closed and joined for parallel URL downloads")

    return results

# ...def parallel_download_urls(...)


@pytest.mark.skip(reason="This is not a test function")
def test_url(url,error_on_failure=True,timeout=None):
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

    r = requests.head(url, stream=True, verify=True, timeout=timeout)

    if error_on_failure and r.status_code != 200:
        raise ValueError('Could not access {}: error {}'.format(url,r.status_code))
    return r.status_code


@pytest.mark.skip(reason="This is not a test function")
def test_urls(urls,error_on_failure=True,n_workers=1,pool_type='thread',timeout=None,verbose=False):
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
        verbose (bool, optional): enable additional debug output

    Returns:
        list: a list of http status codes, the same length and order as [urls]
    """

    if n_workers <= 1:

        status_codes = []

        for url in tqdm(urls,disable=(not verbose)):

            r = requests.get(url, timeout=timeout)

            if error_on_failure and r.status_code != 200:
                raise ValueError('Could not access {}: error {}'.format(url,r.status_code))
            status_codes.append(r.status_code)

    else:

        pool = None
        try:
            if pool_type == 'thread':
                pool = ThreadPool(n_workers)
            else:
                assert pool_type == 'process', 'Unsupported pool type {}'.format(pool_type)
                pool = Pool(n_workers)

            if verbose:
                print('Starting a {} pool with {} workers'.format(pool_type,n_workers))

            status_codes = list(tqdm(pool.imap(
                partial(test_url,error_on_failure=error_on_failure,timeout=timeout),
                urls), total=len(urls), disable=(not verbose)))
        finally:
            if pool:
                pool.close()
                pool.join()
                print('Pool closed and joined for URL tests')

    return status_codes

# ...def test_urls(...)


def get_url_size(url,verbose=False,timeout=None):
    """
    Get the size of the file pointed to by a URL, based on the Content-Length property.  If the
    URL is not available, or the Content-Length property is not available, or the content-Length
    property is not an integer, returns None.

    Args:
        url (str): the url to test
        verbose (bool, optional): enable additional debug output
        timeout (int, optional): timeout in seconds to wait before considering this
            access attempt to be a failure; see requests.head() for precise documentation

    Returns:
        int: the file size in bytes, or None if it can't be retrieved
    """

    try:
        r = urllib.request.Request(url,method='HEAD')
        f = urllib.request.urlopen(r, timeout=timeout)
        if f.status != 200:
            if verbose:
                print('Status {} retrieving file size for {}'.format(f.status,url))
            return None
        size_bytes_str = f.headers.get('Content-Length')
        if size_bytes_str is None:
            if verbose:
                print('No Content-Length header for {}'.format(url))
            return None
        size_bytes = int(size_bytes_str)
        return size_bytes
    except Exception as e:
        if verbose:
            print('Error retrieving file size for {}:\n{}'.format(url,str(e)))
        return None

# ...def get_url_size(...)


def get_url_sizes(urls,n_workers=1,pool_type='thread',timeout=None,verbose=False):
    """
    Retrieve file sizes for the URLs specified by [urls].  Returns None for any URLs
    that we can't access, or URLs for which the Content-Length property is not set.

    Args:
        urls (list): list of URLs for which we should retrieve sizes
        n_workers (int, optional): number of concurrent workers, set to <=1 to disable
            parallelization
        pool_type (str, optional): worker type to use; should be 'thread' or 'process'
        timeout (int, optional): timeout in seconds to wait before considering this
            access attempt to be a failure; see requests.head() for precise documentation
        verbose (bool, optional): print additional debug information

    Returns:
        dict: maps urls to file sizes, which will be None for URLs for which we were unable
        to retrieve a valid size.
    """

    url_to_size = {}

    if n_workers <= 1:

        for url in tqdm(urls, disable=(not verbose)):
            url_to_size[url] = get_url_size(url,verbose=verbose,timeout=timeout)

    else:

        pool = None
        try:
            if pool_type == 'thread':
                pool = ThreadPool(n_workers)
            else:
                assert pool_type == 'process', 'Unsupported pool type {}'.format(pool_type)
                pool = Pool(n_workers)

            if verbose:
                print('Starting a {} pool with {} workers'.format(pool_type,n_workers))

            file_sizes = list(tqdm(pool.imap(
                partial(get_url_size,verbose=verbose,timeout=timeout),
                urls), total=len(urls), disable=(not verbose)))

            for i_url,url in enumerate(urls):
                url_to_size[url] = file_sizes[i_url]
        finally:
            if pool:
                pool.close()
                pool.join()
                print('Pool closed and joined for URL size checks')

    return url_to_size


#%%  Singleton HTTP server

class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    SimpleHTTPRequestHandler sublcass that suppresses console printouts
    """
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args): # noqa
        pass


class SingletonHTTPServer:
    """
    HTTP server that runs on a local port, serving a particular local folder.  Runs as a
    singleton, so starting a server in a new folder closes the previous server.  I use this
    primarily to serve MD/SpeciesNet previews from manage_local_batch, which can exceed
    the 260-character filename length limitation imposed by browser on Windows, so really the
    point here is just to remove characters from the URL.
    """

    _server = None
    _thread = None

    @classmethod
    def start_server(cls, directory, port=8000, host='localhost'):
        """
        Start or restart the HTTP server with a specific directory

        Args:
            directory (str): the root folder served by the server
            port (int, optional): the port on which to create the server
            host (str, optional): the host on which to listen, typically
                either "localhost" (default) or "0.0.0.0"

        Returns:
            str: URL to the running host
        """

        # Stop the existing server instance if necessary
        cls.stop_server()

        # Create new server
        handler = partial(QuietHTTPRequestHandler, directory=directory)
        cls._server = socketserver.TCPServer((host, port), handler)

        # Start server in daemon thread (dies when parent process dies)
        cls._thread = threading.Thread(target=cls._server.serve_forever)
        cls._thread.daemon = True
        cls._thread.start()

        print(f"Serving {directory} at http://{host}:{port}")
        return f"http://{host}:{port}"


    @classmethod
    def stop_server(cls):
        """
        Stop the current server (if one is running)
        """

        if cls._server:
            cls._server.shutdown()
            cls._server.server_close()
            cls._server = None
        if cls._thread:
            cls._thread.join(timeout=1)
            cls._thread = None


    @classmethod
    def is_running(cls):
        """
        Check whether the server is currently running.

        Returns:
            bool: True if the server is running
        """

        return (cls._server is not None) and \
            (cls._thread is not None) and \
            (cls._thread.is_alive())

# ...class SingletonHTTPServer


#%% Tests

# Constants for tests

SMALL_FILE_URL = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
REDIRECT_SRC_URL = "http://google.com"
REDIRECT_DEST_URL = "https://www.google.com/"
NON_EXISTENT_URL = "https://example.com/non_existent_page_404.html"
DEFINITELY_NON_EXISTENT_DOMAIN_URL = "https://thisshouldnotexist1234567890.com/file.txt"
RELATIVE_DOWNLOAD_URL = "https://raw.githubusercontent.com/agentmorris/MegaDetector/main/README.md"
RELATIVE_DOWNLOAD_CONTAIN_TOKEN = 'agentmorris'
RELATIVE_DOWNLOAD_NOT_CONTAIN_TOKEN = 'github'


class TestUrlUtils:
    """
    Tests for url_utils.py
    """

    def set_up(self):
        """
        Create a temporary directory for testing.
        """

        self.test_dir = make_test_folder(subfolder='url_utils_tests')
        self.download_target_dir = os.path.join(self.test_dir, 'downloads')
        os.makedirs(self.download_target_dir, exist_ok=True)


    def tear_down(self):
        """
        Remove the temporary directory after tests and restore module temp_dir.
        """

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


    def test_download_url_to_specified_file(self):
        """
        Test download_url with a specified destination filename.
        """

        dest_filename = os.path.join(self.download_target_dir, "downloaded_google_logo.png")
        returned_filename = download_url(SMALL_FILE_URL,
                                         destination_filename=dest_filename,
                                         verbose=False)
        assert returned_filename == dest_filename
        assert os.path.exists(dest_filename)
        assert os.path.getsize(dest_filename) > 1000


    def test_download_url_to_temp_file(self):
        """
        Test download_url when destination_filename is None.
        """

        returned_filename = download_url(SMALL_FILE_URL,
                                         destination_filename=None,
                                         verbose=False)
        assert os.path.exists(returned_filename)
        assert os.path.getsize(returned_filename) > 1000


    def test_download_url_non_existent(self):
        """
        Test download_url with a non-existent URL.
        """

        dest_filename = os.path.join(self.download_target_dir, "non_existent.html")
        try:
            download_url(NON_EXISTENT_URL, destination_filename=dest_filename, verbose=False)
            raise AssertionError("urllib.error.HTTPError not raised for 404")
        except urllib.error.HTTPError:
            pass

        try:
            download_url(DEFINITELY_NON_EXISTENT_DOMAIN_URL,
                         destination_filename=dest_filename,
                         verbose=False)
            raise AssertionError(
                "urllib.error.URLError or requests.exceptions.ConnectionError not raised for DNS failure")
        except urllib.error.URLError:
            pass
        except requests.exceptions.ConnectionError:
            pass


    def test_download_url_force_download(self):
        """
        Test the force_download parameter of download_url.
        """

        dest_filename = os.path.join(self.download_target_dir, "force_test.png")

        download_url(SMALL_FILE_URL, destination_filename=dest_filename, verbose=False)
        assert os.path.exists(dest_filename)
        initial_mtime = os.path.getmtime(dest_filename)

        download_url(SMALL_FILE_URL, destination_filename=dest_filename, verbose=True)
        assert os.path.getmtime(dest_filename) == initial_mtime

        download_url(SMALL_FILE_URL,
                     destination_filename=dest_filename,
                     force_download=True,
                     verbose=False)
        assert os.path.exists(dest_filename)


    def test_download_url_escape_spaces(self):
        """
        Test download_url with spaces in the URL.
        """

        dest_filename = os.path.join(self.download_target_dir, "escape_test.png")
        download_url(SMALL_FILE_URL,
                     destination_filename=dest_filename,
                     escape_spaces=True,
                     verbose=False)
        assert os.path.exists(dest_filename)


    def test_download_relative_filename(self):
        """
        Test download_relative_filename.
        """

        output_base = os.path.join(self.download_target_dir, "relative_dl")
        returned_filename = download_relative_filename(RELATIVE_DOWNLOAD_URL, output_base, verbose=False)
        assert RELATIVE_DOWNLOAD_CONTAIN_TOKEN in returned_filename
        assert RELATIVE_DOWNLOAD_NOT_CONTAIN_TOKEN not in returned_filename
        assert os.path.exists(returned_filename)
        assert os.path.getsize(returned_filename) > 100


    def test_parallel_download_urls(self):
        """
        Test parallel_download_urls (with n_workers=1 for simplicity).
        """

        url1_target = os.path.join(self.download_target_dir, "parallel_dl_1.png")
        url2_target = os.path.join(self.download_target_dir, "parallel_dl_2_nonexistent.html")

        url_to_target_file = {
            SMALL_FILE_URL: url1_target,
            NON_EXISTENT_URL: url2_target
        }

        results = parallel_download_urls(url_to_target_file, n_workers=1, verbose=False)

        assert len(results) == 2

        status_map = {res['url']: res for res in results}

        assert status_map[SMALL_FILE_URL]['status'] == 'success'
        assert status_map[SMALL_FILE_URL]['target_file'] == url1_target
        assert os.path.exists(url1_target)

        assert status_map[NON_EXISTENT_URL]['status'].startswith('error: HTTP Error 404')
        assert status_map[NON_EXISTENT_URL]['target_file'] == url2_target
        assert not os.path.exists(url2_target)

        if not os.path.exists(url1_target):
             download_url(SMALL_FILE_URL, url1_target, verbose=False)
        results_skip = parallel_download_urls({SMALL_FILE_URL: url1_target},
                                              n_workers=1,
                                              overwrite=False,
                                              verbose=True)
        assert results_skip[0]['status'] == 'skipped'

        results_overwrite = parallel_download_urls({SMALL_FILE_URL: url1_target},
                                                   n_workers=1,
                                                   overwrite=True,
                                                   verbose=False)
        assert results_overwrite[0]['status'] == 'success'


    def test_test_url_and_test_urls(self):
        """
        Test test_url and test_urls functions.
        """

        assert test_url(SMALL_FILE_URL, error_on_failure=False, timeout=10) == 200
        assert test_url(REDIRECT_SRC_URL, error_on_failure=False, timeout=10) in (200,301)

        status_non_existent = test_url(NON_EXISTENT_URL, error_on_failure=False, timeout=5)
        assert status_non_existent == 404

        try:
            test_url(NON_EXISTENT_URL, error_on_failure=True, timeout=5)
            raise AssertionError("ValueError not raised for NON_EXISTENT_URL")
        except ValueError:
            pass

        try:
            test_url(DEFINITELY_NON_EXISTENT_DOMAIN_URL,
                     error_on_failure=True,
                     timeout=5)
            raise AssertionError("requests.exceptions.ConnectionError or urllib.error.URLError not raised")
        except requests.exceptions.ConnectionError:
            pass
        except urllib.error.URLError:
            pass


        urls_to_test = [SMALL_FILE_URL, NON_EXISTENT_URL]
        status_codes = test_urls(urls_to_test, error_on_failure=False, n_workers=1, timeout=10)
        assert len(status_codes) == 2
        assert status_codes[0] == 200
        assert status_codes[1] == 404

        try:
            test_urls(urls_to_test, error_on_failure=True, n_workers=1, timeout=5)
            raise AssertionError("ValueError not raised for urls_to_test")
        except ValueError:
            pass

        good_urls = [SMALL_FILE_URL, REDIRECT_SRC_URL]
        good_status_codes = test_urls(good_urls, error_on_failure=True, n_workers=1, timeout=10)
        assert good_status_codes == [200, 200]


    def test_get_url_size_and_sizes(self):
        """
        Test get_url_size and get_url_sizes functions.
        """

        size = get_url_size(SMALL_FILE_URL, timeout=10)
        assert size is not None
        assert size > 1000

        size_dynamic = get_url_size(REDIRECT_DEST_URL, timeout=10, verbose=True)
        if size_dynamic is not None:
            assert isinstance(size_dynamic, int)

        size_non_existent = get_url_size(NON_EXISTENT_URL, timeout=5)
        assert size_non_existent is None

        size_bad_domain = get_url_size(DEFINITELY_NON_EXISTENT_DOMAIN_URL, timeout=5)
        assert size_bad_domain is None

        urls_for_size = [SMALL_FILE_URL, NON_EXISTENT_URL, REDIRECT_DEST_URL]
        sizes_map = get_url_sizes(urls_for_size, n_workers=1, timeout=10)

        assert SMALL_FILE_URL in sizes_map
        assert sizes_map[SMALL_FILE_URL] == size

        assert NON_EXISTENT_URL in sizes_map
        assert sizes_map[NON_EXISTENT_URL] is None

        assert REDIRECT_DEST_URL in sizes_map
        assert sizes_map[REDIRECT_DEST_URL] == size_dynamic


def _test_url_utils():
    """
    Runs all tests in the TestUrlUtils class.  I generally disable this during testing
    because it creates irritating nondeterminism (because it depends on downloading
    stuff from the Internet), and this is neither a core module nor a module that changes
    often.
    """

    test_instance = TestUrlUtils()
    test_instance.set_up()
    try:
        test_instance.test_download_url_to_specified_file()
        test_instance.test_download_url_to_temp_file()
        test_instance.test_download_url_non_existent()
        test_instance.test_download_url_force_download()
        test_instance.test_download_url_escape_spaces()
        test_instance.test_download_relative_filename()
        test_instance.test_parallel_download_urls()
        test_instance.test_test_url_and_test_urls()
        test_instance.test_get_url_size_and_sizes()
    finally:
        test_instance.tear_down()

# from IPython import embed; embed()
# test_url_utils()
