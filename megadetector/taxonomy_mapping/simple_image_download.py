"""

simple_image_download.py

Downloader for images from Bing Image Search.

pip install python-magic

# On Windows, also run:
pip install python-magic-bin

"""

#%% Imports

import os
import re
import html
import requests
import magic
import random

from urllib.parse import quote


#%% Constants

BING_IMAGE_SEARCH_URL = 'https://www.bing.com/images/search'
HEADERS = {
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


#%% Support functions

def _extract_image_urls_from_bing(html_text, limit):
    """
    Extract image URLs from Bing Image Search HTML.

    Bing embeds image metadata in elements with class="iusc" and a JSON
    blob in the 'm' attribute, which contains 'murl' (media URL).
    """

    decoded = html.unescape(html_text)
    murl_matches = re.findall(r'"murl"\s*:\s*"(https?://[^"]+)"', decoded)

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in murl_matches:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls[:limit]


def check_webpage(url):
    checked_url = None
    try:
        request = requests.get(url, allow_redirects=True, timeout=10)
        if 'html' not in str(request.content):
            checked_url = request
    except Exception as err:
        print(err)
    return checked_url


#%% Main class

class Downloader:
    """
    Main Downloader
    ::param extension:iterable of Files extensions
    """
    def __init__(self, extensions=None):
        if extensions:
            self._extensions = set(*[extensions])
        else:
            self._extensions = {'.jpg', '.png', '.ico', '.gif', '.jpeg'}
        self._directory = "simple_images/"
        self.get_dirs = set()
        self._cached_urls = {}

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, value):
        self._directory = value

    @property
    def cached_urls(self):
        return self._cached_urls

    @property
    def extensions(self):
        return self._extensions

    @extensions.setter
    def extensions(self, value):
        self._extensions = set([value])


    def get_urls(self):
        return [self._cached_urls[url][1].url
                for url in self._cached_urls]

    def _download_page(self, query):
        """Download Bing Image Search results page for the given query."""
        params = {
            'q': query,
            'form': 'HDRSC2',
            'first': '1',
        }
        resp = requests.get(BING_IMAGE_SEARCH_URL, params=params,
                            headers=HEADERS, timeout=10)
        return resp.text

    def search_urls(self, keywords, limit=1, verbose=False, cache=True, timer=None):
        cache_out = {}
        # Split keywords the same way the original code did
        search = [str(item).strip() for item in keywords.split(',')][0].split()
        count = len(search)

        i = 0
        while i < count:
            query = search[i]
            path = self.generate_dir(query)
            raw_html = self._download_page(query)
            image_urls = _extract_image_urls_from_bing(raw_html, limit + 1)

            for img_url in image_urls[:limit + 1]:
                webpage_url = check_webpage(img_url)
                if webpage_url:
                    file_name = Downloader.gen_fn(webpage_url, query)
                    cache_out[file_name] = [path, webpage_url]
            i += 1

        if verbose:
            for url in cache_out:
                print(url)
        if cache:
            self._cached_urls = cache_out
        if not cache_out:
            print('==='*15 + ' < ' + 'NO PICTURES FOUND' + ' > ' + '==='*15)
        return cache_out

    def download(self,
                 keywords=None,
                 limit=1,
                 verbose=False,
                 cache=True,
                 download_cache=False,
                 timer=None):
        if not download_cache:
            content = self.search_urls(keywords, limit, verbose, cache, timer)
        else:
            content = self._cached_urls
            if not content:
                print('Downloader has not URLs saved in Memory yet, run Downloader.search_urls to find pics first')
        paths = []
        for name, (path, url) in content.items():
            fullpath = os.path.join(path, name)
            paths.append(fullpath)
            with open(fullpath, 'wb') as file:
                file.write(url.content)
            if verbose:
                print(f'File Name={name}, Downloaded from {url.url}')
        return paths

    def _create_directories(self, name):
        dir_path = os.path.join(self._directory, name)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError:
            raise
        self.get_dirs.update([name])
        return

    def generate_dir(self, dir_name):
        """Generate Path and Directory, also check if Directory exists or not """
        dir_name = dir_name.replace(" ", "_")
        if dir_name in self.get_dirs:
            pass
        else:
            self._create_directories(dir_name)
        return os.path.join(self._directory,dir_name)

    @staticmethod
    def gen_fn(check, name):
        """Create a file name string and generate a random identifiers otherwise won't import same pic twice"""
        id = str(hex(random.randrange(1000)))
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(check.content)
        file_extension = f'.{file_type.split("/")[1]}'
        file_name = str(name) + "_" + id[2:] + file_extension
        return file_name

    def flush_cache(self):
        """Clear the Downloader instance cache"""
        self._cached_urls = set()


#%% Scrap

if False:

    pass

    #%% Basic test

    downloader = Downloader()
    keywords='mammal'
    limit=10
    verbose=True
    cache=False
    download_cache=False

    output_folder = os.path.expanduser('~/tmp/image-download-test')
    os.makedirs(output_folder,exist_ok=True)
    downloader.directory = output_folder

    downloader.download(keywords=keywords,
                        limit=limit,
                        verbose=verbose,
                        cache=cache,
                        download_cache=download_cache)
