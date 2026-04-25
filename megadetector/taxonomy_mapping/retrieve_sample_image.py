"""

retrieve_sample_image.py

Downloader that retrieves images from Bing  images, used for verifying taxonomy
lookups and looking for egregious mismappings (e.g., "snake" being mapped to a fish called
"snake").

Simple wrapper around simple_image_download, but I've had to swap in and out the underlying
downloader a few times, so I use an extra layer of indirection.

"""

#%% Imports and environment

import os

from megadetector.taxonomy_mapping import simple_image_download

default_output_folder = os.path.expanduser('~/tmp/image-download-test')

image_downloader = simple_image_download.Downloader()


#%% Main entry point

def download_images(query,
                    output_directory=default_output_folder,
                    limit=100,
                    verbose=False):

    query = query.replace(' ','+')

    image_downloader.directory = output_directory
    paths = image_downloader.download(query,
                                      limit=limit,
                                      verbose=verbose,
                                      cache=False,
                                      download_cache=False)
    return paths


#%% Test driver

if False:

    pass

    #%%

    paths = download_images(query='redunca',
                            output_directory=output_folder,
                            limit=20,
                            verbose=True)
