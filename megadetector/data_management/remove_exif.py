"""

remove_exif.py

Removes all EXIF/IPTC/XMP metadata from a folder of images, without making 
backup copies, using pyexiv2.  Ignores non-jpeg images.

This module is rarely used, and pyexiv2 is not thread-safe, so pyexiv2 is not
included in package-level dependency lists.  YMMV.

"""

#%% Imports and constants

import os
import glob

from multiprocessing.pool import Pool as Pool
from tqdm import tqdm


#%% Support functions

# Pyexif2 is not thread safe, do not call this function in parallel within a process
#
# Parallelizing across processes is fine.
def remove_exif_from_image(fn):

    import pyexiv2
    
    try:
        img = pyexiv2.Image(fn)
        img.clear_exif()
        img.clear_iptc()
        img.clear_xmp()
        img.close()        
    except Exception as e:
        print('EXIF error on {}: {}'.format(fn,str(e)))
    
    return True


#%% Remove EXIF data

def remove_exif(image_base_folder,recursive=True,n_processes=1):
    """
    Removes all EXIF/IPTC/XMP metadata from a folder of images, without making 
    backup copies, using pyexiv2.  Ignores non-jpeg images.
    
    Args:
        image_base_folder (str): the folder from which we should remove EXIF data
        recursive (bool, optional): whether to process [image_base_folder] recursively
        n_processes (int, optional): number of concurrent workers.  Because pyexiv2 is not
            thread-safe, only process-based parallelism is supported.        
    """
    try:
        import pyexiv2 #noqa
    except:
        print('pyexiv2 not available; try "pip install pyexiv2"')
        raise

        
    ##%% List files

    assert os.path.isdir(image_base_folder), \
        'Could not find folder {}'.format(image_base_folder)
    all_files = [f for f in glob.glob(image_base_folder+ "*/**", recursive=recursive)]
    image_files = [s for s in all_files if \
                   (s.lower().endswith('.jpg') or s.lower().endswith('.jpeg'))]
        

    ##%% Remove EXIF data (execution)

    if n_processes == 1:
        
        # fn = image_files[0]
        for fn in tqdm(image_files):
            remove_exif_from_image(fn)
            
    else:
        # pyexiv2 is not thread-safe, so we need to use processes
        print('Starting parallel process pool with {} workers'.format(n_processes))
        pool = Pool(n_processes)
        _ = list(tqdm(pool.imap(remove_exif_from_image,image_files),total=len(image_files)))
            
# ...remove_exif(...)


#%% Command-line driver

## TODO
