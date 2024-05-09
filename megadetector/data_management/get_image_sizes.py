"""

get_image_sizes.py

Given a json-formatted list of image filenames, retrieves the width and height of 
every image, optionally writing the results to a new .json file.

"""

#%% Constants and imports

import argparse
import json
import os
import sys

from PIL import Image

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm

from megadetector.utils.path_utils import find_images

image_base = ''
default_n_threads = 1
use_threads = False


#%% Processing functions

def _get_image_size(image_path,image_prefix=None):
    """
    Support function to get the size of a single image.  Returns a (path,w,h) tuple.
    w and h will be -1 if the image fails to load.
    """
    
    if image_prefix is not None:
        full_path = os.path.join(image_prefix,image_path)
    else:
        full_path = image_path
    
    # Is this image on disk?
    if not os.path.isfile(full_path):
        print('Could not find image {}'.format(full_path))
        return (image_path,-1,-1)

    try:        
        pil_im = Image.open(full_path)
        w = pil_im.width            
        h = pil_im.height
        return (image_path,w,h)
    except Exception as e:    
        print('Error reading image {}: {}'.format(full_path,str(e)))
        return (image_path,-1,-1)
    
    
def get_image_sizes(filenames,image_prefix=None,output_file=None,
                    n_workers=default_n_threads,use_threads=True,
                    recursive=True):
    """
    Gets the width and height of all images in [filenames], which can be:
        
    * A .json-formatted file containing list of strings
    * A folder
    * A list of files

    ...returning a list of (path,w,h) tuples, and optionally writing the results to [output_file].
    
    Args:
        filenames (str or list): the image filenames for which we should retrieve sizes, 
            can be the name of a .json-formatted file containing list of strings, a folder 
            in which we should enumerate images, or a list of files.
        image_prefix (str, optional): optional prefix to add to images to get to full paths;
            useful when [filenames] contains relative files, in which case [image_prefix] is the 
            base folder for the source images.
        output_file (str, optional): a .json file to write the imgae sizes
        n_workers (int, optional): number of parallel workers to use, set to <=1 to
            disable parallelization
        use_threads (bool, optional): whether to use threads (True) or processes (False)
            for parallelization; not relevant if [n_workers] <= 1
        recursive (bool, optional): only relevant if [filenames] is actually a folder,
            determines whether image enumeration within that folder will be recursive
        
    Returns:
        list: list of (path,w,h) tuples
    """        
    
    if output_file is not None:
        assert os.path.isdir(os.path.dirname(output_file)), \
            'Illegal output file {}, parent folder does not exist'.format(output_file)
        
    if isinstance(filenames,str) and os.path.isfile(filenames):
        with open(filenames,'r') as f:        
            filenames = json.load(f)
        filenames = [s.strip() for s in filenames]
    elif isinstance(filenames,str) and os.path.isdir(filenames):
        filenames = find_images(filenames,recursive=recursive,
                                return_relative_paths=False,convert_slashes=True)
    else:
        assert isinstance(filenames,list)        
    
    if n_workers <= 1:
        
        all_results = []
        for i_file,fn in tqdm(enumerate(filenames),total=len(filenames)):
            all_results.append(_get_image_size(fn,image_prefix=image_prefix))
    
    else:
        
        print('Creating a pool with {} workers'.format(n_workers))
        if use_threads:
            pool = ThreadPool(n_workers)        
        else:
            pool = Pool(n_workers)
        # all_results = list(tqdm(pool.imap(process_image, filenames), total=len(filenames)))
        all_results = list(tqdm(pool.imap(
            partial(_get_image_size,image_prefix=image_prefix), filenames), total=len(filenames)))
    
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(all_results,f,indent=1)
            
    return all_results

    
#%% Interactive driver

if False:

    pass    

    #%%
    
    # List images in a test folder
    base_dir = r'c:\temp\test_images'
    image_list_file = os.path.join(base_dir,'images.json')
    relative_image_list_file = os.path.join(base_dir,'images_relative.json')
    image_size_file = os.path.join(base_dir,'image_sizes.json')
    from megadetector.utils import path_utils
    image_names = path_utils.find_images(base_dir,recursive=True)
    
    with open(image_list_file,'w') as f:
        json.dump(image_names,f,indent=1)
        
    relative_image_names = []
    for s in image_names:
        relative_image_names.append(os.path.relpath(s,base_dir))
    
    with open(relative_image_list_file,'w') as f:
        json.dump(relative_image_names,f,indent=1)
    
    
    #%%
    
    get_image_sizes(relative_image_list_file,image_size_file,image_prefix=base_dir,n_threads=4)
    
    
#%% Command-line driver
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames',type=str,
                        help='Folder from which we should fetch image sizes, or .json file with a list of filenames')
    parser.add_argument('output_file',type=str,
                        help='Output file (.json) to which we should write image size information')
    parser.add_argument('--image_prefix', type=str, default=None,
                        help='Prefix to append to image filenames, only relevant if [filenames] points to a list of ' + \
                             'relative paths')        
    parser.add_argument('--n_threads', type=int, default=default_n_threads,
                        help='Number of concurrent workers, set to <=1 to disable parallelization (default {})'.format(
                            default_n_threads))
                        
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()
    
    _ = get_image_sizes(filenames=args.filenames,
                        output_file=args.output_file,
                        image_prefix=args.image_prefix,
                        n_workers=args.n_threads)
        
if __name__ == '__main__':
    
    main()
