"""

md_to_labelme.py

"Converts" a MegaDetector output .json file to labelme format (one .json per image
file).  "Convert" is in quotes because this is an opinionated transformation that 
requires a confidence threshold.

TODO:
   
* support variable confidence thresholds across classes
* support classification data

"""

#%% Imports and constants

import os
import json

from tqdm import tqdm

from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from functools import partial

from megadetector.visualization.visualization_utils import open_image
from megadetector.utils.ct_utils import truncate_float
from megadetector.detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP

output_precision = 3
default_confidence_threshold = 0.15


#%% Functions

def get_labelme_dict_for_image(im,image_base_name=None,category_id_to_name=None,
                               info=None,confidence_threshold=None):
    """
    For the given image struct in MD results format, reformat the detections into
    labelme format.
    
    Args:
        im (dict): MegaDetector-formatted results dict, must include 'height' and 'width' fields
        image_base_name (str, optional): written directly to the 'imagePath' field in the output; 
            defaults to os.path.basename(im['file']).
        category_id_to_name (dict, optional): maps string-int category IDs to category names, defaults
            to the standard MD categories
        info (dict, optional): arbitrary metadata to write to the "detector_info" field in the output
            dict
        confidence_threshold (float, optional): only detections at or above this confidence threshold
            will be included in the output dict
    
    Return:
        dict: labelme-formatted dictionary, suitable for writing directly to a labelme-formatted .json file
    """
    
    if image_base_name is None:
        image_base_name = os.path.basename(im['file'])
        
    if category_id_to_name:
        category_id_to_name = DEFAULT_DETECTOR_LABEL_MAP
        
    if confidence_threshold is None:        
        confidence_threshold = -1.0
     
    output_dict = {}
    if info is not None:
        output_dict['detector_info'] = info
    output_dict['version'] = '5.3.0a0'
    output_dict['flags'] = {}
    output_dict['shapes'] = []
    output_dict['imagePath'] = image_base_name
    output_dict['imageHeight'] = im['height']
    output_dict['imageWidth'] = im['width']
    output_dict['imageData'] = None
    output_dict['detections'] = im['detections']
    
    # det = im['detections'][1]
    for det in im['detections']:
        
        if det['conf'] < confidence_threshold:
            continue
        
        shape = {}
        shape['conf'] = det['conf']
        shape['label'] = category_id_to_name[det['category']] 
        shape['shape_type'] = 'rectangle'
        shape['description'] = ''
        shape['group_id'] = None
        
        # MD boxes are [x_min, y_min, width_of_box, height_of_box] (relative)
        # 
        # labelme boxes are [[x0,y0],[x1,y1]] (absolute)
        x0 = truncate_float(det['bbox'][0] * im['width'],output_precision)
        y0 = truncate_float(det['bbox'][1] * im['height'],output_precision)
        x1 = truncate_float(x0 + det['bbox'][2] * im['width'],output_precision)
        y1 = truncate_float(y0 + det['bbox'][3] * im['height'],output_precision)
        shape['points'] = [[x0,y0],[x1,y1]]
        output_dict['shapes'].append(shape)
    
    # ...for each detection
    
    return output_dict

# ...def get_labelme_dict_for_image()


def _write_output_for_image(im,image_base,extension_prefix,info,
                            confidence_threshold,category_id_to_name,overwrite,
                            verbose=False):
    
    if 'failure' in im and im['failure'] is not None:
        assert 'detections' not in im or im['detections'] is None
        if verbose:
            print('Skipping labelme file generation for failed image {}'.format(
                im['file']))
        return
        
    im_full_path = os.path.join(image_base,im['file'])
    json_path = os.path.splitext(im_full_path)[0] + extension_prefix + '.json'
    
    if (not overwrite) and (os.path.isfile(json_path)):
        if verbose:
            print('Skipping existing file {}'.format(json_path))
        return

    output_dict = get_labelme_dict_for_image(im,
                                             image_base_name=os.path.basename(im_full_path),
                                             category_id_to_name=category_id_to_name,
                                             info=info,
                                             confidence_threshold=confidence_threshold)
            
    with open(json_path,'w') as f:
        json.dump(output_dict,f,indent=1)
    
# ...def write_output_for_image(...)



def md_to_labelme(results_file,image_base,confidence_threshold=None,
                  overwrite=False,extension_prefix='',n_workers=1,
                  use_threads=False,bypass_image_size_read=False,
                  verbose=False):
    """
    For all the images in [results_file], write a .json file in labelme format alongside the
    corresponding relative path within image_base.
    
    Args:
        results_file (str): MD results .json file to convert to Labelme format
        image_base (str): folder of images; filenames in [results_file] should be relative to
            this folder
        confidence_threshold (float, optional): only detections at or above this confidence threshold
            will be included in the output dict
        overwrite (bool, optional): whether to overwrite existing output files; if this is False
            and the output file for an image exists, we'll skip that image
        extension_prefix (str, optional): if non-empty, "extension_prefix" will be inserted before the .json 
            extension
        n_workers (int, optional): enables multiprocessing if > 1
        use_threads (bool, optional): if [n_workers] > 1, determines whether we parallelize via threads (True)
            or processes (False)
        bypass_image_size_read (bool, optional): if True, skips reading image sizes and trusts whatever is in
            the MD results file (don't set this to "True" if your MD results file doesn't contain image sizes)
        verbose (bool, optional): enables additionald ebug output    
    """
    
    if extension_prefix is None:
        extension_prefix = ''
        
    # Load MD results if necessary
    if isinstance(results_file,dict):
        md_results = results_file
    else:
        print('Loading MD results...')
        with open(results_file,'r') as f:
            md_results = json.load(f)
        
    # Read image sizes if necessary            
    if bypass_image_size_read:     
        
        print('Bypassing image size read')
        
    else:
    
        # TODO: parallelize this loop
    
        print('Reading image sizes...')
                
        # im = md_results['images'][0]
        for im in tqdm(md_results['images']):
            
            # Make sure this file exists
            im_full_path = os.path.join(image_base,im['file'])
            assert os.path.isfile(im_full_path), 'Image file {} does not exist'.format(im_full_path)
            
            json_path = os.path.splitext(im_full_path)[0] + extension_prefix + '.json'
            
            # Don't even bother reading sizes for files we're not going to generate
            if (not overwrite) and (os.path.isfile(json_path)):
                continue
            
            # Load w/h information if necessary
            if 'height' not in im or 'width' not in im:
                
                try:
                    pil_im = open_image(im_full_path)
                    im['width'] = pil_im.width
                    im['height'] = pil_im.height
                except Exception:
                    print('Warning: cannot open image {}, treating as a failure during inference'.format(
                        im_full_path))
                    if 'failure' not in im:
                        im['failure'] = 'Failure image access'        
    
            # ...if we need to read w/h information
            
        # ...for each image
        
    # ...if we're not bypassing image size read        
    
    print('\nGenerating labelme files...')
        
    # Write output
    if n_workers <= 1:
        for im in tqdm(md_results['images']):    
            _write_output_for_image(im,image_base,extension_prefix,md_results['info'],confidence_threshold,
                                   md_results['detection_categories'],overwrite,verbose)
    else:
        if use_threads:
            print('Starting parallel thread pool with {} workers'.format(n_workers))
            pool = ThreadPool(n_workers)
        else:
            print('Starting parallel process pool with {} workers'.format(n_workers))
            pool = Pool(n_workers)
        _ = list(tqdm(pool.imap(
                partial(_write_output_for_image,
                        image_base=image_base,extension_prefix=extension_prefix,
                        info=md_results['info'],confidence_threshold=confidence_threshold,
                        category_id_to_name=md_results['detection_categories'],
                        overwrite=overwrite,verbose=verbose),
                 md_results['images']),
                 total=len(md_results['images'])))
            
    # ...for each image
    
# ...def md_to_labelme()


#%% Interactive driver

if False:
    
    pass

    #%% Configure options
    
    md_results_file = os.path.expanduser('~/data/md-test.json')
    coco_output_file = os.path.expanduser('~/data/md-test-coco.json')
    image_folder = os.path.expanduser('~/data/md-test')    
    confidence_threshold = 0.2
    overwrite = True    
    
    
    #%% Programmatic execution
    
    md_to_labelme(results_file=md_results_file,
                  image_base=image_folder,
                  confidence_threshold=confidence_threshold,
                  overwrite=overwrite)

    
    #%% Command-line execution
    
    s = 'python md_to_labelme.py {} {} --confidence_threshold {}'.format(md_results_file,
                                                                         image_folder,
                                                                         confidence_threshold)
    if overwrite:
        s += ' --overwrite'
        
    print(s)
    import clipboard; clipboard.copy(s)


    #%% Opening labelme
    
    s = 'python labelme {}'.format(image_folder)
    print(s)
    import clipboard; clipboard.copy(s)
    

#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(
        description='Convert MD output to labelme annotation format')
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to MD results file (.json)')
    
    parser.add_argument(
        'image_base',
        type=str,
        help='Path to images (also the output folder)')
    
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=default_confidence_threshold,
        help='Confidence threshold (default {})'.format(default_confidence_threshold)
        )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing labelme .json files')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    md_to_labelme(args.results_file,args.image_base,args.confidence_threshold,args.overwrite)

if __name__ == '__main__':
    main()
