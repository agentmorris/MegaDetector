########
#
# md_to_labelme.py
#
# "Converts" a MegaDetector output .json file to labelme format (one .json per image
# file).  "Convert" is in quotes because this is an opinionated transformation that 
# requires a confidence threshold.
#
# TODO:
#    
# * support variable confidence thresholds across classes
# * support classification data
#
########

#%% Imports and constants

import os
import json

from tqdm import tqdm

from md_visualization.visualization_utils import open_image
from md_utils.ct_utils import truncate_float

output_precision = 3
default_confidence_threshold = 0.15


#%% Functions

def get_labelme_dict_for_image(im,image_base_name,category_id_to_name,info=None,confidence_threshold=None):
    """
    For the given image struct in MD results format, reformat the detections into
    labelme format.  Returns a dict.
    """
    
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


def md_to_labelme(results_file,image_base,confidence_threshold=None,
                  overwrite=False):
    """
    For all the images in [results_file], write a .json file in labelme format alongside the
    corresponding relative path within image_base.
    """
    
    # Load MD results
    with open(results_file,'r') as f:
        md_results = json.load(f)
        
    # Read image sizes
    #
    # TODO: parallelize this loop
    #
    # im = md_results['images'][0]
    for im in tqdm(md_results['images']):
        
        # Make sure this file exists
        im_full_path = os.path.join(image_base,im['file'])
        assert os.path.isfile(im_full_path), 'Image file {} does not exist'.format(im_full_path)
        
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
    
    # Write output
    for im in tqdm(md_results['images']):
        
        if 'failure' in im and im['failure'] is not None:
            assert 'detections' not in im
            print('Warning: skipping labelme file generation for failed image {}'.format(
                im['file']))
            continue
            
        im_full_path = os.path.join(image_base,im['file'])
        json_path = os.path.splitext(im_full_path)[0] + '.json'
        
        if (not overwrite) and (os.path.isfile(json_path)):
            print('Skipping existing file {}'.format(json_path))
            continue
    
        output_dict = get_labelme_dict_for_image(im,
                                                 image_base_name=os.path.basename(im_full_path),
                                                 category_id_to_name=md_results['detection_categories'],
                                                 info=md_results['info'],
                                                 confidence_threshold=confidence_threshold)
                
        with open(json_path,'w') as f:
            json.dump(output_dict,f,indent=1)
            
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
