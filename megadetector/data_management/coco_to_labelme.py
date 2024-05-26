"""

coco_to_labelme.py

Converts a COCO dataset to labelme format (one .json per image file).

If you want to convert YOLO-formatted data to labelme format, use yolo_to_coco, then 
coco_to_labelme.

"""

#%% Imports and constants

import os
import json

from tqdm import tqdm
from collections import defaultdict

from megadetector.visualization.visualization_utils import open_image


#%% Functions

def get_labelme_dict_for_image_from_coco_record(im,annotations,categories,info=None):
    """
    For the given image struct in COCO format and associated list of annotations, reformats the 
    detections into labelme format.  
    
    Args:
        im (dict): image dict, as loaded from a COCO .json file; 'height' and 'width' are required
        annotations (list): a list of annotations that refer to this image (this function errors if 
            that's not the case)
        categories (list): a list of category in dicts in COCO format ({'id':x,'name':'s'})
        info (dict, optional): a dict to store in a non-standard "custom_info"  field in the output
        
    Returns:
        dict: a dict in labelme format, suitable for writing to a labelme .json file
    """
    
    image_base_name = os.path.basename(im['file_name'])
    
    output_dict = {}
    if info is not None:
        output_dict['custom_info'] = info
    output_dict['version'] = '5.3.0a0'
    output_dict['flags'] = {}
    output_dict['shapes'] = []
    output_dict['imagePath'] = image_base_name
    output_dict['imageHeight'] = im['height']
    output_dict['imageWidth'] = im['width']
    output_dict['imageData'] = None
    
    # Store COCO categories in case we want to reconstruct the original IDs later
    output_dict['coco_categories'] = categories
    
    category_id_to_name = {c['id']:c['name'] for c in categories}
    
    if 'flags' in im:
        output_dict['flags'] = im['flags']
        
    # ann = annotations[0]
    for ann in annotations:
        
        assert ann['image_id'] == im['id'], 'Annotation {} does not refer to image {}'.format(
            ann['id'],im['id'])
        
        if 'bbox' not in ann:
            continue
        
        shape = {}
        shape['label'] = category_id_to_name[ann['category_id']] 
        shape['shape_type'] = 'rectangle'
        shape['description'] = ''
        shape['group_id'] = None
        
        # COCO boxes are [x_min, y_min, width_of_box, height_of_box] (absolute)
        # 
        # labelme boxes are [[x0,y0],[x1,y1]] (absolute)
        x0 = ann['bbox'][0]
        y0 = ann['bbox'][1]
        x1 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1] + ann['bbox'][3]
        
        shape['points'] = [[x0,y0],[x1,y1]]
        output_dict['shapes'].append(shape)
    
    # ...for each detection
    
    return output_dict

# ...def get_labelme_dict_for_image()


def coco_to_labelme(coco_data,image_base,overwrite=False,bypass_image_size_check=False,verbose=False):
    """
    For all the images in [coco_data] (a dict or a filename), write a .json file in 
    labelme format alongside the corresponding relative path within image_base.
    
    Args:
        coco_data (dict or str): path to a COCO-formatted .json file, or an already-loaded
            COCO-formatted dict
        image_base (str): path where images live (filenames in [coco_data] should be relative to
            [image_base]); this is also where labelme files will be written
        overwrite (bool, optional): overwrite existing .json files
        bypass_image_size_check (bool, optional): if you're sure that the COCO data already has
            correct 'width' and 'height' fields, this bypasses the somewhat-slow loading of 
            each image to fetch image sizes
        verbose (bool, optional): enable additional debug output
    """
    
    # Load COCO data if necessary
    if isinstance(coco_data,str):
        with open(coco_data,'r') as f:
            coco_data = json.load(f)
    assert isinstance(coco_data,dict)
     
    
    ## Read image sizes if necessary
    
    if bypass_image_size_check:
        
        print('Bypassing size check')
        
    else:
    
        # TODO: parallelize this loop
        
        print('Reading/validating image sizes...')
        
        # im = coco_data['images'][0]
        for im in tqdm(coco_data['images']):
            
            # Make sure this file exists
            im_full_path = os.path.join(image_base,im['file_name'])
            assert os.path.isfile(im_full_path), 'Image file {} does not exist'.format(im_full_path)
            
            # Load w/h information if necessary
            if 'height' not in im or 'width' not in im:
                
                try:
                    pil_im = open_image(im_full_path)
                    im['width'] = pil_im.width
                    im['height'] = pil_im.height
                except Exception:
                    print('Warning: cannot open image {}'.format(im_full_path))
                    if 'failure' not in im:
                        im['failure'] = 'Failure image access'
    
            # ...if we need to read w/h information
            
        # ...for each image
    
    # ...if we need to load image sizes
    
    
    ## Generate labelme files
    
    print('Generating .json files...')
    
    image_id_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
        
    n_json_files_written = 0
    n_json_files_error = 0
    n_json_files_exist = 0
    
    # Write output
    for im in tqdm(coco_data['images']):
        
        # Skip this image if it failed to load in whatever system generated this COCO file
        skip_image = False
        
        # Errors are represented differently depending on the source
        for error_string in ('failure','error'):
            if (error_string in im) and (im[error_string] is not None):
                if verbose:
                    print('Warning: skipping labelme file generation for failed image {}'.format(
                        im['file_name']))
                skip_image = True
                n_json_files_error += 1
                break
        if skip_image:
            continue
                    
        im_full_path = os.path.join(image_base,im['file_name'])
        json_path = os.path.splitext(im_full_path)[0] + '.json'
        
        if (not overwrite) and (os.path.isfile(json_path)):
            if verbose:
                print('Skipping existing file {}'.format(json_path))
            n_json_files_exist += 1
            continue
    
        annotations_this_image = image_id_to_annotations[im['id']]
        output_dict = get_labelme_dict_for_image_from_coco_record(im,
                                                                  annotations_this_image,
                                                                  coco_data['categories'],
                                                                  info=None)
        
        n_json_files_written += 1
        with open(json_path,'w') as f:
            json.dump(output_dict,f,indent=1)
            
    # ...for each image
    
    print('\nWrote {} .json files (skipped {} for errors, {} because they exist)'.format(
        n_json_files_written,n_json_files_error,n_json_files_exist))
    
# ...def coco_to_labelme()


#%% Interactive driver

if False:
    
    pass

    #%% Configure options
    
    coco_file = \
        r'C:\\temp\\snapshot-exploration\\images\\training-images-good\\training-images-good_from_yolo.json'
    image_folder = os.path.dirname(coco_file)    
    overwrite = True    
    
    
    #%% Programmatic execution
    
    coco_to_labelme(coco_data=coco_file,image_base=image_folder,overwrite=overwrite)

    
    #%% Command-line execution
    
    s = 'python coco_to_labelme.py "{}" "{}"'.format(coco_file,image_folder)
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
        description='Convert a COCO database to labelme annotation format')
    
    parser.add_argument(
        'coco_file',
        type=str,
        help='Path to COCO data file (.json)')
    
    parser.add_argument(
        'image_base',
        type=str,
        help='Path to images (also the output folder)')
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing labelme .json files')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    coco_to_labelme(coco_data=args.coco_file,image_base=args.image_base,overwrite=args.overwrite)    

if __name__ == '__main__':
    main()
