"""

generate_crops_from_cct.py

Given a .json file in COCO Camera Traps format, creates a cropped image for
each bounding box.

"""

#%% Imports and constants

import os
import json

from tqdm import tqdm
from PIL import Image


#%% Functions

def generate_crops_from_cct(cct_file,image_dir,output_dir,padding=0,flat_output=True):
    """
    Given a .json file in COCO Camera Traps format, creates a cropped image for
    each bounding box.
    
    Args:
        cct_file (str): the COCO .json file from which we should load data
        image_dir (str): the folder where the images live; filenames in the .json
            file should be relative to this folder
        output_dir (str): the folder where we should write cropped images
        padding (float, optional): number of pixels we should expand each box before
            cropping
        flat_output (bool, optional): if False, folder structure will be preserved
            in the output, e.g. the image a/b/c/d.jpg will result in image files 
            in the output folder called, e.g., a/b/c/d_crop_000_id_12345.jpg.  If
            [flat_output] is True, the corresponding output image will be 
            a_b_c_d_crop_000_id_12345.jpg.            
    """
    
    ## Read and validate input
    
    assert os.path.isfile(cct_file)
    assert os.path.isdir(image_dir)
    os.makedirs(output_dir,exist_ok=True)

    with open(cct_file,'r') as f:
        d = json.load(f)
   
    
    ## Find annotations for each image
    
    from collections import defaultdict
    
    # This actually maps image IDs to annotations, but only to annotations
    # containing boxes
    image_id_to_boxes = defaultdict(list)
    
    n_boxes = 0
    
    for ann in d['annotations']:
        if 'bbox' in ann:
            image_id_to_boxes[ann['image_id']].append(ann)
            n_boxes += 1
            
    print('Found {} boxes in {} annotations for {} images'.format(
        n_boxes,len(d['annotations']),len(d['images'])))
    
    
    ## Generate crops
        
    # im = d['images'][0]
    for im in tqdm(d['images']):
        
        input_image_fn = os.path.join(os.path.join(image_dir,im['file_name']))
        assert os.path.isfile(input_image_fn), 'Could not find image {}'.format(input_image_fn)
        
        if im['id'] not in image_id_to_boxes:
            continue
        
        annotations_this_image = image_id_to_boxes[im['id']]
        
        # Load the image
        img = Image.open(input_image_fn)
        
        # Generate crops
        # i_ann = 0; ann = annotations_this_image[i_ann]
        for i_ann,ann in enumerate(annotations_this_image):
            
            # x/y/w/h, origin at the upper-left
            bbox = ann['bbox']
            
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            
            xmin -= padding / 2
            ymin -= padding / 2
            xmax += padding / 2
            ymax += padding / 2
            
            xmin = max(xmin,0)
            ymin = max(ymin,0)
            xmax = min(xmax,img.width-1)
            ymax = min(ymax,img.height-1)
                        
            crop = img.crop(box=[xmin, ymin, xmax, ymax])
            
            output_fn = os.path.splitext(im['file_name'])[0].replace('\\','/')
            if flat_output:
                output_fn = output_fn.replace('/','_')
            output_fn = output_fn + '_crop' + str(i_ann).zfill(3) + '_id_' + ann['id']
            output_fn = output_fn + '.jpg'
                
            output_full_path = os.path.join(output_dir,output_fn)
            
            if not flat_output:
                os.makedirs(os.path.dirname(output_full_path),exist_ok=True)
                
            crop.save(output_full_path)
            
        # ...for each box
        
    # ...for each image
    
# ...generate_crops_from_cct()


#%% Interactive driver

if False:
    
    pass

    #%%
    
    cct_file = os.path.expanduser('~/data/noaa/noaa_estuary_fish.json')
    image_dir = os.path.expanduser('~/data/noaa/JPEGImages')
    padding = 50
    flat_output = True
    output_dir = '/home/user/tmp/noaa-fish-crops'
    
    generate_crops_from_cct(cct_file,image_dir,output_dir,padding,flat_output=True)
    files = os.listdir(output_dir)
    
    
#%% Command-line driver

# TODO
