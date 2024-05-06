"""

resize_coco_dataset.py

Given a COCO-formatted dataset, resizes all the images to a target size,
scaling bounding boxes accordingly.

"""

#%% Imports and constants

import os
import json
import shutil

from tqdm import tqdm
from collections import defaultdict

from megadetector.utils.path_utils import insert_before_extension
from megadetector.visualization.visualization_utils import \
    open_image, resize_image, exif_preserving_save


#%% Functions

def resize_coco_dataset(input_folder,input_filename,
                        output_folder,output_filename,
                        target_size=(-1,-1),
                        correct_size_image_handling='copy'):
    """
    Given a COCO-formatted dataset (images in input_folder, data in input_filename), resizes
    all the images to a target size (in output_folder) and scales bounding boxes accordingly.
    
    Args:
        input_folder (str): the folder where images live; filenames in [input_filename] should 
            be relative to [input_folder]
        input_filename (str): the (input) COCO-formatted .json file containing annotations
        output_folder (str): the folder to which we should write resized images; can be the
            same as [input_folder], in which case images are over-written
        output_filename (str): the COCO-formatted .json file we should generate that refers to
            the resized images
        target_size (list or tuple of ints): this should be tuple/list of ints, with length 2 (w,h).
            If either dimension is -1, aspect ratio will be preserved.  If both dimensions are -1, this means 
            "keep the original size".  If  both dimensions are -1 and correct_size_image_handling is copy, this 
            function is basically a no-op.    
        correct_size_image_handling (str): can be 'copy' (in which case the original image is just copied 
            to the output folder) or 'rewrite' (in which case the image is opened via PIL and re-written,
            attempting to preserve the same quality).  The only reason to do use 'rewrite' 'is the case where 
            you're superstitious about biases coming from images in a training set being written by different 
            image encoders.
   
    Returns:
        dict: the COCO database with resized images, identical to the content of [output_filename]
    """
    
    # Read input data
    with open(input_filename,'r') as f:
        d = json.load(f)
    
    # Map image IDs to annotations
    image_id_to_annotations = defaultdict(list)
    for ann in d['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
                                          
    # For each image
    
    # TODO: this is trivially parallelizable
    #
    # im = d['images'][0]
    for im in tqdm(d['images']):
    
        input_fn_relative = im['file_name']
        input_fn_abs = os.path.join(input_folder,input_fn_relative)
        assert os.path.isfile(input_fn_abs), "Can't find image file {}".format(input_fn_abs)
        
        output_fn_abs = os.path.join(output_folder,input_fn_relative)
        os.makedirs(os.path.dirname(output_fn_abs),exist_ok=True)
        
        pil_im = open_image(input_fn_abs)
        input_w = pil_im.width
        input_h = pil_im.height
        
        image_is_already_target_size = \
            (input_w == target_size[0]) and (input_h == target_size[1])
        preserve_original_size = \
            (target_size[0] == -1) and (target_size[1] == -1)
            
        # If the image is already the right size...
        if (image_is_already_target_size or preserve_original_size):
            output_w = input_w
            output_h = input_h
            if correct_size_image_handling == 'copy':
                shutil.copyfile(input_fn_abs,output_fn_abs)
            elif correct_size_image_handling == 'rewrite':
                exif_preserving_save(pil_im,output_fn_abs)
            else:
                raise ValueError('Unrecognized value {} for correct_size_image_handling'.format(
                    correct_size_image_handling))
        else:
            pil_im = resize_image(pil_im, target_size[0], target_size[1])
            output_w = pil_im.width
            output_h = pil_im.height
            exif_preserving_save(pil_im,output_fn_abs)

        im['width'] = output_w
        im['height'] = output_h
        
        # For each box
        annotations_this_image = image_id_to_annotations[im['id']]
        
        # ann = annotations_this_image[0]
        for ann in annotations_this_image:
            
            if 'bbox' in ann:
        
                # boxes are [x,y,w,h]
                bbox = ann['bbox']
                
                # Do we need to scale this box?
                if (output_w != input_w) or (output_h != input_h):
                    width_scale = output_w/input_w
                    height_scale = output_h/input_h
                    bbox = \
                           [bbox[0] * width_scale,
                            bbox[1] * height_scale,
                            bbox[2] * width_scale,
                            bbox[3] * height_scale]
                
                ann['bbox'] = bbox
            
            # ...if this annotation has a box
    
        # ...for each annotation
    
    # ...for each image
    
    # Write output file
    with open(output_filename,'w') as f:
        json.dump(d,f,indent=1)
    
    return d

# ...def resize_coco_dataset(...)
    

#%% Interactive driver

if False:
    
    pass

    #%% Test resizing
        
    input_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
    input_filename = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training.json')
    target_size = (1600,-1)
    
    output_filename = insert_before_extension(input_filename,'resized-test')
    output_folder = input_folder + '-resized-test'
    
    correct_size_image_handling = 'rewrite'
    
    resize_coco_dataset(input_folder,input_filename,
                        output_folder,output_filename,
                        target_size=target_size,
                        correct_size_image_handling=correct_size_image_handling)
    
    
    #%% Preview
    
    from megadetector.visualization import visualize_db
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000

    html_file,_ = visualize_db.visualize_db(output_filename,
                                              os.path.expanduser('~/tmp/resize_coco_preview'),
                                              output_folder,options)
    

    from megadetector.utils import path_utils # noqa
    path_utils.open_file(html_file)
    
    
#%% Command-line driver

# TODO

