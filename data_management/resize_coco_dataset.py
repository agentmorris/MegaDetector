########
#
# resize_coco_dataset.py
#
# Given a COCO-formatted dataset, resize all the images to a target size,
# scaling bounding boxes accordingly.
#
########

#%% Imports and constants

import os
import json
import shutil

from tqdm import tqdm
from collections import defaultdict

from md_utils.path_utils import insert_before_extension
from md_visualization.visualization_utils import \
    open_image, resize_image, exif_preserving_save


#%% Functions

def resize_coco_dataset(input_folder,input_filename,
                        output_folder,output_filename,
                        target_size=(-1,-1),
                        correct_size_image_handling='copy',
                        right_edge_quantization_threshold=None):
    """
    Given a COCO-formatted dataset (images in input_folder, data in input_filename), resize 
    all the images to a target size (in output_folder) and scale bounding boxes accordingly
    (in output_filename).
    
    target_size should be a tuple/list of ints, length 2.  If either dimension is -1, aspect ratio
    will be preserved.  If both dimensions are -1, this means "keep the original size".  If 
    both dimensions are -1 and correct_size_image_handling is copy, this function is basically 
    a no-op, although you might still use it for right_edge_quantization_threshold.
    
    correct_size_image_handling can be 'copy' (in which case the original image is just copied 
    to the output folder) or 'rewrite' (in which case the image is opened via PIL and re-written,
    attempting to preserve the same quality).  The only reason to do this is the case where 
    you're superstitious about biases coming from images in a training set being written
    by different image encoders.
   
    right_edge_quantization_threshold is an off-by-default hack to adjust large datasets where 
    boxes that really should be running off the right side of the image only extend like 99%
    of the way there, due to what appears to be a slight bias inherent to MD.  If a box extends
    within [right_edge_quantization_threshold] (a small number, from 0 to 1, but probably around 
    0.02) of the right edge of the image, it will be extended to the far right edge.
    """
    
    # Read input data
    with open(input_filename,'r') as f:
        d = json.load(f)
    
    # Map image IDs to annotations
    image_id_to_annotations = defaultdict(list)
    for ann in d['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
                                          
    # For each image
    
    # im = d['images'][1]
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
                
                # Do we need to quantize this box?
                if right_edge_quantization_threshold is not None and \
                    right_edge_quantization_threshold > 0:
                    bbox_right_edge_abs = bbox[0] + bbox[2]
                    bbox_right_edge_norm = bbox_right_edge_abs / output_w
                    bbox_right_edge_distance = (1.0 - bbox_right_edge_norm)
                    if bbox_right_edge_distance < right_edge_quantization_threshold:
                        bbox[2] = output_w - bbox[0]
                
                ann['bbox'] = bbox
            
            # ...if this annotation has a box
    
        # ...for each annotation
    
    # ...for each image
    
    # Write output file
    with open(output_filename,'w') as f:
        json.dump(d,f,indent=1)
    
# ...def resize_coco_dataset(...)
    

#%% Interactive driver

if False:
    
    pass

    #%% Test resizing
    
    # input_filename = os.path.expanduser('~/tmp/labelme_to_coco_test.json')
    # input_folder = os.path.expanduser('~/data/labelme-json-test')
    # target_size = (600,-1)
    
    input_folder = os.path.expanduser('~/data/usgs-kissel-training')
    input_filename = os.path.expanduser('~/data/usgs-tegus.json')
    target_size = (1600,-1)
    
    output_filename = insert_before_extension(input_filename,'resized')
    output_folder = input_folder + '-resized'    
    
    correct_size_image_handling = 'rewrite'
    
    right_edge_quantization_threshold = 0.015
    
    resize_coco_dataset(input_folder,input_filename,
                        output_folder,output_filename,
                        target_size=target_size,
                        correct_size_image_handling=correct_size_image_handling,
                        right_edge_quantization_threshold=right_edge_quantization_threshold)
    
    
    #%% Preview
    
    from md_visualization import visualize_db
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000

    html_file,_ = visualize_db.visualize_db(output_filename,
                                              os.path.expanduser('~/tmp/resize_coco_preview'),
                                              output_folder,options)
    

    from md_utils import path_utils # noqa
    path_utils.open_file(html_file)
    
    
#%% Command-line driver

# TODO

