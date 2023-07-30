########
#
# run_tiled_inference.py
#
# Run inference on a folder, fist splitting each image up into tiles of size
# MxN (typically the native inference size of your detector), writing those
# tiles out to a temporary folder, then de-duplicating the results before merging
# them back into a set of detections that make sense on the original images.
#
# This approach will likely fail to detect very large animals, so if you expect both large 
# and small animals (in terms of pixel size), this script is best used in 
# conjunction with a traditional inference pass that looks at whole images.
#
# Currently requires temporary storage at least as large as the input data, generally
# a lot more than that (depending on the overlap between adjacent tiles).  This is 
# inefficient, but easy to debug.
#
########

#%% Imports and constants

import os
import json

from tqdm import tqdm

from detection.run_inference_with_yolov5_val import YoloInferenceOptions,run_inference_with_yolo_val
from detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file

import torch
from torchvision import ops

from md_utils import path_utils
from md_visualization import visualization_utils as vis_utils

default_patch_overlap = 0.5
patch_jpeg_quality = 95

# This isn't NMS in the usual sense of redundant model predictions; this is being
# used to de-duplicate predictions from overlapping patches.
nms_iou_threshold = 0.45


#%% Support functions

def get_patch_boundaries(image_size,patch_size,patch_stride=None):
    """    
    Get a list of patch starting coordinates (x,y) given an image size
    and a stride.  Stride defaults to half the patch size.
    
    image_size, patch_size, and patch_stride are all represented as [w,h].
    """
    
    if patch_stride is None:
        patch_stride = (round(patch_size[0]*(1.0-default_patch_overlap)),
                        round(patch_size[1]*(1.0-default_patch_overlap)))
        
    image_width = image_size[0]
    image_height = image_size[1]
        
    def add_patch_row(patch_start_positions,y_start):
        """
        Add one row to our list of patch start positions, i.e.
        loop over all columns.
        """
        
        x_start = 0; x_end = x_start + patch_size[0] - 1
        
        while(True):
            
            patch_start_positions.append([x_start,y_start])
            
            x_start += patch_stride[0]
            x_end = x_start + patch_size[0] - 1
             
            if x_end == image_width - 1:
                break
            elif x_end > (image_width - 1):
                overshoot = (x_end - image_width) + 1
                x_start -= overshoot
                x_end = x_start + patch_size[0] - 1
                patch_start_positions.append([x_start,y_start])
                break
        
        # ...for each column
        
        return patch_start_positions
        
    patch_start_positions = []
    
    y_start = 0; y_end = y_start + patch_size[1] - 1
        
    while(True):
    
        patch_start_positions = add_patch_row(patch_start_positions,y_start)
        
        y_start += patch_stride[1]
        y_end = y_start + patch_size[1] - 1
        
        if y_end == image_height - 1:
            break
        elif y_end > (image_height - 1):
            overshoot = (y_end - image_height) + 1
            y_start -= overshoot
            y_end = y_start + patch_size[1] - 1
            patch_start_positions = add_patch_row(patch_start_positions,y_start)
            break
    
    # ...for each row
    
    assert patch_start_positions[-1][0]+patch_size[0] == image_width
    assert patch_start_positions[-1][1]+patch_size[1] == image_height
    
    return patch_start_positions


def relative_path_to_image_name(rp):
    """
    Given a path name, replace slashes and backslashes with underscores, so we can
    use the result as a filename.
    """
    
    image_name = rp.lower().replace('\\','/').replace('/','_')
    return image_name


def patch_info_to_patch_name(image_name,patch_x_min,patch_y_min):
    
    patch_name = image_name + '_' + \
        str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
    return patch_name


def extract_patch_from_image(im,patch_xy,patch_size,
                             patch_image_fn=None,patch_folder=None,image_name=None,overwrite=True):
    """
    Extracts a patch from the provided image, writing the patch out to patch_image_fn.
    [im] can be a string or a PIL image.
    
    patch_xy is a length-2 tuple specifying the upper-left corner of the patch.
    
    image_name and patch_folder are only required if patch_image_fn is None.
    
    Returns a dictionary with fields xmin,xmax,ymin,ymax,patch_fn.
    """
    
    if isinstance(im,str):
        pil_im = vis_utils.open_image(im)
    else:
        pil_im = im
        
    patch_x_min = patch_xy[0]
    patch_y_min = patch_xy[1]
    patch_x_max = patch_x_min + patch_size[0] - 1
    patch_y_max = patch_y_min + patch_size[1] - 1

    # PIL represents coordinates in a way that is very hard for me to get my head
    # around, such that even though the "right" and "bottom" arguments to the crop()
    # function are inclusive... well, they're not really.
    #
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
    #
    # So we add 1 to the max values.
    patch_im = pil_im.crop((patch_x_min,patch_y_min,patch_x_max+1,patch_y_max+1))
    assert patch_im.size[0] == patch_size[0]
    assert patch_im.size[1] == patch_size[1]

    if patch_image_fn is None:
        assert patch_folder is not None,\
            "If you don't supply a patch filename to extract_patch_from_image, you need to supply a folder name"
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        patch_image_fn = os.path.join(patch_folder,patch_name + '.jpg')
    
    if os.path.isfile(patch_image_fn) and (not overwrite):
        pass
    else:        
        patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
    
    patch_info = {}
    patch_info['xmin'] = patch_x_min
    patch_info['xmax'] = patch_x_max
    patch_info['ymin'] = patch_y_min
    patch_info['ymax'] = patch_y_max
    patch_info['patch_fn'] = patch_image_fn
    
    return patch_info


def in_place_nms(md_results, iou_thres=0.45, verbose=True):
    """
    Run torch.ops.nms in-place on MD-formatted detection results    
    """
    
    n_detections_before = 0
    n_detections_after = 0
    
    # i_image = 18; im = md_results['images'][i_image]
    for i_image,im in tqdm(enumerate(md_results['images']),total=len(md_results['images'])):
        
        if len(im['detections']) == 0:
            continue
    
        boxes = []
        scores = []
        
        n_detections_before += len(im['detections'])
        
        # det = im['detections'][0]
        for det in im['detections']:
            
            # Using x1/x2 notation rather than x0/x1 notation to be consistent
            # with the Torch documentation.
            x1 = det['bbox'][0]
            y1 = det['bbox'][1]
            x2 = det['bbox'][0] + det['bbox'][2]
            y2 = det['bbox'][1] + det['bbox'][3]
            box = [x1,y1,x2,y2]
            boxes.append(box)
            scores.append(det['conf'])

        # ...for each detection
        
        t_boxes = torch.tensor(boxes)
        t_scores = torch.tensor(scores)
        
        box_indices = ops.nms(t_boxes,t_scores,iou_thres).tolist()
        
        post_nms_detections = [im['detections'][x] for x in box_indices]
        
        assert len(post_nms_detections) <= len(im['detections'])
        
        im['detections'] = post_nms_detections
        
        n_detections_after += len(im['detections'])
        
    # ...for each image
    
    if verbose:
        print('NMS removed {} of {} detections'.format(
            n_detections_before-n_detections_after,
            n_detections_before))
        
# ...in_place_nms()


#%% Main function
    
def run_tiled_inference(model_file, image_folder, tiling_folder, output_file,
                        tile_size_x=1280, tile_size_y=1280, tile_overlap=0.5,
                        checkpoint_path=None, checkpoint_frequency=-1, remove_tiles=False, 
                        yolo_inference_options=None):
    """
    Run inference using [model_file] on the images in [image_folder], fist splitting each image up 
    into tiles of size [tile_size_x] x [tile_size_y], writing those tiles to [tiling_folder],
    then de-duplicating the results before merging them back into a set of detections that make 
    sense on the original images and writing those results to [output_file].  
    
    [tiling_folder] can be any folder, but this function reserves the right to do whatever it wants
    within that folder, including deleting everything, so it's best if it's a new folder.  
    Conceptually this folder is temporary, it's just helpful in this case to not actually
    use the system temp folder, because the tile cache may be very large, 
    
    tile_overlap is the fraction of overlap between tiles.
    
    Optionally removes the temporary tiles.
    
    if yolo_inference_options is supplied, it should be an instance of YoloInferenceOptions; in 
    this case the model will be run with run_inference_with_yolov5_val.  This is typically used to 
    run the model with test-time augmentation.          
    """    

    ##%% Validate arguments
    
    assert tile_overlap < 1 and tile_overlap > 0, \
        'Illegal tile overlap value {}'.format(tile_overlap)
    
    patch_size = [tile_size_x,tile_size_y]
    patch_stride = (round(patch_size[0]*(1.0-tile_overlap)),
                    round(patch_size[1]*(1.0-tile_overlap)))
    
    os.makedirs(tiling_folder,exist_ok=True)
    
    
    ##%% List files
    
    image_files_relative = path_utils.recursive_file_list(image_folder, return_relative_paths=True)
    
    
    ##%% Generate tiles
    
    all_image_patch_info = []
    
    # For each image
    #
    # fn_relative = image_files_relative[0]
    #
    # TODO: parallelize this loop
    for fn_relative in tqdm(image_files_relative):
        
        fn_abs = os.path.join(image_folder,fn_relative)
        
        image_name = relative_path_to_image_name(fn_relative)
        
        # Open the image
        im = vis_utils.open_image(fn_abs)
        image_size = [im.width,im.height]
                
        # Generate patch boundaries (a list of [x,y] starting points)
        patch_boundaries = get_patch_boundaries(image_size,patch_size,patch_stride)        
        
        # Extract patches
        #
        # patch_xy = patch_boundaries[0]
        patches = []
        
        for patch_xy in patch_boundaries:
            
            patch_info = extract_patch_from_image(im,patch_xy,patch_size,
                                     patch_folder=tiling_folder,
                                     image_name=image_name,
                                     overwrite=True)
            patch_info['source_fn'] = fn_relative
            patches.append(patch_info)
            
        image_patch_info = {}
        image_patch_info['patches'] = patches
        image_patch_info['image_fn'] = fn_relative
        
        all_image_patch_info.append(image_patch_info)
        
    # ...for each image
        
    # Write tile information to file; this is just a debugging convenience
    folder_name = relative_path_to_image_name(image_folder)
    if folder_name.startswith('_'):
        folder_name = folder_name[1:]
        
    tile_cache_file = os.path.join(tiling_folder,folder_name + '_patch_info.json')
    with open(tile_cache_file,'w') as f:
        json.dump(all_image_patch_info,f,indent=1)
    
    
    ##%% Run inference on tiles
    
    if yolo_inference_options is not None:
        
        patch_level_output_file = os.path.join(tiling_folder,folder_name + '_patch_level_results.json')
        
        if yolo_inference_options.model_filename is None:
            yolo_inference_options.model_filename = model_file
        else:
            assert yolo_inference_options.model_filename == model_file, \
                'Model file between yolo inference file ({}) and model file parameter ({})'.format(
                    yolo_inference_options.model_filename,model_file)
        
        yolo_inference_options.input_folder = tiling_folder
        yolo_inference_options.output_file = patch_level_output_file
        
        run_inference_with_yolo_val(yolo_inference_options)
        with open(patch_level_output_file,'r') as f:
            patch_level_results = json.load(f)
                    
    else:
        
        patch_file_names = []
        for im in all_image_patch_info:
            for patch in im['patches']:
                patch_file_names.append(patch['patch_fn'])
                
        inference_results = load_and_run_detector_batch(model_file, 
                                                        patch_file_names, 
                                                        checkpoint_path=checkpoint_path,
                                                        checkpoint_frequency=checkpoint_frequency)
        
        patch_level_output_file = os.path.join(tiling_folder,folder_name + '_patch_level_results.json')
        
        patch_level_results = write_results_to_file(inference_results, 
                                                    patch_level_output_file, 
                                                    relative_path_base=tiling_folder, 
                                                    detector_file=model_file)
        
    
    ##%% Map patch-level detections back to the original images    
    
    # Map relative paths for patches to detections
    patch_fn_relative_to_results = {}
    for im in tqdm(patch_level_results['images']):
        patch_fn_relative_to_results[im['file']] = im

    image_level_results = {}
    image_level_results['info'] = patch_level_results['info']
    image_level_results['detection_categories'] = patch_level_results['detection_categories']
    image_level_results['images'] = []
    
    image_fn_relative_to_patch_info = { x['image_fn']:x for x in all_image_patch_info }
    
    # i_image = 0; image_fn_relative = image_files_relative[i_image]
    for i_image,image_fn_relative in tqdm(enumerate(image_files_relative),total=len(image_files_relative)):
        
        image_fn_abs = os.path.join(image_folder,image_fn_relative)
        assert os.path.isfile(image_fn_abs)
                
        output_im = {}
        output_im['file'] = image_fn_relative
        output_im['detections'] = []
            
        pil_im = vis_utils.open_image(image_fn_abs)        
        image_w = pil_im.size[0]
        image_h = pil_im.size[1]
        
        image_patch_info = image_fn_relative_to_patch_info[image_fn_relative]
        assert image_patch_info['patches'][0]['source_fn'] == image_fn_relative
        
        # Patches for this image
        patch_fn_abs_to_patch_info_this_image = {}
        
        for patch_info in image_patch_info['patches']:
            patch_fn_abs_to_patch_info_this_image[patch_info['patch_fn']] = patch_info
                
        # For each patch
        #
        # i_patch = 0; patch_fn_abs = list(patch_fn_abs_to_patch_info_this_image.keys())[i_patch]
        for i_patch,patch_fn_abs in enumerate(patch_fn_abs_to_patch_info_this_image.keys()):
            
            patch_fn_relative = os.path.relpath(patch_fn_abs,tiling_folder)
            patch_results = patch_fn_relative_to_results[patch_fn_relative]
            patch_info = patch_fn_abs_to_patch_info_this_image[patch_fn_abs]
            
            # patch_results['file'] is a relative path, and a subset of patch_info['patch_fn']
            assert patch_results['file'] in patch_info['patch_fn']
            
            patch_w = (patch_info['xmax'] - patch_info['xmin']) + 1
            patch_h = (patch_info['ymax'] - patch_info['ymin']) + 1
            assert patch_w == patch_size[0]
            assert patch_h == patch_size[1]
            
            # det = patch_results['detections'][0]
            for det in patch_results['detections']:
            
                bbox_patch_relative = det['bbox']
                xmin_patch_relative = bbox_patch_relative[0]
                ymin_patch_relative = bbox_patch_relative[1]
                w_patch_relative = bbox_patch_relative[2]
                h_patch_relative = bbox_patch_relative[3]
                
                # Convert from patch-relative normalized values to image-relative absolute values
                w_pixels = w_patch_relative * patch_w
                h_pixels = h_patch_relative * patch_h
                xmin_patch_pixels = xmin_patch_relative * patch_w
                ymin_patch_pixels = ymin_patch_relative * patch_h
                xmin_image_pixels = patch_info['xmin'] + xmin_patch_pixels
                ymin_image_pixels = patch_info['ymin'] + ymin_patch_pixels
                
                # ...and now to image-relative normalized values
                w_image_normalized = w_pixels / image_w
                h_image_normalized = h_pixels / image_h
                xmin_image_normalized = xmin_image_pixels / image_w
                ymin_image_normalized = ymin_image_pixels / image_h
                
                bbox_image_normalized = [xmin_image_normalized,
                                         ymin_image_normalized,
                                         w_image_normalized,
                                         h_image_normalized]
                
                output_det = {}
                output_det['bbox'] = bbox_image_normalized
                output_det['conf'] = det['conf']
                output_det['category'] = det['category']
                
                output_im['detections'].append(output_det)
                
            # ...for each detection
            
        # ...for each patch

        image_level_results['images'].append(output_im)
        
    # ...for each image    

    image_level_results_file_pre_nms = \
        os.path.join(tiling_folder,folder_name + '_image_level_results_pre_nms.json')
    with open(image_level_results_file_pre_nms,'w') as f:
        json.dump(image_level_results,f,indent=1)
        

    ##%% Run NMS
    
    in_place_nms(image_level_results,iou_thres=nms_iou_threshold)

    
    ##%% Write output file
    
    print('Saving image-level results (after NMS) to {}'.format(output_file))
    
    with open(output_file,'w') as f:
        json.dump(image_level_results,f,indent=1)

        
    ##%% Possibly remove tiles
        
    if remove_tiles:
        
        patch_file_names = []
        for im in all_image_patch_info:
            for patch in im['patches']:
                patch_file_names.append(patch['patch_fn'])
                
        for patch_fn_abs in patch_file_names:
            os.remove(patch_fn_abs)
        
    
    ##%% Return
    
    return image_level_results


#%% Interactive driver

if False:
    
    pass

    #%% Run tiled inference
    
    model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    image_folder = os.path.expanduser('~/data/KRU-test')
    tiling_folder = os.path.expanduser('~/tmp/tiling-test')
    output_file = os.path.expanduser('~/tmp/KRU-test-tiled.json')

    tile_size_x=1280
    tile_size_y=1280
    tile_overlap=0.5
    checkpoint_path=None
    checkpoint_frequency=-1
    remove_tiles=False
    
    if False:
        
        yolo_inference_options = None
        
    else:
        
        yolo_inference_options = YoloInferenceOptions()
        yolo_inference_options.yolo_working_folder = os.path.expanduser('~/git/yolov5')
            
            
    #%%
    
    run_tiled_inference(model_file, image_folder, tiling_folder, output_file,
                            tile_size_x=tile_size_x, tile_size_y=tile_size_y, 
                            tile_overlap=tile_overlap,
                            checkpoint_path=checkpoint_path, 
                            checkpoint_frequency=checkpoint_frequency, 
                            remove_tiles=remove_tiles, 
                            yolo_inference_options=yolo_inference_options)
    
    
    #%% Preview tiled inference
    
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)

    options = PostProcessingOptions()
    options.image_base_dir = image_folder
    options.include_almost_detections = True
    options.num_images_to_sample = None
    options.confidence_threshold = 0.2
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    # options.sample_seed = 0

    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = 10
    options.parallelize_rendering_with_threads = False

    preview_base = os.path.join(tiling_folder,'preview')
    os.makedirs(preview_base, exist_ok=True)

    print('Processing post-RDE to {}'.format(preview_base))

    options.api_output_file = output_file
    options.output_dir = preview_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file

    path_utils.open_file(html_output_file)
    
    
#%% Command-line driver