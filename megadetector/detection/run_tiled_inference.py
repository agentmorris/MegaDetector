"""

run_tiled_inference.py

**This script is experimental, YMMV.**

Runs inference on a folder, fist splitting each image up into tiles of size
MxN (typically the native inference size of your detector), writing those
tiles out to a temporary folder, then de-duplicating the resulting detections before 
merging them back into a set of detections that make sense on the original images.

This approach will likely fail to detect very large animals, so if you expect both large 
and small animals (in terms of pixel size), this script is best used in 
conjunction with a traditional inference pass that looks at whole images.

Currently requires temporary storage at least as large as the input data, generally
a lot more than that (depending on the overlap between adjacent tiles).  This is 
inefficient, but easy to debug.

Programmatic invocation supports using YOLOv5's inference scripts (and test-time
augmentation); the command-line interface only supports standard inference right now.

"""

#%% Imports and constants

import os
import json

from tqdm import tqdm

import torch
from torchvision import ops

from megadetector.detection.run_inference_with_yolov5_val import YoloInferenceOptions,run_inference_with_yolo_val
from megadetector.detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file
from megadetector.detection.run_detector import try_download_known_detector
from megadetector.utils import path_utils
from megadetector.visualization import visualization_utils as vis_utils

default_patch_overlap = 0.5
patch_jpeg_quality = 95

# This isn't NMS in the usual sense of redundant model predictions; this is being
# used to de-duplicate predictions from overlapping patches.
nms_iou_threshold = 0.45

default_tile_size = [1280,1280]

default_n_patch_extraction_workers = 1
parallelization_uses_threads = False


#%% Support functions

def get_patch_boundaries(image_size,patch_size,patch_stride=None):
    """
    Computes a list of patch starting coordinates (x,y) given an image size (w,h)
    and a stride (x,y)
    
    Patch size is guaranteed, but the stride may deviate to make sure all pixels are covered.
    I.e., we move by regular strides until the current patch walks off the right/bottom,
    at which point it backs up to one patch from the end.  So if your image is 15
    pixels wide and you have a stride of 10 pixels, you will get starting positions 
    of 0 (from 0 to 9) and 5 (from 5 to 14).
    
    Args:
        image_size (tuple): size of the image you want to divide into patches, as a length-2 tuple (w,h)
        patch_size (tuple): patch size into which you want to divide an image, as a length-2 tuple (w,h)
        patch_stride (tuple or float, optional): stride between patches, as a length-2 tuple (x,y), or a 
            float; if this is a float, it's interpreted as the stride relative to the patch size 
            (0.1 == 10% stride).  Defaults to half the patch size.

    Returns:
        list: list of length-2 tuples, each representing the x/y start position of a patch        
    """
    
    if patch_stride is None:
        patch_stride = (round(patch_size[0]*(1.0-default_patch_overlap)),
                        round(patch_size[1]*(1.0-default_patch_overlap)))
    elif isinstance(patch_stride,float):
        patch_stride = (round(patch_size[0]*(patch_stride)),
                        round(patch_size[1]*(patch_stride)))
        
    image_width = image_size[0]
    image_height = image_size[1]
    
    assert patch_size[0] <= image_size[0], 'Patch width {} is larger than image width {}'.format(
        patch_size[0],image_size[0])
    assert patch_size[1] <= image_size[1], 'Patch height {} is larger than image height {}'.format(
        patch_size[1],image_size[1])
    
    def add_patch_row(patch_start_positions,y_start):
        """
        Add one row to our list of patch start positions, i.e.
        loop over all columns.
        """
        
        x_start = 0; x_end = x_start + patch_size[0] - 1
        
        while(True):
            
            patch_start_positions.append([x_start,y_start])
            
            # If this patch put us right at the end of the last column, we're done
            if x_end == image_width - 1:
                break
            
            # Move one patch to the right
            x_start += patch_stride[0]
            x_end = x_start + patch_size[0] - 1
             
            # If this patch flows over the edge, add one more patch to cover
            # the pixels on the end, then we're done.
            if x_end > (image_width - 1):
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
        
        # If this patch put us right at the bottom of the lats row, we're done
        if y_end == image_height - 1:
            break
        
        # Move one patch down
        y_start += patch_stride[1]
        y_end = y_start + patch_size[1] - 1
        
        # If this patch flows over the bottom, add one more patch to cover
        # the pixels at the bottom, then we're done
        if y_end > (image_height - 1):
            overshoot = (y_end - image_height) + 1
            y_start -= overshoot
            y_end = y_start + patch_size[1] - 1
            patch_start_positions = add_patch_row(patch_start_positions,y_start)
            break
    
    # ...for each row
    
    for p in patch_start_positions:
        assert p[0] >= 0 and p[1] >= 0 and p[0] <= image_width and p[1] <= image_height, \
        'Patch generation error (illegal patch {})'.format(p)
        
    # The last patch should always end at the bottom-right of the image
    assert patch_start_positions[-1][0]+patch_size[0] == image_width, \
        'Patch generation error (last patch does not end on the right)'
    assert patch_start_positions[-1][1]+patch_size[1] == image_height, \
        'Patch generation error (last patch does not end at the bottom)'
    
    # All patches should be unique
    patch_start_positions_tuples = [tuple(x) for x in patch_start_positions]
    assert len(patch_start_positions_tuples) == len(set(patch_start_positions_tuples)), \
        'Patch generation error (duplicate start position)'
    
    return patch_start_positions

# ...get_patch_boundaries()


def patch_info_to_patch_name(image_name,patch_x_min,patch_y_min):
    """
    Gives a unique string name to an x/y coordinate, e.g. turns ("a.jpg",10,20) into
    "a.jpg_0010_0020".
    
    Args:
        image_name (str): image identifier
        patch_x_min (int): x coordinate
        patch_y_min (int): y coordinate
    
    Returns:
        str: name for this patch, e.g. "a.jpg_0010_0020"
    """
    patch_name = image_name + '_' + \
        str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
    return patch_name


def extract_patch_from_image(im,
                             patch_xy,
                             patch_size,
                             patch_image_fn=None,
                             patch_folder=None,
                             image_name=None,
                             overwrite=True):
    """
    Extracts a patch from the provided image, and writes that patch out to a new file.
    
    Args:
        im (str or Image): image from which we should extract a patch, can be a filename or
            a PIL Image object.
        patch_xy (tuple): length-2 tuple of ints (x,y) representing the upper-left corner 
            of the patch to extract
        patch_size (tuple): length-2 tuple of ints (w,h) representing the size of the 
            patch to extract
        patch_image_fn (str, optional): image filename to write the patch to; if this is None
            the filename will be generated from [image_name] and the patch coordinates
        patch_folder (str, optional): folder in which the image lives; only used to generate
            a patch filename, so only required if [patch_image_fn] is None
        image_name (str, optional): the identifier of the source image; only used to generate
            a patch filename, so only required if [patch_image_fn] is None
        overwrite (bool, optional): whether to overwrite an existing patch image
    
    Returns:
        dict: a dictionary with fields xmin,xmax,ymin,ymax,patch_fn
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

# ...def extract_patch_from_image(...)


def in_place_nms(md_results, iou_thres=0.45, verbose=True):
    """
    Run torch.ops.nms in-place on MD-formatted detection results.
    
    Args:
        md_results (dict): detection results for a list of images, in MD results format (i.e., 
            containing a list of image dicts with the key 'images', each of which has a list
            of detections with the key 'detections')
        iou_thres (float, optional): IoU threshold above which we will treat two detections as
            redundant
        verbose (bool, optional): enable additional debug console output
    """
    
    n_detections_before = 0
    n_detections_after = 0
    
    # i_image = 18; im = md_results['images'][i_image]
    for i_image,im in tqdm(enumerate(md_results['images']),total=len(md_results['images'])):
        
        if (im['detections'] is None) or (len(im['detections']) == 0):
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


def _extract_tiles_for_image(fn_relative,image_folder,tiling_folder,patch_size,patch_stride,overwrite):
    """
    Private function to extract tiles for a single image.
    
    Returns a dict with fields 'patches' (see extract_patch_from_image) and 'image_fn'.
    
    If there is an error, 'patches' will be None and the 'error' field will contain
    failure details.  In that case, some tiles may still be generated.
    """
    
    fn_abs = os.path.join(image_folder,fn_relative)
    error = None
    patches = []        
    
    image_name = path_utils.clean_filename(fn_relative,char_limit=None,force_lower=True)
    
    try:
        
        # Open the image
        im = vis_utils.open_image(fn_abs)
        image_size = [im.width,im.height]
                
        # Generate patch boundaries (a list of [x,y] starting points)
        patch_boundaries = get_patch_boundaries(image_size,patch_size,patch_stride)        
        
        # Extract patches
        #
        # patch_xy = patch_boundaries[0]        
        for patch_xy in patch_boundaries:
            
            patch_info = extract_patch_from_image(im,patch_xy,patch_size,
                                     patch_folder=tiling_folder,
                                     image_name=image_name,
                                     overwrite=overwrite)
            patch_info['source_fn'] = fn_relative
            patches.append(patch_info)
        
    except Exception as e:
        
        s = 'Patch generation error for {}: \n{}'.format(fn_relative,str(e))
        print(s)
        # patches = None
        error = s
        
    image_patch_info = {}
    image_patch_info['patches'] = patches
    image_patch_info['image_fn'] = fn_relative
    image_patch_info['error'] = error
    
    return image_patch_info
    
    
#%% Main function
    
def run_tiled_inference(model_file, image_folder, tiling_folder, output_file,
                        tile_size_x=1280, tile_size_y=1280, tile_overlap=0.5,
                        checkpoint_path=None, checkpoint_frequency=-1, remove_tiles=False, 
                        yolo_inference_options=None,
                        n_patch_extraction_workers=default_n_patch_extraction_workers,
                        overwrite_tiles=True,
                        image_list=None):
    """
    Runs inference using [model_file] on the images in [image_folder], fist splitting each image up 
    into tiles of size [tile_size_x] x [tile_size_y], writing those tiles to [tiling_folder],
    then de-duplicating the results before merging them back into a set of detections that make 
    sense on the original images and writing those results to [output_file].  
    
    [tiling_folder] can be any folder, but this function reserves the right to do whatever it wants
    within that folder, including deleting everything, so it's best if it's a new folder.  
    Conceptually this folder is temporary, it's just helpful in this case to not actually
    use the system temp folder, because the tile cache may be very large, so the caller may 
    want it to be on a specific drive.
    
    tile_overlap is the fraction of overlap between tiles.
    
    Optionally removes the temporary tiles.
    
    if yolo_inference_options is supplied, it should be an instance of YoloInferenceOptions; in 
    this case the model will be run with run_inference_with_yolov5_val.  This is typically used to 
    run the model with test-time augmentation.
    
    Args:
        model_file (str): model filename (ending in .pt), or a well-known model name (e.g. "MDV5A")
        image_folder (str): the folder of images to proess (always recursive)
        tiling_folder (str): folder for temporary tile storage; see caveats above
        output_file (str): .json file to which we should write MD-formatted results
        tile_size_x (int, optional): tile width
        tile_size_y (int, optional): tile height
        tile_overlap (float, optional): overlap between adjacenet tiles, as a fraction of the
            tile size
        checkpoint_path (str, optional): checkpoint path; passed directly to run_detector_batch; see
            run_detector_batch for details
        checkpoint_frequency (int, optional): checkpoint frequency; passed directly to run_detector_batch; see
            run_detector_batch for details
        remove_tiles (bool, optional): whether to delete the tiles when we're done
        yolo_inference_options (YoloInferenceOptions, optional): if not None, will run inference with
            run_inference_with_yolov5_val.py, rather than with run_detector_batch.py, using these options
        n_patch_extraction_workers (int, optional): number of workers to use for patch extraction;
            set to <= 1 to disable parallelization
        image_list (list, optional): .json file containing a list of specific images to process.  If 
            this is supplied, and the paths are absolute, [image_folder] will be ignored. If this is supplied,
            and the paths are relative, they should be relative to [image_folder].
    
    Returns:
        dict: MD-formatted results dictionary, identical to what's written to [output_file]
    """

    ##%% Validate arguments
    
    assert tile_overlap < 1 and tile_overlap >= 0, \
        'Illegal tile overlap value {}'.format(tile_overlap)
    
    if tile_size_x == -1:
        tile_size_x = default_tile_size[0]
    if tile_size_y == -1:
        tile_size_y = default_tile_size[1]
        
    patch_size = [tile_size_x,tile_size_y]
    patch_stride = (round(patch_size[0]*(1.0-tile_overlap)),
                    round(patch_size[1]*(1.0-tile_overlap)))
    
    os.makedirs(tiling_folder,exist_ok=True)
    
    ##%% List files
    
    if image_list is None:
        
        print('Enumerating images in {}'.format(image_folder))
        image_files_relative = path_utils.find_images(image_folder, recursive=True, return_relative_paths=True)    
        assert len(image_files_relative) > 0, 'No images found in folder {}'.format(image_folder)
        
    else:
        
        print('Loading image list from {}'.format(image_list))
        with open(image_list,'r') as f:
            image_files_relative = json.load(f)
        n_absolute_paths = 0
        for i_fn,fn in enumerate(image_files_relative):
            if os.path.isabs(fn):
                n_absolute_paths += 1
                try:
                    fn_relative = os.path.relpath(fn,image_folder)
                except ValueError:
                    'Illegal absolute path supplied to run_tiled_inference, {} is outside of {}'.format(
                        fn,image_folder)
                    raise
                assert not fn_relative.startswith('..'), \
                    'Illegal absolute path supplied to run_tiled_inference, {} is outside of {}'.format(
                        fn,image_folder)
                image_files_relative[i_fn] = fn_relative
        if (n_absolute_paths != 0) and (n_absolute_paths != len(image_files_relative)):
            raise ValueError('Illegal file list: converted {} of {} paths to relative'.format(
            n_absolute_paths,len(image_files_relative)))
    
    ##%% Generate tiles
    
    all_image_patch_info = None
    
    print('Extracting patches from {} images'.format(len(image_files_relative)))
    
    n_workers = n_patch_extraction_workers
    
    if n_workers <= 1:
        
        all_image_patch_info = []
        
        # fn_relative = image_files_relative[0]        
        for fn_relative in tqdm(image_files_relative):        
            image_patch_info = \
                _extract_tiles_for_image(fn_relative,image_folder,tiling_folder,patch_size,patch_stride,
                                         overwrite=overwrite_tiles)
            all_image_patch_info.append(image_patch_info)
            
    else:
        
        from multiprocessing.pool import ThreadPool
        from multiprocessing.pool import Pool
        from functools import partial

        if n_workers > len(image_files_relative):
            
            print('Pool of {} requested, but only {} images available, reducing pool to {}'.\
                  format(n_workers,len(image_files_relative),len(image_files_relative)))
            n_workers = len(image_files_relative)
                                
        if parallelization_uses_threads:
            pool = ThreadPool(n_workers); poolstring = 'threads'                
        else:
            pool = Pool(n_workers); poolstring = 'processes'

        print('Starting patch extraction pool with {} {}'.format(n_workers,poolstring))
        
        all_image_patch_info = list(tqdm(pool.imap(
                partial(_extract_tiles_for_image,
                        image_folder=image_folder,
                        tiling_folder=tiling_folder,
                        patch_size=patch_size,
                        patch_stride=patch_stride,
                        overwrite=overwrite_tiles), 
                image_files_relative),total=len(image_files_relative)))
        
    # ...for each image
    
    # Write tile information to file; this is just a debugging convenience
    folder_name = path_utils.clean_filename(image_folder,force_lower=True)
    if folder_name.startswith('_'):
        folder_name = folder_name[1:]
        
    tile_cache_file = os.path.join(tiling_folder,folder_name + '_patch_info.json')
    with open(tile_cache_file,'w') as f:
        json.dump(all_image_patch_info,f,indent=1)
    
    # Keep track of patches that failed
    images_with_patch_errors = {}
    for patch_info in all_image_patch_info:
        if patch_info['error'] is not None:
            images_with_patch_errors[patch_info['image_fn']] = patch_info
    
    
    ##%% Run inference on tiles
    
    # When running with run_inference_with_yolov5_val, we'll pass the folder
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
    
    # For standard inference, we'll pass a list of files
    else:
        
        patch_file_names = []
        for im in all_image_patch_info:
            # If there was a patch generation error, don't run inference 
            if patch_info['error'] is not None:
                assert im['image_fn'] in images_with_patch_errors
                continue
            for patch in im['patches']:
                patch_file_names.append(patch['patch_fn'])
                
        inference_results = load_and_run_detector_batch(model_file, 
                                                        patch_file_names, 
                                                        checkpoint_path=checkpoint_path,
                                                        checkpoint_frequency=checkpoint_frequency,
                                                        quiet=True)
        
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
    for i_image,image_fn_relative in tqdm(enumerate(image_files_relative),
                                          total=len(image_files_relative)):
        
        image_fn_abs = os.path.join(image_folder,image_fn_relative)
        assert os.path.isfile(image_fn_abs)
                
        output_im = {}
        output_im['file'] = image_fn_relative
        
        # If we had a patch generation error
        if image_fn_relative in images_with_patch_errors:
            
            patch_info = image_fn_relative_to_patch_info[image_fn_relative]
            assert patch_info['error'] is not None
            
            output_im['detections'] = None
            output_im['failure'] = 'Patch generation error'
            output_im['failure_details'] = patch_info['error']
            image_level_results['images'].append(output_im)
            continue
                    
        try:
            pil_im = vis_utils.open_image(image_fn_abs)        
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]
        
        # This would be a very unusual situation; we're reading back an image here that we already
        # (successfully) read once during patch generation.
        except Exception as e:
            print('Warning: image read error after successful patch generation for {}:\n{}'.format(
                image_fn_relative,str(e)))
            output_im['detections'] = None
            output_im['failure'] = 'Patch processing error'
            output_im['failure_details'] = str(e)
            image_level_results['images'].append(output_im)
            continue            
        
        output_im['detections'] = []
        
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
            
            # If there was an inference failure on one patch, report the image
            # as an inference failure
            if 'detections' not in patch_results:
                assert 'failure' in patch_results
                output_im['detections'] = None
                output_im['failure'] = patch_results['failure']
                break
            
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

    #%% Run tiled inference (in Python)
    
    model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    image_folder = os.path.expanduser('~/data/KRU-test')
    tiling_folder = os.path.expanduser('~/tmp/tiling-test')
    output_file = os.path.expanduser('~/tmp/KRU-test-tiled.json')

    tile_size_x = 3000
    tile_size_y = 3000
    tile_overlap = 0.5
    checkpoint_path = None
    checkpoint_frequency = -1
    remove_tiles = False
    
    use_yolo_inference = False
    
    if not use_yolo_inference:
        
        yolo_inference_options = None
        
    else:
        
        yolo_inference_options = YoloInferenceOptions()
        yolo_inference_options.yolo_working_folder = os.path.expanduser('~/git/yolov5')
                    
    run_tiled_inference(model_file, image_folder, tiling_folder, output_file,
                            tile_size_x=tile_size_x, tile_size_y=tile_size_y, 
                            tile_overlap=tile_overlap,
                            checkpoint_path=checkpoint_path, 
                            checkpoint_frequency=checkpoint_frequency, 
                            remove_tiles=remove_tiles, 
                            yolo_inference_options=yolo_inference_options)
    
    
    #%% Run tiled inference (generate a command)
    
    import os
    
    model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    image_folder = os.path.expanduser('~/data/KRU-test')
    tiling_folder = os.path.expanduser('~/tmp/tiling-test')
    output_file = os.path.expanduser('~/tmp/KRU-test-tiled.json')
    tile_size = [5152,3968]
    tile_overlap = 0.8
    
    cmd = f'python run_tiled_inference.py {model_file} {image_folder} {tiling_folder} {output_file} ' + \
          f'--tile_overlap {tile_overlap} --no_remove_tiles --tile_size_x {tile_size[0]} --tile_size_y {tile_size[1]}'
    
    print(cmd)
    import clipboard; clipboard.copy(cmd)
    
    
    #%% Preview tiled inference
    
    from megadetector.postprocessing.postprocess_batch_results import \
        PostProcessingOptions, process_batch_results

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

    options.md_results_file = output_file
    options.output_dir = preview_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file

    path_utils.open_file(html_output_file)
    
    
#%% Command-line driver

import sys,argparse

def main():
            
    parser = argparse.ArgumentParser(
        description='Chop a folder of images up into tiles, run MD on the tiles, and stitch the results together')
    parser.add_argument(
        'model_file',
        help='Path to detector model file (.pb or .pt)')
    parser.add_argument(
        'image_folder',
        help='Folder containing images for inference (always recursive, unless image_list is supplied)')
    parser.add_argument(
        'tiling_folder',
        help='Temporary folder where tiles and intermediate results will be stored')
    parser.add_argument(
        'output_file',
        help='Path to output JSON results file, should end with a .json extension')
    parser.add_argument(
        '--no_remove_tiles',
        action='store_true',
        help='Tiles are removed by default; this option suppresses tile deletion')    
    parser.add_argument(
        '--tile_size_x',
        type=int,
        default=default_tile_size[0],
        help=('Tile width (defaults to {})'.format(default_tile_size[0])))
    parser.add_argument(
        '--tile_size_y',
        type=int,
        default=default_tile_size[0],
        help=('Tile height (defaults to {})'.format(default_tile_size[1])))
    parser.add_argument(
        '--tile_overlap',
        type=float,
        default=default_patch_overlap,
        help=('Overlap between tiles [0,1] (defaults to {})'.format(default_patch_overlap)))
    parser.add_argument(
        '--overwrite_handling',
        type=str,
        default='skip',
        help=('Behavior when the target file exists (skip/overwrite/error) (default skip)'))
    parser.add_argument(
        '--image_list',
        type=str,
        default=None,
        help=('A .json list of relative filenames (or absolute paths contained within image_folder) to include'))
        
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    model_file = try_download_known_detector(args.model_file)
    assert os.path.exists(model_file), \
        'detector file {} does not exist'.format(args.model_file)
        
    if os.path.exists(args.output_file):
        if args.overwrite_handling == 'skip':
            print('Warning: output file {} exists, skipping'.format(args.output_file))
            return
        elif args.overwrite_handling == 'overwrite':
            print('Warning: output file {} exists, overwriting'.format(args.output_file))
        elif args.overwrite_handling == 'error':
            raise ValueError('Output file {} exists'.format(args.output_file))
        else:
            raise ValueError('Unknown output handling method {}'.format(args.overwrite_handling))
        

    remove_tiles = (not args.no_remove_tiles)

    run_tiled_inference(model_file, args.image_folder, args.tiling_folder, args.output_file,
                        tile_size_x=args.tile_size_x, tile_size_y=args.tile_size_y, 
                        tile_overlap=args.tile_overlap,
                        remove_tiles=remove_tiles,
                        image_list=args.image_list)
        
if __name__ == '__main__':
    main()
