"""

create_crop_folder.py

Given a MegaDetector .json file and a folder of images, creates a new folder
of images representing all above-threshold crops from the original folder.

"""

#%% Constants and imports

import os
import json
from tqdm import tqdm

from multiprocessing.pool import Pool, ThreadPool
from collections import defaultdict
from functools import partial

from megadetector.utils.path_utils import insert_before_extension
from megadetector.visualization.visualization_utils import crop_image
from megadetector.visualization.visualization_utils import exif_preserving_save


#%% Support classes

class CreateCropFolderOptions:
    """
    Options used to parameterize create_crop_folder().
    """
    
    def __init__(self):
        
        #: Confidence threshold determining which detections get written
        self.confidence_threshold = 0.1
        
        #: Number of pixels to expand each crop
        self.expansion = 0
        
        #: JPEG quality to use for saving crops (None for default)
        self.quality = 95
        
        #: Whether to overwrite existing images
        self.overwrite = True
        
        #: Number of concurrent workers
        self.n_workers = 8
        
        #: Whether to use processes ('process') or threads ('thread') for parallelization
        self.pool_type = 'thread'
                
          
#%% Support functions

def _get_crop_filename(image_fn,crop_id):
    """
    Generate crop filenames in a consistent way.
    """
    if isinstance(crop_id,int):
        crop_id = str(crop_id).zfill(3)
    assert isinstance(crop_id,str)
    return insert_before_extension(image_fn,'crop_' + crop_id)


def _generate_crops_for_single_image(crops_this_image,
                                     input_folder,
                                     output_folder,
                                     options):
    """
    Generate all the crops required for a single image.
    """
    if len(crops_this_image) == 0:
        return
    
    image_fn_relative = crops_this_image[0]['image_fn_relative']    
    input_fn_abs = os.path.join(input_folder,image_fn_relative)
    assert os.path.isfile(input_fn_abs)
    
    detections_to_crop = [c['detection'] for c in crops_this_image]
    
    cropped_images = crop_image(detections_to_crop,
                                input_fn_abs,
                                confidence_threshold=0,
                                expansion=options.expansion)
    
    assert len(cropped_images) == len(crops_this_image)
    
    # i_crop = 0; crop_info = crops_this_image[0]
    for i_crop,crop_info in enumerate(crops_this_image):
        
        assert crop_info['image_fn_relative'] == image_fn_relative
        crop_filename_relative = _get_crop_filename(image_fn_relative, crop_info['crop_id'])        
        crop_filename_abs = os.path.join(output_folder,crop_filename_relative).replace('\\','/')
        
        if os.path.isfile(crop_filename_abs) and not options.overwrite:
            continue
         
        cropped_image = cropped_images[i_crop]            
        os.makedirs(os.path.dirname(crop_filename_abs),exist_ok=True)            
        exif_preserving_save(cropped_image,crop_filename_abs,quality=options.quality)
    
    # ...for each crop


#%% Main function
    
def create_crop_folder(input_file,
                       input_folder,
                       output_folder,
                       output_file=None,
                       crops_output_file=None,
                       options=None):
    """
    Given a MegaDetector .json file and a folder of images, creates a new folder
    of images representing all above-threshold crops from the original folder.
    
    Optionally writes a new .json file that attaches unique IDs to each detection.
    
    Args:
        input_file (str): MD-formatted .json file to process
        input_folder (str): Input image folder
        output_folder (str): Output (cropped) image folder
        output_file (str, optional): new .json file that attaches unique IDs to each detection.
        crops_output_file (str, optional): new .json file that includes whole-image detections
            for each of the crops, using confidence values from the original results
        options (CreateCropFolderOptions, optional): crop parameters    
    """
        
    ## Validate options, prepare output folders
    
    if options is None:
        options = CreateCropFolderOptions()

    assert os.path.isfile(input_file), 'Input file {} not found'.format(input_file)
    assert os.path.isdir(input_folder), 'Input folder {} not found'.format(input_folder)
    os.makedirs(output_folder,exist_ok=True)
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    
    
    ##%% Read input
    
    with open(input_file,'r') as f:
        detection_results = json.load(f)
       
        
    ##%% Make a list crops that we need to create
    
    # Maps input images to list of dicts, with keys 'crop_id','detection'
    image_fn_relative_to_crops = defaultdict(list)
    n_crops = 0
    
    # im = detection_results['images'][0]
    for i_image,im in enumerate(detection_results['images']):
        
        if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
            continue
        
        detections_this_image = im['detections']
        
        image_fn_relative = im['file']
        
        for i_detection,det in enumerate(detections_this_image):
            
            if det['conf'] > options.confidence_threshold:
                            
                det['crop_id'] = i_detection
                
                crop_info = {'image_fn_relative':image_fn_relative,
                             'crop_id':i_detection,
                             'detection':det}
                
                crop_filename_relative = _get_crop_filename(image_fn_relative, 
                                                            crop_info['crop_id'])
                det['crop_filename_relative'] = crop_filename_relative

                image_fn_relative_to_crops[image_fn_relative].append(crop_info)
                n_crops += 1
                
    # ...for each input image   

    print('Prepared a list of {} crops from {} of {} input images'.format(
        n_crops,len(image_fn_relative_to_crops),len(detection_results['images'])))
        
    
    ##%% Generate crops
        
    if options.n_workers <= 1:
                
        # image_fn_relative = next(iter(image_fn_relative_to_crops))
        for image_fn_relative in tqdm(image_fn_relative_to_crops.keys()):
            crops_this_image = image_fn_relative_to_crops[image_fn_relative]            
            _generate_crops_for_single_image(crops_this_image=crops_this_image,
                                             input_folder=input_folder,
                                             output_folder=output_folder,
                                             options=options)
                
    else:
        
        print('Creating a {} pool with {} workers'.format(options.pool_type,options.n_workers))

        if options.pool_type == 'thread':
            pool = ThreadPool(options.n_workers)
        else:
            assert options.pool_type == 'process'
            pool = Pool(options.n_workers)
        
        # Each element in this list is the list of crops for a single image
        crop_lists = list(image_fn_relative_to_crops.values())
        
        with tqdm(total=len(image_fn_relative_to_crops)) as pbar:
            for i,_ in enumerate(pool.imap_unordered(partial(
                        _generate_crops_for_single_image,
                            input_folder=input_folder,
                            output_folder=output_folder,
                            options=options),
                        crop_lists)):
                pbar.update()

    # ...if we're using parallel processing    
    
    ##%% Write output file
    
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(detection_results,f,indent=1)
        
    if crops_output_file is not None:
        
        original_images = detection_results['images']
        
        detection_results_cropped = detection_results
        detection_results_cropped['images'] = []
        
        # im = original_images[0]
        for im in original_images:
            
            if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
                continue
            
            detections_this_image = im['detections']            
            image_fn_relative = im['file']
            
            for i_detection,det in enumerate(detections_this_image):
                
                if 'crop_id' in det:
                    im_out = {}
                    im_out['file'] = det['crop_filename_relative']
                    det_out = {}
                    det_out['category'] = det['category']
                    det_out['conf'] = det['conf']
                    det_out['bbox'] = [0, 0, 1, 1]
                    im_out['detections'] = [det_out]
                    detection_results_cropped['images'].append(im_out)
            
                # ...if we need to include this crop in the new .json file
                
            # ...for each crop
            
        # ...for each original image
        
        with open(crops_output_file,'w') as f:
            json.dump(detection_results_cropped,f,indent=1)
    
# ...def create_crop_folder()


#%% Command-line driver

# TODO