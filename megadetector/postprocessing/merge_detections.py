"""

merge_detections.py

Merge high-confidence detections from one or more results files into another 
file.  Typically used to combine results from MDv5b and/or MDv4 into a "primary"
results file from MDv5a.

Detection categories must be the same in both files; if you want to first remap
one file's category mapping to be the same as another's, see remap_detection_categories.

If you want to literally merge two .json files, see combine_api_outputs.py.

"""

#%% Constants and imports

import argparse
import sys
import json
import os

from tqdm import tqdm

from megadetector.utils.ct_utils import get_iou


#%% Structs

class MergeDetectionsOptions:
    
    def __init__(self):
        
        #: Maximum detection size to include in the merged output
        self.max_detection_size = 1.01
        
        #: Minimum detection size to include in the merged output
        self.min_detection_size = 0
        
        #: Exclude detections whose confidence in the source file(s) is less
        #: than this.  Should have the same length as the number of source files.
        self.source_confidence_thresholds = [0.05]
        
        #: Don't bother merging into target images if there is a similar detection
        #: above this threshold (or if there is *any* detection above this threshold,
        #: and merge_empty_only is True)
        self.target_confidence_threshold = 0.2
        
        #: If you want to merge only certain categories, specify one
        #: (but not both) of these.  These are category IDs, not names.
        self.categories_to_include = None
        
        #: If you want to merge only certain categories, specify one
        #: (but not both) of these.  These are category IDs, not names.
        self.categories_to_exclude = None

        #: Only merge detections into images that have *no* detections in the 
        #: target results file.
        self.merge_empty_only = False
        
        #: IoU threshold above which two detections are considered the same
        self.iou_threshold = 0.65
        
        #: Error if this is False and the output file exists
        self.overwrite = False


#%% Main function

def merge_detections(source_files,target_file,output_file,options=None):
    """
    Merge high-confidence detections from one or more results files into another 
    file.   Typically used to combine results from MDv5b and/or MDv4 into a "primary"
    results file from MDv5a.
    
    [source_files] (a list of files or a single filename) specifies the set of 
    results files that will be merged into [target_file].  The difference between a 
    "source file" and the "target file" is that if no merging is necessary, either because
    two boxes are nearly identical or because merge_only_empty is True and the target
    file already has above-threshold detection for an image+category, the output file gets
    the results of the "target" file.  I.e., the "target" file wins all ties.
    
    The results are written to [output_file].

    """
    
    if isinstance(source_files,str):
        source_files = [source_files]    
        
    if options is None:
        options = MergeDetectionsOptions()    
        
    if (not options.overwrite) and (os.path.isfile(output_file)):
        print('File {} exists, bypassing merge'.format(output_file))
        return
    
    assert not ((options.categories_to_exclude is not None) and \
                (options.categories_to_include is not None)), \
                'categories_to_include and categories_to_exclude are mutually exclusive'
    
    if options.categories_to_exclude is not None:
        options.categories_to_exclude = [int(c) for c in options.categories_to_exclude]
        
    if options.categories_to_include is not None:
        options.categories_to_include = [int(c) for c in options.categories_to_include]
        
    assert len(source_files) == len(options.source_confidence_thresholds), \
        '{} source files provided, but {} source confidence thresholds provided'.format(
            len(source_files),len(options.source_confidence_thresholds))
    
    for fn in source_files:
        assert os.path.isfile(fn), 'Could not find source file {}'.format(fn)
    
    assert os.path.isfile(target_file)
    
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    
    with open(target_file,'r') as f:
        output_data = json.load(f)

    print('Loaded results for {} images'.format(len(output_data['images'])))
    
    fn_to_image = {}
    
    # im = output_data['images'][0]
    for im in output_data['images']:
        fn_to_image[im['file']] = im
    
    if 'detections_transferred_from' not in output_data['info']:
        output_data['info']['detections_transferred_from'] = []

    if 'detector' not in output_data['info']:
        output_data['info']['detector'] = 'MDv4 (assumed)'
        
    detection_categories_raw = output_data['detection_categories'].keys()
    
    # Determine whether we should be processing all categories, or just a subset
    # of categories.
    detection_categories = []

    if options.categories_to_exclude is not None:    
        for c in detection_categories_raw:
            if int(c) not in options.categories_to_exclude:
                detection_categories.append(c)
            else:
                print('Excluding category {}'.format(c))
    elif options.categories_to_include is not None:
        for c in detection_categories_raw:
            if int(c) in options.categories_to_include:
                print('Including category {}'.format(c))
                detection_categories.append(c)
    else:
        detection_categories = detection_categories_raw
    
    # i_source_file = 0; source_file = source_files[i_source_file]
    for i_source_file,source_file in enumerate(source_files):
    
        print('Processing detections from file {}'.format(source_file))
        
        with open(source_file,'r') as f:
            source_data = json.load(f)
        
        if 'detector' in source_data['info']:
            source_detector_name = source_data['info']['detector']
        else:
            source_detector_name = os.path.basename(source_file)
        
        output_data['info']['detections_transferred_from'].append(os.path.basename(source_file))
        output_data['info']['detector'] = output_data['info']['detector'] + ' + ' + source_detector_name
        
        assert source_data['detection_categories'] == output_data['detection_categories'], \
            'Cannot merge files with different detection category maps'
        
        source_confidence_threshold = options.source_confidence_thresholds[i_source_file]
        
        # source_im = source_data['images'][0]
        for source_im in tqdm(source_data['images']):
            
            image_filename = source_im['file']            
            
            assert image_filename in fn_to_image, 'Image {} not in target image set'.format(image_filename)
            target_im = fn_to_image[image_filename]
            
            if 'detections' not in source_im or source_im['detections'] is None:
                continue
            
            if 'detections' not in target_im or target_im['detections'] is None:
                continue
                    
            source_detections_this_image = source_im['detections']
            target_detections_this_image = target_im['detections']
              
            detections_to_transfer = []
            
            # detection_category = list(detection_categories)[0]
            for detection_category in detection_categories:
                
                target_detections_this_category = \
                    [det for det in target_detections_this_image if det['category'] == \
                     detection_category]
                
                max_target_confidence_this_category = 0.0
                
                if len(target_detections_this_category) > 0:
                    max_target_confidence_this_category = max([det['conf'] for \
                      det in target_detections_this_category])
                
                # If we have a valid detection in the target file, and we're only merging
                # into images that have no detections at all, we don't need to review the individual
                # detections in the source file.
                if options.merge_empty_only and \
                    (max_target_confidence_this_category >= options.target_confidence_threshold):
                    continue
                
                source_detections_this_category_raw = [det for det in \
                  source_detections_this_image if det['category'] == detection_category]
                
                # Boxes are x/y/w/h
                # source_sizes = [det['bbox'][2]*det['bbox'][3] for det in source_detections_this_category_raw]
                
                # Only look at source boxes within the size range
                source_detections_this_category_filtered = [
                    det for det in source_detections_this_category_raw if \
                        (det['bbox'][2]*det['bbox'][3] <= options.max_detection_size) and \
                        (det['bbox'][2]*det['bbox'][3] >= options.min_detection_size) \
                        ]
                           
                # det = source_detections_this_category_filtered[0]
                for det in source_detections_this_category_filtered:
                    
                    if det['conf'] >= source_confidence_threshold:

                        # Check only whole images
                        if options.merge_empty_only:
                            
                            # We verified this above, asserting here for clarity
                            assert max_target_confidence_this_category < options.target_confidence_threshold
                            det['transferred_from'] = source_detector_name
                            detections_to_transfer.append(det)

                        # Check individual detections                       
                        else:
                            
                            # Does this source detection match any existing above-threshold
                            # target category detections?
                            matches_existing_box = False
                            
                            # target_detection = target_detections_this_category[0]
                            for target_detection in target_detections_this_category:
                                
                                if (target_detection['conf'] >= options.target_confidence_threshold) \
                                   and \
                                   (get_iou(det['bbox'],target_detection['bbox']) >= options.iou_threshold):
                                    matches_existing_box = True
                                    break
                                
                            if (not matches_existing_box):
                                det['transferred_from'] = source_detector_name
                                detections_to_transfer.append(det)
                    
                    # ...if this source detection is above the confidence threshold
                    
                # ...for each source detection within category                            
                                    
            # ...for each detection category
            
            if len(detections_to_transfer) > 0:
                
                # print('Adding {} detections to image {}'.format(len(detections_to_transfer),image_filename))
                detections = fn_to_image[image_filename]['detections']                
                detections.extend(detections_to_transfer)

                # Update the max_detection_conf field (if present)
                if 'max_detection_conf' in fn_to_image[image_filename]:
                    fn_to_image[image_filename]['max_detection_conf'] = \
                        max([d['conf'] for d in detections])
            
            # ...if we have any detections to transfer
            
        # ...for each image
        
    # ...for each source file        
    
    with open(output_file,'w') as f:
        json.dump(output_data,f,indent=1)
    
    print('Saved merged results to {}'.format(output_file))


#%% Command-line driver

def main():
    
    default_options = MergeDetectionsOptions()
    
    parser = argparse.ArgumentParser(
        description='Merge detections from one or more MegaDetector results files into an existing reuslts file')
    parser.add_argument(
        'source_files',
        nargs="+",
        help='Path to source .json file(s) to merge from')
    parser.add_argument(
        'target_file',
        help='Path to a .json file to merge detections into')
    parser.add_argument(
        'output_file',
        help='Path to output .json results file')
    parser.add_argument(
        '--max_detection_size',
        type=float,
        default=default_options.max_detection_size,
        help='Ignore detections with an area larger than this (as a fraction of ' + \
             'image size) (default {})'.format(
             default_options.max_detection_size))
    parser.add_argument(
        '--min_detection_size',
        default=default_options.min_detection_size,
        type=float,
        help='Ignore detections with an area smaller than this (as a fraction of ' + \
              'image size) (default {})'.format(
              default_options.min_detection_size))
    parser.add_argument(
        '--source_confidence_thresholds',
        nargs="+",
        type=float,
        default=default_options.source_confidence_thresholds,
        help='List of thresholds for each source file (default {}). '.format(
            default_options.source_confidence_thresholds) + \
            'Merge only if the source file\'s detection confidence is higher than its ' + \
            'corresponding threshold.  Should be the same length as the number of source files.')
    parser.add_argument(
        '--target_confidence_threshold',
        type=float,
        default=default_options.target_confidence_threshold,
        help='Don\'t merge if target file\'s detection confidence is already higher ' + \
             'than this (default {}). '.format(
             default_options.target_confidence_threshold))
    parser.add_argument(
        '--categories_to_include',
        type=int,
        nargs="+",
        default=None,
        help='List of numeric detection category IDs to include')
    parser.add_argument(
        '--categories_to_exclude',
        type=int,
        nargs="+",
        default=None,
        help='List of numeric detection categories to include')
    parser.add_argument(
        '--merge_empty_only',
        action='store_true',
        help='Ignore individual detections and only merge images for which the target ' + \
             'file contains no detections')   
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=default_options.iou_threshold,
        help='Sets the minimum IoU for a source detection to be considered the same as ' + \
             'a target detection (default {})'.format(default_options.iou_threshold))

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    options = MergeDetectionsOptions()
    options.max_detection_size = args.max_detection_size
    options.min_detection_size = args.min_detection_size
    options.source_confidence_thresholds = args.source_confidence_thresholds
    options.target_confidence_threshold = args.target_confidence_threshold
    options.categories_to_include = args.categories_to_include
    options.categories_to_exclude = args.categories_to_exclude
    options.merge_empty_only = args.merge_empty_only
    options.iou_threshold = args.iou_threshold
    
    merge_detections(args.source_files, args.target_file, args.output_file, options)


#%% Test driver

if False:
    
    #%%
    
    options = MergeDetectionsOptions()
    options.max_detection_size = 0.1
    options.target_confidence_threshold = 0.3
    options.categories_to_include = [1]
    source_files = ['/home/user/postprocessing/iwildcam/iwildcam-mdv4-2022-05-01/combined_api_outputs/iwildcam-mdv4-2022-05-01_detections.json']
    options.source_confidence_thresholds = [0.8]
    target_file = '/home/user/postprocessing/iwildcam/iwildcam-mdv5-camcocoinat-2022-05-02/combined_api_outputs/iwildcam-mdv5-camcocoinat-2022-05-02_detections.json'
    output_file = '/home/user/postprocessing/iwildcam/merged-detections/mdv4_mdv5-camcocoinat-2022-05-02.json'
    merge_detections(source_files, target_file, output_file, options)
    
    options = MergeDetectionsOptions()
    options.max_detection_size = 0.1
    options.target_confidence_threshold = 0.3
    options.categories_to_include = [1]
    source_files = [
        '/home/user/postprocessing/iwildcam/iwildcam-mdv4-2022-05-01/combined_api_outputs/iwildcam-mdv4-2022-05-01_detections.json',
        '/home/user/postprocessing/iwildcam/iwildcam-mdv5-camonly-2022-05-02/combined_api_outputs/iwildcam-mdv5-camonly-2022-05-02_detections.json',
        ]
    options.source_confidence_thresholds = [0.8,0.5]
    target_file = '/home/user/postprocessing/iwildcam/iwildcam-mdv5-camcocoinat-2022-05-02/combined_api_outputs/iwildcam-mdv5-camcocoinat-2022-05-02_detections.json'
    output_file = '/home/user/postprocessing/iwildcam/merged-detections/mdv4_mdv5-camonly_mdv5-camcocoinat-2022-05-02.json'
    merge_detections(source_files, target_file, output_file, options)
    
if __name__ == '__main__':
    main()

