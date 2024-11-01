"""

detector_calibration.py

Tools for comparing/calibrating confidence values from detectors, particularly different
versions of MegaDetector.

"""

#%% Constants and imports

import random

from tqdm import tqdm
from enum import IntEnum
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from megadetector.postprocessing.validate_batch_results import \
    validate_batch_results, ValidateBatchResultsOptions
from megadetector.utils.ct_utils import get_iou


#%% Classes

class CalibrationOptions:
    """
    Options controlling comparison/calibration behavior.
    """
    
    def __init__(self):
        
        #: IoU threshold used for determining whether two detections are the same
        #:
        #: When multiple detections match, we will only use the highest-matching IoU.
        self.iou_threshold = 0.75
        
        #: Minimum confidence threshold to consider for calibration (should be lower than
        #: the lowest value you would use in realistic situations)
        self.confidence_threshold = 0.025
    
        #: Should we populate the data_a and data_b fields in the return value?
        self.return_data = False
        
        #: Model name to use in printouts and plots for result set A
        self.model_name_a = 'model_a'
        
        #: Model name to use in printouts and plots for result set B
        self.model_name_b = 'model_b'
        
        #: Maximum number of samples to use for plotting or calibration per category,
        #: or None to use all paired values.
        self.max_samples_per_category = None
        
        #: List of category IDs to use for plotting comparisons, or None to plot
        #: all categories.
        self.categories_to_plot = None
        
        #: Optionally map category ID to name in plot labels
        self.category_id_to_name = None
        
# ...class CalibrationOptions

class ConfidenceMatchColumns(IntEnum):
    
    COLUMN_CONF_A = 0
    COLUMN_CONF_B = 1
    COLUMN_CONF_IOU = 2
    COLUMN_CONF_I_IMAGE = 3
    COLUMN_CONF_CATEGORY_ID = 4
    
class CalibrationResults:
    """
    Results of a model-to-model comparison.
    """
        
    def __init__(self):
        
        #: List of tuples: [conf_a, conf_b, iou, i_image, category_id]
        self.confidence_matches = []
        
        #: Populated with the data loaded from json_filename_a if options.return_data is True
        self.data_a = None
        
        #: Populated with the data loaded from json_filename_b if options.return_data is True
        self.data_b = None

# ...class CalibrationResults        


#%% Calibration functions

def compare_model_confidence_values(json_filename_a,json_filename_b,options=None):
    """
    Compare confidence values across two .json results files.  Compares only detections that
    can be matched by IoU, i.e., does not do anything with detections that only appear in one file.
    
    Args: 
        json_filename_a (str or dict): filename containing results from the first model to be compared; 
            should refer to the same images as [json_filename_b].  Can also be a loaded results dict.
        json_filename_b (str or dict): filename containing results from the second model to be compared; 
            should refer to the same images as [json_filename_a].  Can also be a loaded results dict.
        options (CalibrationOptions, optional): all the parameters used to control this process, see 
            CalibrationOptions for details
            
    Returns:
        CalibrationResults: description of the comparison results
    """
    
    ## Option handling 
    
    if options is None:
        options = CalibrationOptions()
    
    validation_options = ValidateBatchResultsOptions()
    validation_options.return_data = True
    
    if isinstance(json_filename_a,str):
        results_a = validate_batch_results(json_filename_a,options=validation_options)
        assert len(results_a['validation_results']['errors']) == 0
    else:
        assert isinstance(json_filename_a,dict)
        results_a = json_filename_a        
        
    if isinstance(json_filename_b,str):
        results_b = validate_batch_results(json_filename_b,options=validation_options)
        assert len(results_b['validation_results']['errors']) == 0
    else:
        assert isinstance(json_filename_b,dict)
        results_b = json_filename_b
        
        
    ## Make sure these results sets are comparable
    
    image_filenames_a = [im['file'] for im in results_a['images']]
    image_filenames_b = [im['file'] for im in results_b['images']]
    
    assert set(image_filenames_a) == set(image_filenames_b), \
        'Cannot calibrate non-matching image sets'
    
    categories_a = results_a['detection_categories']
    categories_b = results_b['detection_categories']
    assert set(categories_a.keys()) == set(categories_b.keys())
    for k in categories_a.keys():
        assert categories_a[k] == categories_b[k], 'Category mismatch'
    
    
    ## Compare detections
    
    image_filename_b_to_im = {}
    for im in results_b['images']:
        image_filename_b_to_im[im['file']] = im
    
    n_detections_a = 0
    n_detections_a_queried = 0
    n_detections_a_matched = 0
    
    confidence_matches = []
    
    # For each image
    # im_a = results_a['images'][0]
    for i_image,im_a in tqdm(enumerate(results_a['images']),total=len(results_a['images'])):
            
        fn = im_a['file']
        im_b = image_filename_b_to_im[fn]
        
        if 'detections' not in im_a or im_a['detections'] is None:
            continue
        if 'detections' not in im_b or im_b['detections'] is None:
            continue
    
        # For each detection in result set A...
        #
        # det_a = im_a['detections'][0]
        for det_a in im_a['detections']:
            
            n_detections_a += 1
            
            conf_a = det_a['conf']
            category_id = det_a['category']
            
            # Is this above threshold?
            if conf_a < options.confidence_threshold:
                continue
            
            n_detections_a_queried += 1
            
            bbox_a = det_a['bbox']
            
            best_iou = None
            best_iou_conf = None
                        
            # For each detection in result set B...
            #
            # det_b = im_b['detections'][0]
            for det_b in im_b['detections']:
            
                # Is this the same category?
                if det_b['category'] != category_id:
                    continue
                
                conf_b = det_b['conf']
                
                # Is this above threshold?
                if conf_b < options.confidence_threshold:
                    continue
                
                bbox_b = det_b['bbox']
                
                iou = get_iou(bbox_a,bbox_b)
                
                # Is this an adequate IoU to consider?
                if iou < options.iou_threshold:
                    continue
                
                # Is this the best match so far?
                if best_iou is None or iou > best_iou:
                    best_iou = iou
                    best_iou_conf = conf_b
    
            # ...for each detection in im_b
        
            if best_iou is not None:
                n_detections_a_matched += 1
                conf_result = [conf_a,best_iou_conf,best_iou,i_image,category_id]
                confidence_matches.append(conf_result)
        
        # ...for each detection in im_a
        
    # ...for each image in result set A
    
    print('\nOf {} detections in result set A, queried {}, matched {}'.format(
        n_detections_a,n_detections_a_queried,n_detections_a_matched))
    assert len(confidence_matches) == n_detections_a_matched

    calibration_results = CalibrationResults()        
    calibration_results.confidence_matches = confidence_matches

    if options.return_data:
        calibration_results.data_a = results_a
        calibration_results.data_b = results_b
        
    return calibration_results

# ...def compare_model_confidence_values(...)


#%% Plotting functions

def plot_matched_confidence_values(calibration_results,output_filename,options=None):
    """
    Given a set of paired confidence values for matching detections (from 
    compare_model_confidence_values), plot histograms of those pairs for each
    detection category.
    
    Args: 
        calibration_results (CalibrationResults): output from a call to 
            compare_model_confidence_values, containing paired confidence
            values for two sets of detection results.
        output_filename (str): filename to write the plot (.png or .jpg)
        options (CalibrationOptions, optional): plotting options, see
            CalibrationOptions for details.
    """
    
    fig_w = 12
    fig_h = 8
    n_hist_bins = 80
    
    if options is None:
        options = CalibrationOptions()
    
    # Find matched confidence pairs for each category ID
    category_to_matches = defaultdict(list)
    
    confidence_matches = calibration_results.confidence_matches
    for m in confidence_matches:
        category_id = m[ConfidenceMatchColumns.COLUMN_CONF_CATEGORY_ID]
        category_to_matches[category_id].append(m)
    
    # Optionally sample matches
    category_to_samples = defaultdict(list)
           
    for i_category,category_id in enumerate(category_to_matches.keys()):
        
        matches_this_category = category_to_matches[category_id]
        
        if (options.max_samples_per_category is None) or \
            (len(matches_this_category) <= options.max_samples_per_category):
            category_to_samples[category_id] = matches_this_category
        else:
            assert len(matches_this_category) > options.max_samples_per_category
            category_to_samples[category_id] = random.sample(matches_this_category,options.max_samples_per_category)
    
    del category_to_matches
    del confidence_matches
    
    categories_to_plot = list(category_to_samples.keys())
    
    if options.categories_to_plot is not None:
        categories_to_plot = [category_id for category_id in categories_to_plot if\
                              category_id in options.categories_to_plot]    
    
    n_subplots = len(categories_to_plot)
    
    plt.ioff()

    fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)    
    # fig,axes = plt.subplots(nrows=n_subplots,ncols=1)    
    
    axes = fig.subplots(n_subplots, 1)

    # i_category = 0; category_id = categories_to_plot[i_category]
    for i_category,category_id in enumerate(categories_to_plot):
    
        ax = axes[i_category]
        
        category_string = category_id
        if options.category_id_to_name is not None and \
            category_id in options.category_id_to_name:
            category_string = options.category_id_to_name[category_id]
            
        samples_this_category = category_to_samples[category_id]
        x = [m[0] for m in samples_this_category]
        y = [m[1] for m in samples_this_category]
        
        weights_a = np.ones_like(x)/float(len(x))
        weights_b = np.ones_like(y)/float(len(y))
        ax.hist(x,histtype='step',bins=n_hist_bins,density=False,color='red',weights=weights_a)
        ax.hist(y,histtype='step',bins=n_hist_bins,density=False,color='blue',weights=weights_b)
        ax.legend([options.model_name_a,options.model_name_b])
        ax.set_ylabel(category_string)
        # plt.tight_layout()
        
        # I experimented with heat maps, but they weren't very informative.
        # Leaving this code here in case I revisit.  Note to self: scatter plots
        # were a disaster.
        if False:        
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=30)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(heatmap.T, extent=extent, origin='lower', norm='log')
    
    # ...for each category for which we need to generate a histogram
    
    plt.close(fig)
    fig.savefig(output_filename,dpi=100)
    
# ...def plot_matched_confidence_values(...)


#%% Interactive driver(s)

if False:

    #%%
    
    options = ValidateBatchResultsOptions()
    # json_filename = r'g:\temp\format.json'
    # json_filename = r'g:\temp\test-videos\video_results.json'
    json_filename = r'g:\temp\test-videos\image_results.json'
    options.check_image_existence = True
    options.relative_path_base = r'g:\temp\test-videos'
    validate_batch_results(json_filename,options)
    

