"""

detector_calibration.py

Tools for comparing/calibrating confidence values from detectors, particularly different
versions of MegaDetector.

"""

#%% Constants and imports

import random
import copy

from tqdm import tqdm
from enum import IntEnum
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from megadetector.postprocessing.validate_batch_results import \
    validate_batch_results, ValidateBatchResultsOptions
from megadetector.utils.ct_utils import get_iou, max_none, is_iterable


#%% Classes

class CalibrationOptions:
    """
    Options controlling comparison/calibration behavior.
    """

    def __init__(self):

        #: IoU threshold used for determining whether two detections are the same
        #:
        #: When multiple detections match, we will only use the highest-matching IoU.
        self.iou_threshold = 0.6

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
        #: or None to use all paired values.  If separate_plots_by_category is False,
        #: this is the overall number of points sampled.
        self.max_samples_per_category = None

        #: Should we make separate plots for each category?  Mutually exclusive with
        #: separate_plots_by_correctness.
        self.separate_plots_by_category = True

        #: Should we make separate plots for TPs/FPs?  Mutually exclusive with
        #: separate_plots_by_category.
        self.separate_plots_by_correctness = False

        #: List of category IDs to use for plotting comparisons, or None to plot
        #: all categories.
        self.categories_to_plot = None

        #: Optionally map category ID to name in plot labels
        self.category_id_to_name = None

        #: Enable additional debug output
        self.verbose = True

# ...class CalibrationOptions

class CalibrationMatchColumns(IntEnum):
    """
    Enumeration defining columns in the calibration_matches list we'll assemble below.
    """

    COLUMN_CONF_A = 0
    COLUMN_CONF_B = 1
    COLUMN_IOU = 2
    COLUMN_I_IMAGE = 3
    COLUMN_CATEGORY_ID = 4
    COLUMN_MATCHES_GT = 5

class CalibrationResults:
    """
    Results of a model-to-model comparison.
    """

    def __init__(self):

        #: List of tuples: [conf_a, conf_b, iou, i_image, category_id, matches_gt]
        #:
        #: If ground truth is supplied, [matches_gt] is a bool indicating whether either
        #: of the detected boxes matches a ground truth box of the same category.  If
        #: ground truth is not supplied, [matches_gt] is None.
        self.calibration_matches = []

        #: Populated with the data loaded from json_filename_a if options.return_data is True
        self.data_a = None

        #: Populated with the data loaded from json_filename_b if options.return_data is True
        self.data_b = None

# ...class CalibrationResults


#%% Calibration functions

def compare_model_confidence_values(json_filename_a,json_filename_b,json_filename_gt=None,options=None):
    """
    Compare confidence values across two .json results files.  Compares only detections that
    can be matched by IoU, i.e., does not do anything with detections that only appear in one file.

    Args:
        json_filename_a (str or dict): filename containing results from the first model to be compared;
            should refer to the same images as [json_filename_b].  Can also be a loaded results dict.
        json_filename_b (str or dict): filename containing results from the second model to be compared;
            should refer to the same images as [json_filename_a].  Can also be a loaded results dict.
        json_filename_gt (str or dict, optional): filename containing ground truth; should refer to the
            same images as [json_filename_a] and [json_filename_b].  Can also be a loaded results dict.
            Should be in COCO format.
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

    # Load ground truth, if supplied
    gt_data = None

    if json_filename_gt is not None:
        if isinstance(json_filename_gt,str):
            gt_data = validate_batch_results(json_filename_gt,
                                             options=validation_options)
        else:
            assert isinstance(json_filename_gt,dict)
            gt_data = json_filename_gt

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


    ## Load ground truth if necessary

    gt_category_name_to_id = None
    gt_image_id_to_annotations = None
    image_filename_to_gt_im = None

    if gt_data is not None:

        gt_category_name_to_id = {}
        for c in gt_data['categories']:
            gt_category_name_to_id[c['name']] = c['id']

        image_filename_to_gt_im = {}
        for im in gt_data['images']:
            assert 'width' in im and 'height' in im, \
                'I can only compare against GT that has "width" and "height" fields'
            image_filename_to_gt_im[im['file_name']] = im

        assert set(image_filename_to_gt_im.keys()) == set(image_filenames_a), \
            'Ground truth filename list does not match image filename list'

        gt_image_id_to_annotations = defaultdict(list)
        for ann in gt_data['annotations']:
            gt_image_id_to_annotations[ann['image_id']].append(ann)


    ## Compare detections

    image_filename_b_to_im = {}
    for im in results_b['images']:
        image_filename_b_to_im[im['file']] = im

    n_detections_a = 0
    n_detections_a_queried = 0
    n_detections_a_matched = 0

    calibration_matches = []

    # For each image
    # im_a = results_a['images'][0]
    for i_image,im_a in tqdm(enumerate(results_a['images']),total=len(results_a['images'])):

        fn = im_a['file']
        im_b = image_filename_b_to_im[fn]

        if 'detections' not in im_a or im_a['detections'] is None:
            continue
        if 'detections' not in im_b or im_b['detections'] is None:
            continue

        im_gt = None
        if gt_data is not None:
            im_gt = image_filename_to_gt_im[fn]

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
            best_bbox_b = None

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
                    best_bbox_b = bbox_b

            # ...for each detection in im_b

            # If we found a match between A and B
            if best_iou is not None:

                n_detections_a_matched += 1

                # Does this pair of matched detections also match a ground truth box?
                matches_gt = None

                if im_gt is not None:

                    def max_iou_between_detection_and_gt(detection_box,category_name,im_gt,gt_annotations):

                        max_iou = None

                        # Which category ID are we looking for?
                        gt_category_id_for_detected_category_name = \
                            gt_category_name_to_id[category_name]

                        # For each GT annotation
                        #
                        # ann = gt_annotations[0]
                        for ann in gt_annotations:

                            # Only match against boxes in the same category
                            if ann['category_id'] != gt_category_id_for_detected_category_name:
                                continue
                            if 'bbox' not in ann:
                                continue

                            # Normalize this box
                            #
                            # COCO format: [x,y,width,height]
                            # normalized format: [x_min, y_min, width_of_box, height_of_box]
                            normalized_gt_box = [ann['bbox'][0]/im_gt['width'],ann['bbox'][1]/im_gt['height'],
                                                 ann['bbox'][2]/im_gt['width'],ann['bbox'][3]/im_gt['height']]

                            iou = get_iou(detection_box, normalized_gt_box)
                            if max_iou is None or iou > max_iou:
                                max_iou = iou

                        # ...for each gt box

                        return max_iou

                    # ...def min_iou_between_detections_and_gt(...)

                    gt_annotations = gt_image_id_to_annotations[im_gt['id']]

                    # If they matched, the A and B boxes have the same category by definition
                    category_name = categories_a[det_a['category']]

                    max_iou_with_bbox_a = \
                        max_iou_between_detection_and_gt(bbox_a,category_name,im_gt,gt_annotations)
                    max_iou_with_bbox_b = \
                        max_iou_between_detection_and_gt(best_bbox_b,category_name,im_gt,gt_annotations)

                    max_iou_with_either_detection_set = max_none(max_iou_with_bbox_a,
                                                                 max_iou_with_bbox_b)

                    matches_gt = False
                    if (max_iou_with_either_detection_set is not None) and \
                       (max_iou_with_either_detection_set >= options.iou_threshold):
                        matches_gt = True

                # ...if we have ground truth

                conf_result = [conf_a,best_iou_conf,best_iou,i_image,category_id,matches_gt]
                calibration_matches.append(conf_result)

            # ...if we had a match between A and B
        # ...for each detection in im_a

    # ...for each image in result set A

    if options.verbose:

        print('\nOf {} detections in result set A, queried {}, matched {}'.format(
            n_detections_a,n_detections_a_queried,n_detections_a_matched))

        if gt_data is not None:
            n_matches = 0
            for m in calibration_matches:
               assert m[CalibrationMatchColumns.COLUMN_MATCHES_GT] is not None
               if m[CalibrationMatchColumns.COLUMN_MATCHES_GT]:
                   n_matches += 1
            print('{} matches also matched ground truth'.format(n_matches))

    assert len(calibration_matches) == n_detections_a_matched

    calibration_results = CalibrationResults()
    calibration_results.calibration_matches = calibration_matches

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

    assert not (options.separate_plots_by_category and \
                options.separate_plots_by_correctness), \
        'separate_plots_by_category and separate_plots_by_correctness are mutually exclusive'

    category_id_to_name = None
    category_to_samples = None

    calibration_matches = calibration_results.calibration_matches

    # If we're just lumping everything into one plot
    if (not options.separate_plots_by_category) and (not options.separate_plots_by_correctness):

        category_id_to_name = {'0':'all_categories'}
        category_to_samples = {'0': []}

        # Make everything category "0" (arbitrary)
        calibration_matches = copy.deepcopy(calibration_matches)
        for m in calibration_matches:
            m[CalibrationMatchColumns.COLUMN_CATEGORY_ID] = '0'
        if (options.max_samples_per_category is not None) and \
           (len(calibration_matches) > options.max_samples_per_category):
            calibration_matches = \
                random.sample(calibration_matches,options.max_samples_per_category)
        category_to_samples['0'] = calibration_matches

    # If we're separating into lines for FPs and TPs (but not separating by category)
    elif options.separate_plots_by_correctness:

        assert not options.separate_plots_by_category

        category_id_tp = '0'
        category_id_fp = '1'

        category_id_to_name = {category_id_tp:'TP', category_id_fp:'FP'}
        category_to_samples = {category_id_tp: [], category_id_fp: []}

        for m in calibration_matches:
            assert m[CalibrationMatchColumns.COLUMN_MATCHES_GT] is not None, \
                "Can't plot by correctness when GT status is not available for every match"
            if m[CalibrationMatchColumns.COLUMN_MATCHES_GT]:
                category_to_samples[category_id_tp].append(m)
            else:
                category_to_samples[category_id_fp].append(m)

    # If we're separating by category
    else:

        assert options.separate_plots_by_category

        category_to_samples = defaultdict(list)

        category_to_matches = defaultdict(list)
        for m in calibration_matches:
            category_id = m[CalibrationMatchColumns.COLUMN_CATEGORY_ID]
            category_to_matches[category_id].append(m)

        category_id_to_name = None
        if options.category_id_to_name is not None:
            category_id_to_name = options.category_id_to_name

        for i_category,category_id in enumerate(category_to_matches.keys()):

            matches_this_category = category_to_matches[category_id]

            if (options.max_samples_per_category is None) or \
                (len(matches_this_category) <= options.max_samples_per_category):
                category_to_samples[category_id] = matches_this_category
            else:
                assert len(matches_this_category) > options.max_samples_per_category
                category_to_samples[category_id] = random.sample(matches_this_category,options.max_samples_per_category)

        del category_to_matches

    del calibration_matches

    if options.verbose:
        n_samples_for_histogram = 0
        for c in category_to_samples:
            n_samples_for_histogram += len(category_to_samples[c])
        print('Creating a histogram based on {} samples'.format(n_samples_for_histogram))

    categories_to_plot = list(category_to_samples.keys())

    if options.categories_to_plot is not None:
        categories_to_plot = [category_id for category_id in categories_to_plot if\
                              category_id in options.categories_to_plot]

    n_subplots = len(categories_to_plot)

    plt.ioff()

    fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
    # fig,axes = plt.subplots(nrows=n_subplots,ncols=1)

    axes = fig.subplots(n_subplots, 1)

    if not is_iterable(axes):
        assert n_subplots == 1
        axes = [axes]

    # i_category = 0; category_id = categories_to_plot[i_category]
    for i_category,category_id in enumerate(categories_to_plot):

        ax = axes[i_category]

        category_string = str(category_id)
        if (category_id_to_name is not None) and (category_id in category_id_to_name):
            category_string = category_id_to_name[category_id]

        samples_this_category = category_to_samples[category_id]
        x = [m[0] for m in samples_this_category]
        y = [m[1] for m in samples_this_category]

        weights_a = np.ones_like(x)/float(len(x))
        weights_b = np.ones_like(y)/float(len(y))

        # Plot the first lie a little thicker so the second line will always show up
        ax.hist(x,histtype='step',bins=n_hist_bins,density=False,color='red',weights=weights_a,linewidth=3.0)
        ax.hist(y,histtype='step',bins=n_hist_bins,density=False,color='blue',weights=weights_b,linewidth=1.5)

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

