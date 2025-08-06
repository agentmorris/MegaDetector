"""

compare_batch_results.py

Compare sets of batch results; typically used to compare:

* Results from different MegaDetector versions
* Results before/after RDE
* Results with/without augmentation

Makes pairwise comparisons between sets of results, but can take lists of results files
(will perform all pairwise comparisons).  Results are written to an HTML page that shows the
number and nature of disagreements (in the sense of each image being a detection or non-detection),
with sample images for each category.

Operates in one of three modes, depending on whether ground truth labels/boxes are available:

* The most common mode assumes no ground truth, just finds agreement/disagreement between
  results files, or class discrepancies.

* If image-level ground truth is available, finds image-level agreements on TPs/TNs/FPs/FNs, but also
  finds image-level TPs/TNs/FPs/FNs that are unique to each set of results (at the specified confidence
  threshold).

* If box-level ground truth is available, finds box-level agreements on TPs/TNs/FPs/FNs, but also finds
  image-level TPs/TNs/FPs/FNs that are unique to each set of results (at the specified confidence
  threshold).

"""

#%% Imports

import json
import os
import re
import random
import copy
import urllib
import itertools
import sys
import argparse
import textwrap

import numpy as np

from tqdm import tqdm
from functools import partial
from collections import defaultdict

from PIL import ImageFont, ImageDraw

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

from megadetector.visualization import visualization_utils
from megadetector.utils.write_html_image_list import write_html_image_list
from megadetector.utils.ct_utils import invert_dictionary, get_iou
from megadetector.utils import path_utils
from megadetector.visualization.visualization_utils import get_text_size

def _maxempty(L): # noqa
    """
    Return the maximum value in a list, or 0 if the list is empty
    """

    if len(L) == 0:
        return 0
    else:
        return max(L)


#%% Constants and support classes

class PairwiseBatchComparisonOptions:
    """
    Defines the options used for a single pairwise comparison; a list of these
    pairwise options sets is stored in the BatchComparisonsOptions class.
    """

    def __init__(self):

        #: First filename to compare
        self.results_filename_a = None

        #: Second filename to compare
        self.results_filename_b = None

        #: Description to use in the output HTML for filename A
        self.results_description_a = None

        #: Description to use in the output HTML for filename B
        self.results_description_b = None

        #: Per-class detection thresholds to use for filename A (including a 'default' threshold)
        self.detection_thresholds_a = {'animal':0.15,'person':0.15,'vehicle':0.15,'default':0.15}

        #: Per-class detection thresholds to use for filename B (including a 'default' threshold)
        self.detection_thresholds_b = {'animal':0.15,'person':0.15,'vehicle':0.15,'default':0.15}

        #: Rendering threshold to use for all categories for filename A
        self.rendering_confidence_threshold_a = 0.1

        #: Rendering threshold to use for all categories for filename B
        self.rendering_confidence_threshold_b = 0.1

# ...class PairwiseBatchComparisonOptions


class BatchComparisonOptions:
    """
    Defines the options for a set of (possibly many) pairwise comparisons.
    """

    def __init__(self):

        #: Folder to which we should write HTML output
        self.output_folder = None

        #: Base folder for images (which are specified as relative files)
        self.image_folder = None

        #: Job name to use in the HTML output file
        self.job_name = ''

        #: Maximum number of images to render for each category, where a "category" here is
        #: "detections_a_only", "detections_b_only", etc., or None to render all images.
        self.max_images_per_category = 1000

        #: Maximum number of images per HTML page (paginates if a category page goes beyond this),
        #: or None to disable pagination.
        self.max_images_per_page = None

        #: Colormap to use for detections in file A (maps detection categories to colors)
        self.colormap_a = ['Red']

        #: Colormap to use for detections in file B (maps detection categories to colors)
        self.colormap_b = ['RoyalBlue']

        #: Process-based parallelization isn't supported yet; this must be "True"
        self.parallelize_rendering_with_threads = True

        #: List of filenames to include in the comparison, or None to use all files
        self.filenames_to_include = None

        #: List of category names to include in the comparison, or None to use all categories
        self.category_names_to_include = None

        #: Compare only detections/non-detections, ignore categories (still renders categories)
        self.class_agnostic_comparison = False

        #: Width of images to render in the output HTML
        self.target_width = 800

        #: Number of workers to use for rendering, or <=1 to disable parallelization
        self.n_rendering_workers = 20

        #: Random seed for image sampling (not used if max_images_per_category is None)
        self.random_seed = 0

        #: Whether to sort results by confidence; if this is False, sorts by filename
        self.sort_by_confidence = False

        #: The expectation is that all results sets being compared will refer to the same images; if this
        #: is True (default), we'll error if that's not the case, otherwise non-matching lists will just be
        #: a warning.
        self.error_on_non_matching_lists = True

        #: Ground truth .json file in COCO Camera Traps format, or an already-loaded COCO dictionary
        self.ground_truth_file = None

        #: IoU threshold to use when comparing to ground truth with boxes
        self.gt_iou_threshold = 0.5

        #: Category names that refer to empty images when image-level ground truth is provided
        self.gt_empty_categories = ['empty','blank','misfire']

        #: Should we show image-level labels as text on each image when boxes are not available?
        self.show_labels_for_image_level_gt = True

        #: Should we show category names (instead of numbers) on GT boxes?
        self.show_category_names_on_gt_boxes = True

        #: Should we show category names (instead of numbers) on detected boxes?
        self.show_category_names_on_detected_boxes = True

        #: List of PairwiseBatchComparisonOptions that defines the comparisons we'll render.
        self.pairwise_options = []

        #: Only process images whose file names contain this token
        #:
        #: This can also be a pointer to a function that takes a string (filename)
        #: and returns a bool (if the function returns True, the image will be
        #: included in the comparison).
        self.required_token = None

        #: Enable additional debug output
        self.verbose = False

        #: Separate out the "clean TP" and "clean TN" categories, only relevant when GT is
        #: available.
        self.include_clean_categories = True

        #: When rendering to the output table, optionally write alternative strings
        #: to describe images
        self.fn_to_display_fn = None

        #: Should we run urllib.parse.quote() on paths before using them as links in the
        #: output page?
        self.parse_link_paths = True

        #: Should we include a TOC?  TOC is always omitted if <=2 comparisons are performed.
        self.include_toc = True

# ...class BatchComparisonOptions


class PairwiseBatchComparisonResults:
    """
    The results from a single pairwise comparison.
    """

    def __init__(self):

        #: String of HTML content suitable for rendering to an HTML file
        self.html_content = None

        #: Possibly-modified version of the PairwiseBatchComparisonOptions supplied as input.
        self.pairwise_options = None

        #: A dictionary with keys representing category names; in the no-ground-truth case, for example,
        #: category names are:
        #:
        #: common_detections
        #: common_non_detections
        #: detections_a_only
        #: detections_b_only
        #: class_transitions
        #
        #: Values are dicts with fields 'im_a', 'im_b', 'sort_conf', and 'im_gt'
        self.categories_to_image_pairs = None

        #: Short identifier for this comparison
        self.comparison_short_name = None

        #: Friendly identifier for this comparison
        self.comparison_friendly_name = None

# ...class PairwiseBatchComparisonResults


class BatchComparisonResults:
    """
    The results from a set of pairwise comparisons
    """

    def __init__(self):

        #: Filename containing HTML output
        self.html_output_file = None

        #: A list of PairwiseBatchComparisonResults
        self.pairwise_results = None

# ...class BatchComparisonResults


main_page_style_header = """<head><title>Results comparison</title>
    <style type="text/css">
    a { text-decoration: none; }
    body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
    div.contentdiv { margin-left: 20px; }
    </style>
    </head>"""

main_page_header = '<html>\n{}\n<body>\n'.format(main_page_style_header)
main_page_footer = '<br/><br/><br/></body></html>\n'


#%% Comparison functions

def _render_image_pair(fn,image_pairs,category_folder,options,pairwise_options):
    """
    Render two sets of results (i.e., a comparison) for a single image.

    Args:
        fn (str): image filename
        image_pairs (dict): dict mapping filenames to pairs of image dicts
        category_folder (str): folder to which to render this image, typically
            "detections_a_only", "detections_b_only", etc.
        options (BatchComparisonOptions): job options
        pairwise_options (PairwiseBatchComparisonOptions): pairwise comparison options

    Returns:
        str: rendered image filename
    """

    input_image_path = os.path.join(options.image_folder,fn)
    assert os.path.isfile(input_image_path), 'Image {} does not exist'.format(input_image_path)

    im = visualization_utils.open_image(input_image_path)
    image_pair = image_pairs[fn]
    detections_a = image_pair['im_a']['detections']
    detections_b = image_pair['im_b']['detections']

    custom_strings_a = [''] * len(detections_a)
    custom_strings_b = [''] * len(detections_b)

    # This function is often used to compare results before/after various merging
    # steps, so we have some special-case formatting based on the "transferred_from"
    # field generated in merge_detections.py.
    for i_det,det in enumerate(detections_a):
        if 'transferred_from' in det:
            custom_strings_a[i_det] = '({})'.format(
                det['transferred_from'].split('.')[0])

    for i_det,det in enumerate(detections_b):
        if 'transferred_from' in det:
            custom_strings_b[i_det] = '({})'.format(
                det['transferred_from'].split('.')[0])

    if options.target_width is not None:
        im = visualization_utils.resize_image(im, options.target_width)

    label_map = None
    if options.show_category_names_on_detected_boxes:
        label_map=options.detection_category_id_to_name

    visualization_utils.render_detection_bounding_boxes(detections_a,im,
        confidence_threshold=pairwise_options.rendering_confidence_threshold_a,
        thickness=4,expansion=0,
        label_map=label_map,
        colormap=options.colormap_a,
        textalign=visualization_utils.TEXTALIGN_LEFT,
        vtextalign=visualization_utils.VTEXTALIGN_TOP,
        custom_strings=custom_strings_a)
    visualization_utils.render_detection_bounding_boxes(detections_b,im,
        confidence_threshold=pairwise_options.rendering_confidence_threshold_b,
        thickness=2,expansion=0,
        label_map=label_map,
        colormap=options.colormap_b,
        textalign=visualization_utils.TEXTALIGN_LEFT,
        vtextalign=visualization_utils.VTEXTALIGN_BOTTOM,
        custom_strings=custom_strings_b)

    # Do we also need to render ground truth?
    if 'im_gt' in image_pair and image_pair['im_gt'] is not None:

        im_gt = image_pair['im_gt']
        annotations_gt = image_pair['annotations_gt']
        gt_boxes = []
        for ann in annotations_gt:
            if 'bbox' in ann:
                gt_boxes.append(ann['bbox'])
        gt_categories = [ann['category_id'] for ann in annotations_gt]

        if len(gt_boxes) > 0:

            label_map = None
            if options.show_category_names_on_gt_boxes:
                label_map=options.gt_category_id_to_name

            assert len(gt_boxes) == len(gt_categories)
            gt_colormap = ['yellow']*(max(gt_categories)+1)
            visualization_utils.render_db_bounding_boxes(boxes=gt_boxes,
                                                         classes=gt_categories,
                                                         image=im,
                                                         original_size=(im_gt['width'],im_gt['height']),
                                                         label_map=label_map,
                                                         thickness=1,
                                                         expansion=0,
                                                         textalign=visualization_utils.TEXTALIGN_RIGHT,
                                                         vtextalign=visualization_utils.VTEXTALIGN_TOP,
                                                         text_rotation=-90,
                                                         colormap=gt_colormap)

        else:

            if options.show_labels_for_image_level_gt:

                gt_categories_set = set([ann['category_id'] for ann in annotations_gt])
                gt_category_names = [options.gt_category_id_to_name[category_name] for
                                     category_name in gt_categories_set]
                category_string = ','.join(gt_category_names)
                category_string = '(' + category_string + ')'

                try:
                    font = ImageFont.truetype('arial.ttf', 25)
                except OSError:
                    font = ImageFont.load_default()

                draw = ImageDraw.Draw(im)

                text_width, text_height = get_text_size(font,category_string)

                text_left = 10
                text_bottom = text_height + 10
                margin = np.ceil(0.05 * text_height)

                draw.text(
                    (text_left + margin, text_bottom - text_height - margin),
                    category_string,
                    fill='white',
                    font=font)

        # ...if we have boxes in the GT

    # ...if we need to render ground truth

    output_image_fn = path_utils.flatten_path(fn)
    output_image_path = os.path.join(category_folder,output_image_fn)
    im.save(output_image_path)
    return output_image_path

# ...def _render_image_pair()


def _result_types_to_comparison_category(result_types_present_a,
                                         result_types_present_b,
                                         ground_truth_type,
                                         options):
    """
    Given the set of result types (tp,tn,fp,fn) present in each of two sets of results
    for an image, determine the category to which we want to assign this image.
    """

    # The "common_tp" category is for the case where both models have *only* TPs
    if ('tp' in result_types_present_a) and ('tp' in result_types_present_b) and \
        (len(result_types_present_a) == 1) and (len(result_types_present_b) == 1):
        return 'common_tp'

    # The "common_tn" category is for the case where both models have *only* TNs
    if ('tn' in result_types_present_a) and ('tn' in result_types_present_b) and \
        (len(result_types_present_a) == 1) and (len(result_types_present_b) == 1):
        return 'common_tn'

    """
    # The "common_fp" category is for the case where both models have *only* FPs
    if ('fp' in result_types_present_a) and ('fp' in result_types_present_b) and \
         (len(result_types_present_a) == 1) and (len(result_types_present_b) == 1):
        return 'common_fp'
    """

    # The "common_fp" category is for the case where both models have at least one FP,
    # and no FNs.
    if ('fp' in result_types_present_a) and ('fp' in result_types_present_b) and \
         ('fn' not in result_types_present_a) and ('fn' not in result_types_present_b):
        return 'common_fp'

    """
    # The "common_fn" category is for the case where both models have *only* FNs
    if ('fn' in result_types_present_a) and ('fn' in result_types_present_b) and \
         (len(result_types_present_a) == 1) and (len(result_types_present_b) == 1):
        return 'common_fn'
    """

    # The "common_fn" category is for the case where both models have at least one FN,
    # and no FPs
    if ('fn' in result_types_present_a) and ('fn' in result_types_present_b) and \
        ('fp' not in result_types_present_a) and ('fp' not in result_types_present_b):
        return 'common_fn'

    ## The tp-only categories are for the case where one model has *only* TPs

    if ('tp' in result_types_present_a) and (len(result_types_present_a) == 1):
        # Clean TPs are cases where the other model has only FNs, no FPs
        if options.include_clean_categories:
            if  ('fn' in result_types_present_b) and \
                ('fp' not in result_types_present_b) and \
                ('tp' not in result_types_present_b):
                return 'clean_tp_a_only'
        # Otherwise, TPs are cases where one model has only TPs, and the other model
        # has any mistakse
        if ('fn' in result_types_present_b) or ('fp' in result_types_present_b):
            return 'tp_a_only'

    if ('tp' in result_types_present_b) and (len(result_types_present_b) == 1):
        # Clean TPs are cases where the other model has only FNs, no FPs
        if options.include_clean_categories:
            if  ('fn' in result_types_present_a) and \
                ('fp' not in result_types_present_a) and \
                ('tp' not in result_types_present_a):
                return 'clean_tp_b_only'
        # Otherwise, TPs are cases where one model has only TPs, and the other model
        # has any mistakse
        if ('fn' in result_types_present_a) or ('fp' in result_types_present_a):
            return 'tp_b_only'

    # The tn-only categories are for the case where one model has a TN and the
    # other has at least one fp
    if 'tn' in result_types_present_a and 'fp' in result_types_present_b:
        assert len(result_types_present_a) == 1
        assert len(result_types_present_b) == 1
        return 'tn_a_only'
    if 'tn' in result_types_present_b and 'fp' in result_types_present_a:
        assert len(result_types_present_a) == 1
        assert len(result_types_present_b) == 1
        return 'tn_b_only'

    # The 'fpfn' category is for everything else
    return 'fpfn'

# ...def _result_types_to_comparison_category(...)


def _subset_md_results(results,options):
    """
    Subset a set of MegaDetector results according to the rules defined in the
    BatchComparisonOptions object [options].  Typically used to filter for files
    containing a particular string.  Modifies [results] in place, also returns.

    Args:
        results (dict): MD results
        options (BatchComparisonOptions): job options containing filtering rules
    """

    if options.required_token is None:
        return results

    images_to_keep = []
    for im in results['images']:
        # Is [required_token] a string?
        if isinstance(options.required_token,str):
            if options.required_token in im['file']:
                images_to_keep.append(im)
        # Otherwise [required_token] is a function
        else:
            assert callable(options.required_token), 'Illegal value for required_token'
            if options.required_token(im['file']):
                images_to_keep.append(im)


    if options.verbose:
        print('Keeping {} of {} images in MD results'.format(
            len(images_to_keep),len(results['images'])))

    results['images'] = images_to_keep
    return results

# ...def _subset_md_results(...)


def _subset_ground_truth(gt_data,options):
    """
    Subset a set of COCO annotations according to the rules defined in the
    BatchComparisonOptions object [options].  Typically used to filter for files
    containing a particular string.  Modifies [results] in place, also returns.

    Args:
        gt_data (dict): COCO-formatted annotations
        options (BatchComparisonOptions): job options containing filtering rules
    """

    if options.required_token is None:
        return gt_data

    images_to_keep = []
    for im in gt_data['images']:
        if isinstance(options.required_token,str):
            if options.required_token in im['file_name']:
                images_to_keep.append(im)
        else:
            if options.required_token(im['file_name']):
                images_to_keep.append(im)

    image_ids_to_keep_set = set([im['id'] for im in images_to_keep])

    annotations_to_keep = []
    for ann in gt_data['annotations']:
        if ann['image_id'] in image_ids_to_keep_set:
            annotations_to_keep.append(ann)

    if options.verbose:
        print('Keeping {} of {} images, {} of {} annotations in GT data'.format(
            len(images_to_keep),len(gt_data['images']),
            len(annotations_to_keep),len(gt_data['annotations'])))

    gt_data['images'] = images_to_keep
    gt_data['annotations'] = annotations_to_keep

    return gt_data

# ...def _subset_ground_truth(...)


def _pairwise_compare_batch_results(options,output_index,pairwise_options):
    """
    The main entry point for this module is compare_batch_results(), which calls
    this function for each pair of comparisons the caller has requested.  Generates an
    HTML page for this comparison.  Returns a BatchComparisonResults object.

    Args:
        options (BatchComparisonOptions): overall job options for this comparison group
        output_index (int): a numeric index used for generating HTML titles
        pairwise_options (PairwiseBatchComparisonOptions): job options for this comparison

    Returns:
        PairwiseBatchComparisonResults: the results of this pairwise comparison
    """

    # pairwise_options is passed as a parameter here, and should not be specified
    # in the options object.
    assert options.pairwise_options is None

    if options.random_seed is not None:
        random.seed(options.random_seed)

    # Warn the user if some "detections" might not get rendered
    max_classification_threshold_a = max(list(pairwise_options.detection_thresholds_a.values()))
    max_classification_threshold_b = max(list(pairwise_options.detection_thresholds_b.values()))

    if pairwise_options.rendering_confidence_threshold_a > max_classification_threshold_a:
        print('*** Warning: rendering threshold A ({}) is higher than max confidence threshold A ({}) ***'.format(
            pairwise_options.rendering_confidence_threshold_a,max_classification_threshold_a))

    if pairwise_options.rendering_confidence_threshold_b > max_classification_threshold_b:
        print('*** Warning: rendering threshold B ({}) is higher than max confidence threshold B ({}) ***'.format(
            pairwise_options.rendering_confidence_threshold_b,max_classification_threshold_b))


    ##%% Validate inputs

    assert os.path.isfile(pairwise_options.results_filename_a), \
        "Can't find results file {}".format(pairwise_options.results_filename_a)
    assert os.path.isfile(pairwise_options.results_filename_b), \
        "Can't find results file {}".format(pairwise_options.results_filename_b)
    assert os.path.isdir(options.image_folder), \
        "Can't find image folder {}".format(options.image_folder)
    os.makedirs(options.output_folder,exist_ok=True)


    ##%% Load both result sets

    with open(pairwise_options.results_filename_a,'r') as f:
        results_a = json.load(f)

    with open(pairwise_options.results_filename_b,'r') as f:
        results_b = json.load(f)

    # Don't let path separators confuse things
    for im in results_a['images']:
        if 'file' in im:
            im['file'] = im['file'].replace('\\','/')
    for im in results_b['images']:
        if 'file' in im:
            im['file'] = im['file'].replace('\\','/')

    if not options.class_agnostic_comparison:
        assert results_a['detection_categories'] == results_b['detection_categories'], \
            "Cannot perform a class-sensitive comparison across results with different categories"

    detection_categories_a = results_a['detection_categories']
    detection_categories_b = results_b['detection_categories']
    detection_category_id_to_name = detection_categories_a
    detection_category_name_to_id = invert_dictionary(detection_categories_a)
    options.detection_category_id_to_name = detection_category_id_to_name

    if pairwise_options.results_description_a is None:
        if 'detector' not in results_a['info']:
            print('No model metadata supplied for results-A, assuming MDv4')
            pairwise_options.results_description_a = 'MDv4 (assumed)'
        else:
            pairwise_options.results_description_a = results_a['info']['detector']

    if pairwise_options.results_description_b is None:
        if 'detector' not in results_b['info']:
            print('No model metadata supplied for results-B, assuming MDv4')
            pairwise_options.results_description_b = 'MDv4 (assumed)'
        else:
            pairwise_options.results_description_b = results_b['info']['detector']

    # Restrict this comparison to specific files if requested
    results_a = _subset_md_results(results_a, options)
    results_b = _subset_md_results(results_b, options)

    images_a = results_a['images']
    images_b = results_b['images']

    filename_to_image_a = {im['file']:im for im in images_a}
    filename_to_image_b = {im['file']:im for im in images_b}


    ##%% Make sure they represent the same set of images

    filenames_a = [im['file'] for im in images_a]
    filenames_b_set = set([im['file'] for im in images_b])

    if len(images_a) != len(images_b):
        s = 'set A has {} images, set B has {}'.format(len(images_a),len(images_b))
        if options.error_on_non_matching_lists:
            raise ValueError(s)
        else:
            print('Warning: ' + s)
    else:
        if options.error_on_non_matching_lists:
            for fn in filenames_a:
                assert fn in filenames_b_set

    assert len(filenames_a) == len(images_a)
    assert len(filenames_b_set) == len(images_b)

    if options.filenames_to_include is None:
        filenames_to_compare = filenames_a
    else:
        filenames_to_compare = options.filenames_to_include


    ##%% Determine whether ground truth is available

    # ...and determine what type of GT is available, boxes or image-level labels

    gt_data = None
    gt_category_id_to_detection_category_id = None

    if options.ground_truth_file is None:

        ground_truth_type = 'no_gt'

    else:

        # Read ground truth data if necessary
        if isinstance(options.ground_truth_file,dict):
            gt_data = options.ground_truth_file
        else:
            assert isinstance(options.ground_truth_file,str)
            with open(options.ground_truth_file,'r') as f:
                gt_data = json.load(f)

        # Restrict this comparison to specific files if requested
        gt_data = _subset_ground_truth(gt_data, options)

        # Do we have box-level ground truth or image-level ground truth?
        found_box = False

        for ann in gt_data['annotations']:
            if 'bbox' in ann:
                found_box = True
                break

        if found_box:
            ground_truth_type = 'bbox_gt'
        else:
            ground_truth_type = 'image_level_gt'

        gt_category_name_to_id = {c['name']:c['id'] for c in gt_data['categories']}
        gt_category_id_to_name = invert_dictionary(gt_category_name_to_id)
        options.gt_category_id_to_name = gt_category_id_to_name

        if ground_truth_type == 'bbox_gt':

            if not options.class_agnostic_comparison:
                assert set(gt_category_name_to_id.keys()) == set(detection_category_name_to_id.keys()), \
                    'Cannot compare detections to GT with different categories when class_agnostic_comparison is False'
                gt_category_id_to_detection_category_id = {}
                for category_name in gt_category_name_to_id:
                    gt_category_id = gt_category_name_to_id[category_name]
                    detection_category_id = detection_category_name_to_id[category_name]
                    gt_category_id_to_detection_category_id[gt_category_id] = detection_category_id

        elif ground_truth_type == 'image_level_gt':

            if not options.class_agnostic_comparison:
                for detection_category_name in detection_category_name_to_id:
                    if detection_category_name not in gt_category_name_to_id:
                        raise ValueError('Detection category {} not available in GT category list'.format(
                            detection_category_name))
                for gt_category_name in gt_category_name_to_id:
                    if gt_category_name in options.gt_empty_categories:
                        continue
                    if (gt_category_name not in detection_category_name_to_id):
                        raise ValueError('GT category {} not available in detection category list'.format(
                            gt_category_name))

    assert ground_truth_type in ('no_gt','bbox_gt','image_level_gt')

    # Make sure ground truth data refers to at least *some* of the same files that are in our
    # results files
    if gt_data is not None:

        filenames_to_compare_set = set(filenames_to_compare)
        gt_filenames = [im['file_name'] for im in gt_data['images']]
        gt_filenames_set = set(gt_filenames)

        common_filenames = filenames_to_compare_set.intersection(gt_filenames_set)
        assert len(common_filenames) > 0, 'MD results files and ground truth file have no images in common'

        filenames_only_in_gt = gt_filenames_set.difference(filenames_to_compare_set)
        if len(filenames_only_in_gt) > 0:
            print('Warning: {} files are only available in the ground truth (not in MD results)'.format(
                len(filenames_only_in_gt)))

        filenames_only_in_results = gt_filenames_set.difference(gt_filenames)
        if len(filenames_only_in_results) > 0:
            print('Warning: {} files are only available in the MD results (not in ground truth)'.format(
                len(filenames_only_in_results)))

        if options.error_on_non_matching_lists:
            if len(filenames_only_in_gt) > 0 or len(filenames_only_in_results) > 0:
               raise ValueError('GT image set is not identical to result image sets')

        filenames_to_compare = sorted(list(common_filenames))

        # Map filenames to ground truth images and annotations
        filename_to_image_gt = {im['file_name']:im for im in gt_data['images']}
        gt_image_id_to_image = {}
        for im in gt_data['images']:
            gt_image_id_to_image[im['id']] = im
        gt_image_id_to_annotations = defaultdict(list)
        for ann in gt_data['annotations']:
            gt_image_id_to_annotations[ann['image_id']].append(ann)

        # Convert annotations to relative (MD) coordinates

        # ann = gt_data['annotations'][0]
        for ann in gt_data['annotations']:
            gt_image = gt_image_id_to_image[ann['image_id']]
            if 'bbox' not in ann:
                continue
            # COCO format: [x,y,width,height]
            # normalized format: [x_min, y_min, width_of_box, height_of_box]
            normalized_bbox = [ann['bbox'][0]/gt_image['width'],ann['bbox'][1]/gt_image['height'],
                               ann['bbox'][2]/gt_image['width'],ann['bbox'][3]/gt_image['height']]
            ann['normalized_bbox'] = normalized_bbox


    ##%% Find differences

    # See PairwiseBatchComparisonResults for a description
    categories_to_image_pairs = {}

    # This will map category names that can be used in filenames (e.g. "common_non_detections" or
    # "false_positives_a_only" to friendly names (e.g. "Common non-detections")
    categories_to_page_titles = None

    if ground_truth_type == 'no_gt':

        categories_to_image_pairs['common_detections'] = {}
        categories_to_image_pairs['common_non_detections'] = {}
        categories_to_image_pairs['detections_a_only'] = {}
        categories_to_image_pairs['detections_b_only'] = {}
        categories_to_image_pairs['class_transitions'] = {}

        categories_to_page_titles = {
            'common_detections':'Detections common to both models',
            'common_non_detections':'Non-detections common to both models',
            'detections_a_only':'Detections reported by model A only',
            'detections_b_only':'Detections reported by model B only',
            'class_transitions':'Detections reported as different classes by models A and B'
        }


    elif (ground_truth_type == 'bbox_gt') or (ground_truth_type == 'image_level_gt'):

        categories_to_image_pairs['common_tp'] = {}
        categories_to_image_pairs['common_tn'] = {}
        categories_to_image_pairs['common_fp'] = {}
        categories_to_image_pairs['common_fn'] = {}

        categories_to_image_pairs['tp_a_only'] = {}
        categories_to_image_pairs['tp_b_only'] = {}
        categories_to_image_pairs['tn_a_only'] = {}
        categories_to_image_pairs['tn_b_only'] = {}

        categories_to_image_pairs['fpfn'] = {}

        categories_to_page_titles = {
            'common_tp':'Common true positives',
            'common_tn':'Common true negatives',
            'common_fp':'Common false positives',
            'common_fn':'Common false negatives',
            'tp_a_only':'TP (A only)',
            'tp_b_only':'TP (B only)',
            'tn_a_only':'TN (A only)',
            'tn_b_only':'TN (B only)',
            'fpfn':'More complicated discrepancies'
        }

        if options.include_clean_categories:

            categories_to_image_pairs['clean_tp_a_only'] = {}
            categories_to_image_pairs['clean_tp_b_only'] = {}
            # categories_to_image_pairs['clean_tn_a_only'] = {}
            # categories_to_image_pairs['clean_tn_b_only'] = {}

            categories_to_page_titles['clean_tp_a_only'] = 'Clean TP wins for A'
            categories_to_page_titles['clean_tp_b_only'] = 'Clean TP wins for B'
            # categories_to_page_titles['clean_tn_a_only'] = 'Clean TN wins for A'
            # categories_to_page_titles['clean_tn_b_only'] = 'Clean TN wins for B'


    else:

        raise Exception('Unknown ground truth type: {}'.format(ground_truth_type))

    # Map category IDs to thresholds
    category_id_to_threshold_a = {}
    category_id_to_threshold_b = {}

    for category_id in detection_categories_a:
        category_name = detection_categories_a[category_id]
        if category_name in pairwise_options.detection_thresholds_a:
            category_id_to_threshold_a[category_id] = \
                pairwise_options.detection_thresholds_a[category_name]
        else:
            category_id_to_threshold_a[category_id] = \
                pairwise_options.detection_thresholds_a['default']

    for category_id in detection_categories_b:
        category_name = detection_categories_b[category_id]
        if category_name in pairwise_options.detection_thresholds_b:
            category_id_to_threshold_b[category_id] = \
                pairwise_options.detection_thresholds_b[category_name]
        else:
            category_id_to_threshold_b[category_id] = \
                pairwise_options.detection_thresholds_b['default']

    # fn = filenames_to_compare[0]
    for i_file,fn in tqdm(enumerate(filenames_to_compare),total=len(filenames_to_compare)):

        if fn not in filename_to_image_b:

            # We shouldn't have gotten this far if error_on_non_matching_lists is set
            assert not options.error_on_non_matching_lists

            print('Skipping filename {}, not in image set B'.format(fn))
            continue

        im_a = filename_to_image_a[fn]
        im_b = filename_to_image_b[fn]

        im_pair = {}
        im_pair['im_a'] = im_a
        im_pair['im_b'] = im_b
        im_pair['im_gt'] = None
        im_pair['annotations_gt'] = None

        if gt_data is not None:

            if fn not in filename_to_image_gt:

                # We shouldn't have gotten this far if error_on_non_matching_lists is set
                assert not options.error_on_non_matching_lists

                print('Skipping filename {}, not in ground truth'.format(fn))
                continue

            im_gt = filename_to_image_gt[fn]
            annotations_gt = gt_image_id_to_annotations[im_gt['id']]
            im_pair['im_gt'] = im_gt
            im_pair['annotations_gt'] = annotations_gt

        comparison_category = None

        # Compare image A to image B, without ground truth
        if ground_truth_type == 'no_gt':

            categories_above_threshold_a = set()

            if 'detections' not in im_a or im_a['detections'] is None:
                assert 'failure' in im_a and im_a['failure'] is not None
                continue

            if 'detections' not in im_b or im_b['detections'] is None:
                assert 'failure' in im_b and im_b['failure'] is not None
                continue

            invalid_category_error = False

            # det = im_a['detections'][0]
            for det in im_a['detections']:

                category_id = det['category']

                if category_id not in category_id_to_threshold_a:
                    print('Warning: unexpected category {} for model A on file {}'.format(category_id,fn))
                    invalid_category_error = True
                    break

                conf = det['conf']
                conf_thresh = category_id_to_threshold_a[category_id]
                if conf >= conf_thresh:
                    categories_above_threshold_a.add(category_id)

            if invalid_category_error:
                continue

            categories_above_threshold_b = set()

            for det in im_b['detections']:

                category_id = det['category']

                if category_id not in category_id_to_threshold_b:
                    print('Warning: unexpected category {} for model B on file {}'.format(category_id,fn))
                    invalid_category_error = True
                    break

                conf = det['conf']
                conf_thresh = category_id_to_threshold_b[category_id]
                if conf >= conf_thresh:
                    categories_above_threshold_b.add(category_id)

            if invalid_category_error:

                continue

            # Should we be restricting the comparison to only certain categories?
            if options.category_names_to_include is not None:

                # Just in case the user provided a single category instead of a list
                if isinstance(options.category_names_to_include,str):
                    options.category_names_to_include = [options.category_names_to_include]

                category_name_to_id_a = invert_dictionary(detection_categories_a)
                category_name_to_id_b = invert_dictionary(detection_categories_b)
                category_ids_to_include_a = []
                category_ids_to_include_b = []

                for category_name in options.category_names_to_include:
                    if category_name in category_name_to_id_a:
                        category_ids_to_include_a.append(category_name_to_id_a[category_name])
                    if category_name in category_name_to_id_b:
                        category_ids_to_include_b.append(category_name_to_id_b[category_name])

                # Restrict the categories we treat as above-threshold to the set we're supposed
                # to be using
                categories_above_threshold_a = [category_id for category_id in categories_above_threshold_a if \
                                                category_id in category_ids_to_include_a]
                categories_above_threshold_b = [category_id for category_id in categories_above_threshold_b if \
                                                category_id in category_ids_to_include_b]

            detection_a = (len(categories_above_threshold_a) > 0)
            detection_b = (len(categories_above_threshold_b) > 0)

            if detection_a and detection_b:
                if (categories_above_threshold_a == categories_above_threshold_b) or \
                    options.class_agnostic_comparison:
                    comparison_category = 'common_detections'
                else:
                    comparison_category = 'class_transitions'
            elif (not detection_a) and (not detection_b):
                comparison_category = 'common_non_detections'
            elif detection_a and (not detection_b):
                comparison_category = 'detections_a_only'
            else:
                assert detection_b and (not detection_a)
                comparison_category = 'detections_b_only'

            max_conf_a = _maxempty([det['conf'] for det in im_a['detections']])
            max_conf_b = _maxempty([det['conf'] for det in im_b['detections']])

            # Only used if sort_by_confidence is True
            if comparison_category == 'common_detections':
                sort_conf = max(max_conf_a,max_conf_b)
            elif comparison_category == 'common_non_detections':
                sort_conf = max(max_conf_a,max_conf_b)
            elif comparison_category == 'detections_a_only':
                sort_conf = max_conf_a
            elif comparison_category == 'detections_b_only':
                sort_conf = max_conf_b
            elif comparison_category == 'class_transitions':
                sort_conf = max(max_conf_a,max_conf_b)
            else:
                print('Warning: unknown comparison category {}'.format(comparison_category))
                sort_conf = max(max_conf_a,max_conf_b)

        elif ground_truth_type == 'bbox_gt':

            def _boxes_match(det,gt_ann):

                # if we're doing class-sensitive comparisons, only match same-category classes
                if not options.class_agnostic_comparison:
                    detection_category_id = det['category']
                    gt_category_id = gt_ann['category_id']
                    if detection_category_id != \
                        gt_category_id_to_detection_category_id[gt_category_id]:
                        return False

                if 'bbox' not in gt_ann:
                    return False

                assert 'normalized_bbox' in gt_ann
                iou = get_iou(det['bbox'],gt_ann['normalized_bbox'])

                return iou >= options.gt_iou_threshold

            # ...def _boxes_match(...)

            # Categorize each model into TP/TN/FP/FN
            def _categorize_image_with_box_gt(im_detection,im_gt,annotations_gt,category_id_to_threshold):

                annotations_gt = [ann for ann in annotations_gt if 'bbox' in ann]

                assert im_detection['file'] == im_gt['file_name']

                # List of result types - tn, tp, fp, fn - present in this image.  tn is
                # mutually exclusive with the others.
                result_types_present = set()

                # Find detections above threshold
                detections_above_threshold = []

                # det = im_detection['detections'][0]
                for det in im_detection['detections']:
                    category_id = det['category']
                    threshold = category_id_to_threshold[category_id]
                    if det['conf'] > threshold:
                        detections_above_threshold.append(det)

                if len(detections_above_threshold) == 0 and len(annotations_gt) == 0:
                    result_types_present.add('tn')
                    return result_types_present

                # Look for a match for each detection
                #
                # det = detections_above_threshold[0]
                for det in detections_above_threshold:

                    det_matches_annotation = False

                    # gt_ann = annotations_gt[0]
                    for gt_ann in annotations_gt:
                        if _boxes_match(det, gt_ann):
                            det_matches_annotation = True
                            break

                    if det_matches_annotation:
                        result_types_present.add('tp')
                    else:
                        result_types_present.add('fp')

                # Look for a match for each GT bbox
                #
                # gt_ann = annotations_gt[0]
                for gt_ann in annotations_gt:

                    annotation_matches_det = False

                    for det in detections_above_threshold:

                        if _boxes_match(det, gt_ann):
                            annotation_matches_det = True
                            break

                    if annotation_matches_det:
                        # We should have found this when we looped over detections
                        assert 'tp' in result_types_present
                    else:
                        result_types_present.add('fn')

                # ...for each above-threshold detection

                return result_types_present

            # ...def _categorize_image_with_box_gt(...)

            # im_detection = im_a; category_id_to_threshold = category_id_to_threshold_a
            result_types_present_a = \
                _categorize_image_with_box_gt(im_a,im_gt,annotations_gt,category_id_to_threshold_a)
            result_types_present_b = \
                _categorize_image_with_box_gt(im_b,im_gt,annotations_gt,category_id_to_threshold_b)


            ## Some combinations are nonsense

            # TNs are mutually exclusive with other categories
            if 'tn' in result_types_present_a or 'tn' in result_types_present_b:
                assert len(result_types_present_a) == 1
                assert len(result_types_present_b) == 1

            # If either model has a TP or FN, the other has to have a TP or FN, since
            # there was something in the GT
            if ('tp' in result_types_present_a) or ('fn' in result_types_present_a):
                assert 'tp' in result_types_present_b or 'fn' in result_types_present_b
            if ('tp' in result_types_present_b) or ('fn' in result_types_present_b):
                assert 'tp' in result_types_present_a or 'fn' in result_types_present_a

            # If either model has a TP or FN, the other has to have a TP or FN, since
            # there was something in the GT
            if ('tp' in result_types_present_a) or ('fn' in result_types_present_a):
                assert 'tp' in result_types_present_b or 'fn' in result_types_present_b
            if ('tp' in result_types_present_b) or ('fn' in result_types_present_b):
                assert 'tp' in result_types_present_a or 'fn' in result_types_present_a


            ## Choose a comparison category based on result types

            comparison_category = _result_types_to_comparison_category(
                result_types_present_a,result_types_present_b,ground_truth_type,options)

            # TODO: this may or may not be the right way to interpret sorting
            # by confidence in this case, e.g., we may want to sort by confidence
            # of correct or incorrect matches.  But this isn't *wrong*.
            max_conf_a = _maxempty([det['conf'] for det in im_a['detections']])
            max_conf_b = _maxempty([det['conf'] for det in im_b['detections']])
            sort_conf = max(max_conf_a,max_conf_b)

        else:

            # Categorize each model into TP/TN/FP/FN
            def _categorize_image_with_image_level_gt(im_detection,im_gt,annotations_gt,
                                                      category_id_to_threshold):

                assert im_detection['file'] == im_gt['file_name']

                # List of result types - tn, tp, fp, fn - present in this image.
                result_types_present = set()

                # Find detections above threshold
                category_names_detected = set()

                # det = im_detection['detections'][0]
                for det in im_detection['detections']:
                    category_id = det['category']
                    threshold = category_id_to_threshold[category_id]
                    if det['conf'] > threshold:
                        category_name = detection_category_id_to_name[det['category']]
                        category_names_detected.add(category_name)

                category_names_in_gt = set()

                # ann = annotations_gt[0]
                for ann in annotations_gt:
                    category_name = gt_category_id_to_name[ann['category_id']]
                    category_names_in_gt.add(category_name)

                for category_name in category_names_detected:

                    if category_name in category_names_in_gt:
                        result_types_present.add('tp')
                    else:
                        result_types_present.add('fp')

                for category_name in category_names_in_gt:

                    # Is this an empty image?
                    if category_name in options.gt_empty_categories:

                        assert all([cn in options.gt_empty_categories for cn in category_names_in_gt]), \
                            'Image {} has both empty and non-empty ground truth labels'.format(
                                im_detection['file'])
                        if len(category_names_detected) > 0:
                            result_types_present.add('fp')
                            # If there is a false positive present in an empty image, there can't
                            # be any other result types present
                            assert len(result_types_present) == 1
                        else:
                            result_types_present.add('tn')

                    elif category_name in category_names_detected:

                        assert 'tp' in result_types_present

                    else:

                        result_types_present.add('fn')

                return result_types_present

            # ...def _categorize_image_with_image_level_gt(...)

            # im_detection = im_a; category_id_to_threshold = category_id_to_threshold_a
            result_types_present_a = \
                _categorize_image_with_image_level_gt(im_a,im_gt,annotations_gt,category_id_to_threshold_a)
            result_types_present_b = \
                _categorize_image_with_image_level_gt(im_b,im_gt,annotations_gt,category_id_to_threshold_b)


            ## Some combinations are nonsense

            # If either model has a TP or FN, the other has to have a TP or FN, since
            # there was something in the GT
            if ('tp' in result_types_present_a) or ('fn' in result_types_present_a):
                assert 'tp' in result_types_present_b or 'fn' in result_types_present_b
            if ('tp' in result_types_present_b) or ('fn' in result_types_present_b):
                assert 'tp' in result_types_present_a or 'fn' in result_types_present_a


            ## Choose a comparison category based on result types

            comparison_category = _result_types_to_comparison_category(
                result_types_present_a,result_types_present_b,ground_truth_type,options)

            # TODO: this may or may not be the right way to interpret sorting
            # by confidence in this case, e.g., we may want to sort by confidence
            # of correct or incorrect matches.  But this isn't *wrong*.
            max_conf_a = _maxempty([det['conf'] for det in im_a['detections']])
            max_conf_b = _maxempty([det['conf'] for det in im_b['detections']])
            sort_conf = max(max_conf_a,max_conf_b)

    # ...what kind of ground truth (if any) do we have?

        assert comparison_category is not None
        categories_to_image_pairs[comparison_category][fn] = im_pair
        im_pair['sort_conf'] = sort_conf

    # ...for each filename


    ##%% Sample and plot differences

    pool = None

    if options.n_rendering_workers > 1:
       worker_type = 'processes'
       if options.parallelize_rendering_with_threads:
           worker_type = 'threads'
       print('Rendering images with {} {}'.format(options.n_rendering_workers,worker_type))
       if options.parallelize_rendering_with_threads:
           pool = ThreadPool(options.n_rendering_workers)
       else:
           pool = Pool(options.n_rendering_workers)

    local_output_folder = os.path.join(options.output_folder,'cmp_' + \
                                       str(output_index).zfill(3))

    def render_detection_comparisons(category,image_pairs,image_filenames):

        print('Rendering detections for category {}'.format(category))

        category_folder = os.path.join(local_output_folder,category)
        os.makedirs(category_folder,exist_ok=True)

        # fn = image_filenames[0]
        if options.n_rendering_workers <= 1:
            output_image_paths = []
            for fn in tqdm(image_filenames):
                output_image_paths.append(_render_image_pair(fn,image_pairs,category_folder,
                                                            options,pairwise_options))
        else:
            output_image_paths = list(tqdm(pool.imap(
                partial(_render_image_pair, image_pairs=image_pairs,
                        category_folder=category_folder,options=options,
                        pairwise_options=pairwise_options),
                image_filenames),
                total=len(image_filenames)))

        return output_image_paths

    # ...def render_detection_comparisons()

    if len(options.colormap_a) > 1:
        color_string_a = str(options.colormap_a)
    else:
        color_string_a = options.colormap_a[0]

    if len(options.colormap_b) > 1:
        color_string_b = str(options.colormap_b)
    else:
        color_string_b = options.colormap_b[0]


    # For each category, generate comparison images and the
    # comparison HTML page.
    #
    # category = 'common_detections'
    for category in categories_to_image_pairs.keys():

        # Choose detection pairs we're going to render for this category
        image_pairs = categories_to_image_pairs[category]
        image_filenames = list(image_pairs.keys())

        if options.max_images_per_category is not None and options.max_images_per_category > 0:
            if len(image_filenames) > options.max_images_per_category:
                print('Sampling {} of {} image pairs for category {}'.format(
                    options.max_images_per_category,
                    len(image_filenames),
                    category))
                image_filenames = random.sample(image_filenames,
                                                options.max_images_per_category)
            assert len(image_filenames) <= options.max_images_per_category

        input_image_absolute_paths = [os.path.join(options.image_folder,fn) for fn in image_filenames]

        category_image_output_paths = render_detection_comparisons(category,
                                                            image_pairs,image_filenames)

        category_html_filename = os.path.join(local_output_folder,
                                              category + '.html')
        category_image_output_paths_relative = [os.path.relpath(s,local_output_folder) \
                                         for s in category_image_output_paths]

        image_info = []

        assert len(category_image_output_paths_relative) == len(input_image_absolute_paths)

        for i_fn,fn in enumerate(category_image_output_paths_relative):

            input_path_relative = image_filenames[i_fn]
            image_pair = image_pairs[input_path_relative]
            image_a = image_pair['im_a']
            image_b = image_pair['im_b']

            if options.fn_to_display_fn is not None:
                assert input_path_relative in options.fn_to_display_fn, \
                    'fn_to_display_fn provided, but {} is not mapped'.format(input_path_relative)
                display_path = options.fn_to_display_fn[input_path_relative]
            else:
                display_path = input_path_relative

            sort_conf = image_pair['sort_conf']

            max_conf_a = _maxempty([det['conf'] for det in image_a['detections']])
            max_conf_b = _maxempty([det['conf'] for det in image_b['detections']])

            title = display_path + ' (max conf {:.2f},{:.2f})'.format(max_conf_a,max_conf_b)

            if options.parse_link_paths:
                link_target_string = urllib.parse.quote(input_image_absolute_paths[i_fn])
            else:
                link_target_string = input_image_absolute_paths[i_fn]

            info = {
                'filename': fn,
                'title': title,
                'textStyle': 'font-family:verdana,arial,calibri;font-size:' + \
                    '80%;text-align:left;margin-top:20;margin-bottom:5',
                'linkTarget': link_target_string,
                'sort_conf':sort_conf
            }

            image_info.append(info)

        # ...for each image

        category_page_header_string = '<h1>{}</h1>\n'.format(categories_to_page_titles[category])
        category_page_header_string += '<p style="font-weight:bold;">\n'
        category_page_header_string += 'Model A: {} ({})<br/>\n'.format(
            pairwise_options.results_description_a,color_string_a)
        category_page_header_string += 'Model B: {} ({})'.format(
            pairwise_options.results_description_b,color_string_b)
        category_page_header_string += '</p>\n'

        category_page_header_string += '<p>\n'
        category_page_header_string += 'Detection thresholds for A ({}):\n{}<br/>'.format(
            pairwise_options.results_description_a,str(pairwise_options.detection_thresholds_a))
        category_page_header_string += 'Detection thresholds for B ({}):\n{}<br/>'.format(
            pairwise_options.results_description_b,str(pairwise_options.detection_thresholds_b))
        category_page_header_string += 'Rendering threshold for A ({}):\n{}<br/>'.format(
            pairwise_options.results_description_a,
            str(pairwise_options.rendering_confidence_threshold_a))
        category_page_header_string += 'Rendering threshold for B ({}):\n{}<br/>'.format(
            pairwise_options.results_description_b,
            str(pairwise_options.rendering_confidence_threshold_b))
        category_page_header_string += '</p>\n'

        subpage_header_string = '\n'.join(category_page_header_string.split('\n')[1:])

        # Default to sorting by filename
        if options.sort_by_confidence:
            image_info = sorted(image_info, key=lambda d: d['sort_conf'], reverse=True)
        else:
            image_info = sorted(image_info, key=lambda d: d['filename'])

        write_html_image_list(
            category_html_filename,
            images=image_info,
            options={
                'headerHtml': category_page_header_string,
                'subPageHeaderHtml': subpage_header_string,
                'maxFiguresPerHtmlFile': options.max_images_per_page
            })

    # ...for each category

    if pool is not None:
        try:
            pool.close()
            pool.join()
            print("Pool closed and joined for comparison rendering")
        except Exception:
            pass
    ##%% Write the top-level HTML file content

    html_output_string  = ''

    def _sanitize_id_name(s, lower=True):
        """
        Remove characters in [s] that are not allowed in HTML id attributes
        """

        s = re.sub(r'[^a-zA-Z0-9_-]', '', s)
        s = re.sub(r'^[^a-zA-Z]*', '', s)
        if lower:
            s = s.lower()
        return s

    comparison_short_name = '{}_vs_{}'.format(
        _sanitize_id_name(pairwise_options.results_description_a),
        _sanitize_id_name(pairwise_options.results_description_b))

    comparison_friendly_name = '{} vs {}'.format(
        pairwise_options.results_description_a,
        pairwise_options.results_description_b
    )

    html_output_string += '<p id="{}">Comparing <b>{}</b> (A, {}) to <b>{}</b> (B, {})</p>'.format(
        comparison_short_name,
        pairwise_options.results_description_a,color_string_a.lower(),
        pairwise_options.results_description_b,color_string_b.lower())
    html_output_string += '<div class="contentdiv">\n'
    html_output_string += 'Detection thresholds for {}:\n{}<br/>'.format(
        pairwise_options.results_description_a,
        str(pairwise_options.detection_thresholds_a))
    html_output_string += 'Detection thresholds for {}:\n{}<br/>'.format(
        pairwise_options.results_description_b,
        str(pairwise_options.detection_thresholds_b))
    html_output_string += 'Rendering threshold for {}:\n{}<br/>'.format(
        pairwise_options.results_description_a,
        str(pairwise_options.rendering_confidence_threshold_a))
    html_output_string += 'Rendering threshold for {}:\n{}<br/>'.format(
        pairwise_options.results_description_b,
        str(pairwise_options.rendering_confidence_threshold_b))

    html_output_string += '<br/>'

    html_output_string += 'Rendering a maximum of {} images per category<br/>'.format(
        options.max_images_per_category)

    html_output_string += '<br/>'

    category_summary = ''
    for i_category,category_name in enumerate(categories_to_image_pairs):
        if i_category > 0:
            category_summary += '<br/>'
        category_summary += '{} {}'.format(
            len(categories_to_image_pairs[category_name]),
            category_name.replace('_',' '))

    category_summary = \
        'Of {} total files:<br/><br/><div style="margin-left:15px;">{}</div><br/>'.format(
            len(filenames_to_compare),category_summary)

    html_output_string += category_summary

    html_output_string += 'Comparison pages:<br/><br/>\n'
    html_output_string += '<div style="margin-left:15px;">\n'

    comparison_path_relative = os.path.relpath(local_output_folder,options.output_folder)
    for category in categories_to_image_pairs.keys():
        category_html_filename = os.path.join(comparison_path_relative,category + '.html')
        html_output_string += '<a href="{}">{}</a><br/>\n'.format(
            category_html_filename,category)

    html_output_string += '</div>\n'
    html_output_string += '</div>\n'

    pairwise_results = PairwiseBatchComparisonResults()

    pairwise_results.comparison_short_name = comparison_short_name
    pairwise_results.comparison_friendly_name = comparison_friendly_name
    pairwise_results.html_content = html_output_string
    pairwise_results.pairwise_options = pairwise_options
    pairwise_results.categories_to_image_pairs = categories_to_image_pairs

    return pairwise_results

# ...def _pairwise_compare_batch_results()


def compare_batch_results(options):
    """
    The main entry point for this module.  Runs one or more batch results comparisons,
    writing results to an html page.  Most of the work is deferred to _pairwise_compare_batch_results().

    Args:
        options (BatchComparisonOptions): job options to use for this comparison task, including the
            list of specific pairswise comparisons to make (in the pairwise_options field)

    Returns:
        BatchComparisonResults: the results of this comparison task
    """

    assert options.output_folder is not None
    assert options.image_folder is not None
    assert options.pairwise_options is not None

    options = copy.deepcopy(options)

    if not isinstance(options.pairwise_options,list):
        options.pairwise_options = [options.pairwise_options]

    pairwise_options_list = options.pairwise_options
    n_comparisons = len(pairwise_options_list)

    options.pairwise_options = None

    html_content = ''
    all_pairwise_results = []

    # i_comparison = 0; pairwise_options = pairwise_options_list[i_comparison]
    for i_comparison,pairwise_options in enumerate(pairwise_options_list):

        print('Running comparison {} of {}'.format(i_comparison,n_comparisons))
        pairwise_results = \
            _pairwise_compare_batch_results(options,i_comparison,pairwise_options)
        html_content += pairwise_results.html_content
        all_pairwise_results.append(pairwise_results)

    # ...for each pairwise comparison

    html_output_string = main_page_header
    job_name_string = ''
    if len(options.job_name) > 0:
        job_name_string = ' for {}'.format(options.job_name)
    html_output_string += '<h2>Comparison of results{}</h2>\n'.format(
        job_name_string)

    if options.include_toc and (len(pairwise_options_list) > 2):
        toc_string = '<p><b>Contents</b></p>\n'
        toc_string += '<div class="contentdiv">\n'
        for r in all_pairwise_results:
            toc_string += '<a href="#{}">{}</a><br/>'.format(r.comparison_short_name,
                                                            r.comparison_friendly_name)
        toc_string += '</div>\n'
        html_output_string += toc_string

    html_output_string += html_content
    html_output_string += main_page_footer

    html_output_file = os.path.join(options.output_folder,'index.html')
    with open(html_output_file,'w') as f:
        f.write(html_output_string)

    results = BatchComparisonResults()
    results.html_output_file = html_output_file
    results.pairwise_results = all_pairwise_results
    return results


def n_way_comparison(filenames,
                     options,
                     detection_thresholds=None,
                     rendering_thresholds=None,
                     model_names=None):
    """
    Performs N pairwise comparisons for the list of results files in [filenames], by generating
    sets of pairwise options and calling compare_batch_results.

    Args:
        filenames (list): list of MD results filenames to compare
        options (BatchComparisonOptions): task options set in which pairwise_options is still
            empty; that will get populated from [filenames]
        detection_thresholds (list, optional): list of detection thresholds with the same length
            as [filenames], or None to use sensible defaults
        rendering_thresholds (list, optional): list of rendering thresholds with the same length
            as [filenames], or None to use sensible defaults
        model_names (list, optional): list of model names to use the output HTML file, with
            the same length as [filenames], or None to use sensible defaults

    Returns:
        BatchComparisonResults: the results of this comparison task
    """

    if detection_thresholds is None:
        detection_thresholds = [0.15] * len(filenames)
    assert len(detection_thresholds) == len(filenames), \
        '[detection_thresholds] should be the same length as [filenames]'

    if rendering_thresholds is not None:
        assert len(rendering_thresholds) == len(filenames)
        '[rendering_thresholds] should be the same length as [filenames]'
    else:
        rendering_thresholds = [(x*0.6666) for x in detection_thresholds]

    if model_names is not None:
        assert len(model_names) == len(filenames), \
            '[model_names] should be the same length as [filenames]'

    options.pairwise_options = []

    # Choose all pairwise combinations of the files in [filenames]
    for i, j in itertools.combinations(list(range(0,len(filenames))),2):

        pairwise_options = PairwiseBatchComparisonOptions()

        pairwise_options.results_filename_a = filenames[i]
        pairwise_options.results_filename_b = filenames[j]

        pairwise_options.rendering_confidence_threshold_a = rendering_thresholds[i]
        pairwise_options.rendering_confidence_threshold_b = rendering_thresholds[j]

        pairwise_options.detection_thresholds_a = {'default':detection_thresholds[i]}
        pairwise_options.detection_thresholds_b = {'default':detection_thresholds[j]}

        if model_names is not None:
            pairwise_options.results_description_a = model_names[i]
            pairwise_options.results_description_b = model_names[j]

        options.pairwise_options.append(pairwise_options)

    return compare_batch_results(options)

# ...def n_way_comparison(...)


def find_image_level_detections_above_threshold(results,threshold=0.2,category_names=None):
    """
    Returns images in the set of MD results [results] with detections above
    a threshold confidence level, optionally only counting certain categories.

    Args:
        results (str or dict): the set of results, either a .json filename or a results
            dict
        threshold (float, optional): the threshold used to determine the target number of
            detections in [results]
        category_names (list or str, optional): the list of category names to consider (defaults
            to using all categories), or the name of a single category.

    Returns:
        list: the images with above-threshold detections
    """
    if isinstance(results,str):
        with open(results,'r') as f:
            results = json.load(f)

    category_ids_to_consider = None

    if category_names is not None:

        if isinstance(category_names,str):
            category_names = [category_names]

        category_id_to_name = results['detection_categories']
        category_name_to_id = invert_dictionary(category_id_to_name)

        category_ids_to_consider = []

        # category_name = category_names[0]
        for category_name in category_names:
            category_id = category_name_to_id[category_name]
            category_ids_to_consider.append(category_id)

        assert len(category_ids_to_consider) > 0, \
            'Category name list did not map to any category IDs'

    images_above_threshold = []

    for im in results['images']:

        if ('detections' in im) and (im['detections'] is not None) and (len(im['detections']) > 0):
            confidence_values_this_image = [0]
            for det in im['detections']:
                if category_ids_to_consider is not None:
                    if det['category'] not in category_ids_to_consider:
                        continue
                confidence_values_this_image.append(det['conf'])
            if max(confidence_values_this_image) >= threshold:
                images_above_threshold.append(im)

    # ...for each image

    return images_above_threshold

# ...def find_image_level_detections_above_threshold(...)


def find_equivalent_threshold(results_a,
                              results_b,
                              threshold_a=0.2,
                              category_names=None,
                              verbose=False):
    """
    Given two sets of detector results, finds the confidence threshold for results_b
    that produces the same fraction of *images* with detections as threshold_a does for
    results_a.  Uses all categories.

    Args:
        results_a (str or dict): the first set of results, either a .json filename or a results
            dict
        results_b (str or dict): the second set of results, either a .json filename or a results
            dict
        threshold_a (float, optional): the threshold used to determine the target number of
            detections in results_a
        category_names (list or str, optional): the list of category names to consider (defaults
            to using all categories), or the name of a single category.
        verbose (bool, optional): enable additional debug output

    Returns:
        float: the threshold that - when applied to results_b - produces the same number
        of image-level detections that results from applying threshold_a to results_a
    """

    if isinstance(results_a,str):
        if verbose:
            print('Loading results from {}'.format(results_a))
        with open(results_a,'r') as f:
            results_a = json.load(f)

    if isinstance(results_b,str):
        if verbose:
            print('Loading results from {}'.format(results_b))
        with open(results_b,'r') as f:
            results_b = json.load(f)

    category_ids_to_consider_a = None
    category_ids_to_consider_b = None

    if category_names is not None:

        if isinstance(category_names,str):
            category_names = [category_names]

        categories_a = results_a['detection_categories']
        categories_b = results_b['detection_categories']
        category_name_to_id_a = invert_dictionary(categories_a)
        category_name_to_id_b = invert_dictionary(categories_b)

        category_ids_to_consider_a = []
        category_ids_to_consider_b = []

        # category_name = category_names[0]
        for category_name in category_names:
            category_id_a = category_name_to_id_a[category_name]
            category_id_b = category_name_to_id_b[category_name]
            category_ids_to_consider_a.append(category_id_a)
            category_ids_to_consider_b.append(category_id_b)

        assert len(category_ids_to_consider_a) > 0 and len(category_ids_to_consider_b) > 0, \
            'Category name list did not map to any category IDs in one or both detection sets'

    def _get_confidence_values_for_results(images,category_ids_to_consider,threshold):
        """
        Return a list of the maximum confidence value for each image in [images].
        Returns zero confidence for images with no detections (or no detections
        in the specified categories).  Does not return anything for invalid images.
        """

        confidence_values = []
        images_above_threshold = []

        for im in images:
            if 'detections' in im and im['detections'] is not None:
                if len(im['detections']) == 0:
                    confidence_values.append(0)
                else:
                    confidence_values_this_image = []
                    for det in im['detections']:
                        if category_ids_to_consider is not None:
                            if det['category'] not in category_ids_to_consider:
                                continue
                        confidence_values_this_image.append(det['conf'])
                    if len(confidence_values_this_image) == 0:
                        confidence_values.append(0)
                    else:
                        max_conf_value = max(confidence_values_this_image)

                        if threshold is not None and max_conf_value >= threshold:
                            images_above_threshold.append(im)
                        confidence_values.append(max_conf_value)
        # ...for each image

        return confidence_values, images_above_threshold

    # ...def _get_confidence_values_for_results(...)

    confidence_values_a,images_above_threshold_a = \
        _get_confidence_values_for_results(results_a['images'],
                                          category_ids_to_consider_a,
                                          threshold_a)

    # Not necessary, but facilitates debugging
    confidence_values_a = sorted(confidence_values_a)

    if verbose:
        print('For result set A, considering {} of {} images'.format(
            len(confidence_values_a),len(results_a['images'])))
    confidence_values_a_above_threshold = [c for c in confidence_values_a if c >= threshold_a]

    confidence_values_b,_ = _get_confidence_values_for_results(results_b['images'],
                                                              category_ids_to_consider_b,
                                                              threshold=None)
    if verbose:
        print('For result set B, considering {} of {} images'.format(
            len(confidence_values_b),len(results_b['images'])))

    confidence_values_b = sorted(confidence_values_b)

    # Find the threshold that produces the same fraction of detections for results_b
    target_detection_fraction = len(confidence_values_a_above_threshold) / len(confidence_values_a)

    # How many detections do we want in results_b?
    target_number_of_detections = round(len(confidence_values_b) * target_detection_fraction)

    # How many non-detections do we want in results_b?
    target_number_of_non_detections = len(confidence_values_b) - target_number_of_detections
    detection_cutoff_index = max(target_number_of_non_detections,0)
    threshold_b = confidence_values_b[detection_cutoff_index]

    confidence_values_b_above_threshold = [c for c in confidence_values_b if c >= threshold_b]
    confidence_values_b_above_reference_threshold = [c for c in confidence_values_b if c >= threshold_a]

    # Special case: if the number of detections above the selected threshold is the same as the
    # number above the reference threshold, use the reference threshold
    if len(confidence_values_b_above_threshold) == len(confidence_values_b_above_reference_threshold):
        print('Detection count for reference threshold matches target threshold')
        threshold_b = threshold_a

    if verbose:
        print('{} confidence values above threshold (A)'.format(
            len(confidence_values_a_above_threshold)))
        confidence_values_b_above_threshold = \
            [c for c in confidence_values_b if c >= threshold_b]
        print('{} confidence values above threshold (B)'.format(
            len(confidence_values_b_above_threshold)))

    return threshold_b

# ...def find_equivalent_threshold(...)


#%% Interactive driver

if False:

    #%% Test two-way comparison

    options = BatchComparisonOptions()

    options.parallelize_rendering_with_threads = True

    options.job_name = 'BCT'
    options.output_folder = r'g:\temp\comparisons'
    options.image_folder = r'g:\camera_traps\camera_trap_images'
    options.max_images_per_category = 100
    options.sort_by_confidence = True

    options.pairwise_options = []

    results_base = os.path.expanduser('~/postprocessing/bellevue-camera-traps')
    filenames = [
        os.path.join(results_base,r'bellevue-camera-traps-2023-12-05-v5a.0.0\combined_api_outputs\bellevue-camera-traps-2023-12-05-v5a.0.0_detections.json'),
        os.path.join(results_base,r'bellevue-camera-traps-2023-12-05-aug-v5a.0.0\combined_api_outputs\bellevue-camera-traps-2023-12-05-aug-v5a.0.0_detections.json')
        ]

    detection_thresholds = [0.15,0.15]
    rendering_thresholds = None

    results = n_way_comparison(filenames,
                               options,
                               detection_thresholds,
                               rendering_thresholds=rendering_thresholds)

    from megadetector.utils.path_utils import open_file
    open_file(results.html_output_file)


    #%% Test three-way comparison

    options = BatchComparisonOptions()

    options.parallelize_rendering_with_threads = False

    options.job_name = 'KGA-test'
    options.output_folder = os.path.expanduser('~/tmp/md-comparison-test')
    options.image_folder = os.path.expanduser('~/data/KGA')

    options.pairwise_options = []

    filenames = [
        os.path.expanduser('~/data/KGA-4.json'),
        os.path.expanduser('~/data/KGA-5a.json'),
        os.path.expanduser('~/data/KGA-5b.json')
        ]

    detection_thresholds = [0.7,0.15,0.15]

    results = n_way_comparison(filenames,options,detection_thresholds,rendering_thresholds=None)

    from megadetector.utils.path_utils import open_file
    open_file(results.html_output_file)


#%% Command-line driver

"""
python compare_batch_results.py ~/tmp/comparison-test ~/data/KGA \
    ~/data/KGA-5a.json ~/data/KGA-5b.json ~/data/KGA-4.json \
    --detection_thresholds 0.15 0.15 0.7 --rendering_thresholds 0.1 0.1 0.6 --use_processes
"""

def main(): # noqa

    options = BatchComparisonOptions()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
           Example:

           python compare_batch_results.py output_folder image_folder mdv5a.json mdv5b.json mdv4.json --detection_thresholds 0.15 0.15 0.7
           '''))

    parser.add_argument('output_folder', type=str, help='folder to which to write html results')

    parser.add_argument('image_folder', type=str, help='image source folder')

    parser.add_argument('results_files', nargs='*', type=str, help='list of .json files to be compared')

    parser.add_argument('--detection_thresholds', nargs='*', type=float,
                        help='list of detection thresholds, same length as the number of .json files, ' + \
                             'defaults to 0.15 for all files')

    parser.add_argument('--rendering_thresholds', nargs='*', type=float,
                        help='list of rendering thresholds, same length as the number of .json files, ' + \
                             'defaults to 0.10 for all files')

    parser.add_argument('--max_images_per_category', type=int, default=options.max_images_per_category,
                       help='number of images to sample for each agreement category (common detections, etc.)')

    parser.add_argument('--target_width', type=int, default=options.target_width,
                        help='output image width, defaults to {}'.format(options.target_width))

    parser.add_argument('--use_processes', action='store_true',
                        help='use processes rather than threads for parallelization')

    parser.add_argument('--open_results', action='store_true',
                        help='open the output html file when done')

    parser.add_argument('--n_rendering_workers', type=int, default=options.n_rendering_workers,
                        help='number of workers for parallel rendering, defaults to {}'.format(
                            options.n_rendering_workers))

    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    print('Output folder:')
    print(args.output_folder)

    print('\nResults files:')
    print(args.results_files)

    print('\nDetection thresholds:')
    print(args.detection_thresholds)

    print('\nRendering thresholds:')
    print(args.rendering_thresholds)

    # Convert to options objects
    options = BatchComparisonOptions()

    options.output_folder = args.output_folder
    options.image_folder = args.image_folder
    options.target_width = args.target_width
    options.n_rendering_workers = args.n_rendering_workers
    options.max_images_per_category = args.max_images_per_category

    if args.use_processes:
        options.parallelize_rendering_with_threads = False

    results = n_way_comparison(args.results_files,
                               options,
                               args.detection_thresholds,
                               args.rendering_thresholds)

    if args.open_results:
        path_utils.open_file(results.html_output_file)

    print('Wrote results to {}'.format(results.html_output_file))

# ...main()


if __name__ == '__main__':

    main()
