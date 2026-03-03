"""

analyze_classification_results.py

Given a results file in MD format, and a ground truth file in COCO Camera Traps format,
both containing classification results, perform various analyses, including:

* Precision/recall analysis
* Confusion matrix with links to visualization pages

Only analyzes image-level correctness, i.e., box locations are ignored in both
the predictions and the ground truth.

"""

#%% Imports and constants

import os
import json
import random
import argparse
import sys

import numpy as np

from collections import defaultdict
from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from tqdm import tqdm

from megadetector.utils.path_utils import flatten_path
from megadetector.utils.write_html_image_list import write_html_image_list
from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file
from megadetector.detection.run_detector import get_typical_confidence_threshold_from_results
from megadetector.visualization import visualization_utils as vis_utils

# See "detection_category_mapping" below
default_detection_category_mapping = {}
default_detection_category_mapping['person'] = 'human'
default_detection_category_mapping['vehicle'] = 'vehicle'


#%% Support classes

class ClassificationAnalysisOptions:
    """
    Options used to parameterize analyze_classification_results().
    """

    def __init__(self):

        ### Required inputs

        #: MD-formatted results file to analyze
        self.results_file = None

        #: Ground truth file in COCO Camera Traps format
        self.gt_file = None

        ### Optional inputs

        #: Ignore all detections below this confidence threshold
        #:
        #: If this is None, a confidence threshold is selected based on the detector
        #: version.
        self.detection_threshold = None

        #: Folder where images live; filenames in [results_file] and [gt_file] should
        #: be relative to this path.  Only required if html_output_dir is not None.
        self.image_base_dir = None

        #: Folder to which we should write html output page
        self.html_output_dir = None

        #: Maximum number of total images to render.  Only relevant if html_output_dir is
        #: not None.
        self.max_total_images = 8000

        #: Maximum number of images to render per confusion matrix cell.  Only relevant if
        #: html_output_dir is not None.
        self.max_images_per_cell = 50

        #: Random seed to be used if image sampling is necessary
        self.random_seed = 0

        #: Confidence threshold to apply to classification (not detection) results
        self.classification_confidence_threshold = 0.5

        #: A dict mapping detection category names to classification category names, for
        #: categories we want to handle specially.  Any detection in a matching category with
        #: an above-threshold confidence value is treated as if it had a classification
        #: with the corresponding (mapped) classification category, with a confidence of 1.0,
        #: whether or not that category exists in the ground truth.
        #:
        #: For example, by default a detection with category "person" with confidence 0.4
        #: should be treated as a classification of category "human" with confidence 1.0.
        #:
        #: Defaults to detection_category_mapping.
        self.detection_category_mapping = None

        #: If True, the entire analysis will be performed at the *sequence* level, rather
        #: than the image level.
        self.sequence_level_analysis = False

        #: Number of workers to use when rendering images
        self.rendering_workers = 10

        #: Should we use threads ("threads") or processes ("processes") for rendering?
        #:
        #: Only relevant if rendering_workers is > 1.
        self.rendering_pool_type = 'threads'

        #: Width of rendered output images (-1 to preserve original size)
        self.output_image_width = 1000

        #: Number of top misprediction categories to show in the per-category
        #: statistics table (for both "mispredicted as this" and "this was
        #: mispredicted as" columns).
        self.n_mispredictions_for_table = 3

        #: List of category names to completely exclude from the analysis,
        #: whether they appear in ground truth or predictions.  Checked after
        #: detection category remapping (e.g. person -> human).
        self.categories_to_ignore = None

        #: If True, collapse each image's (or sequence's) predictions to a
        #: single category.  The winner is chosen by the *number* of above-
        #: threshold classifications for that category, with max classification
        #: confidence as a tie-breaker.  For sequence-level analyses, collapsing
        #: happens at the sequence level (counts are summed across images in
        #: the sequence), not per-image.
        self.single_prediction_per_image = False

        #: If True, collapse each image's (or sequence's) ground truth to a
        #: single category.  For image-level analyses, ties are broken
        #: alphabetically.  For sequence-level analyses, ties are broken by
        #: count (how many images in the sequence have that category), then
        #: alphabetically.
        self.single_label_per_image = False

    # ...def __init__(...)

# ...class ClassificationAnalysisOptions


class AnalysisResults:
    """
    Results returned by analyze_classification_results().
    """

    def __init__(self):

        #: Dictionary mapping category names to dicts, where each item has
        #: at least the keys "precision", "recall", "f1", "n_ground_truth", "n_predicted"
        self.per_category_results = None

        self.macro_f1 = None
        self.micro_f1 = None
        self.micro_precision = None
        self.micro_recall = None
        self.accuracy = None

        #: The confusion matrix as a 2D numpy array
        self.confusion_matrix = None

        #: Ordered list of category names corresponding to matrix rows/columns
        self.active_categories = None

        #: Path to the output HTML file (if generated)
        self.html_output_file = None

    # ...def __init__(...)

# ...class AnalysisResults


#%% Support functions

def _get_image_predicted_categories(im, detection_threshold,
                                    classification_confidence_threshold,
                                    detection_category_id_to_name,
                                    classification_category_id_to_name,
                                    detection_category_mapping):
    """
    For a single MD results image entry, returns a tuple of:

    - a dict mapping predicted category names (lowercase) to their max confidence
    - a dict mapping predicted category names (lowercase) to the number of
      above-threshold classifications for that category
    """

    predicted_categories = {}
    predicted_counts = defaultdict(int)

    has_above_threshold_detection = False

    if im.get('detections') is None:
        return {'empty': 1.0}, {'empty': 1}

    for det in im['detections']:

        if det['conf'] < detection_threshold:
            continue

        has_above_threshold_detection = True

        det_category_name = detection_category_id_to_name.get(det['category'], None)
        if det_category_name is None:
            continue

        det_category_name_lower = det_category_name.lower()

        # Check if this detection category is in the mapping (e.g., person -> human)
        if det_category_name_lower in detection_category_mapping:
            mapped_name = detection_category_mapping[det_category_name_lower].lower()
            if mapped_name not in predicted_categories or predicted_categories[mapped_name] < 1.0:
                predicted_categories[mapped_name] = 1.0
            predicted_counts[mapped_name] += 1
        else:
            # Look at classifications
            classifications = det.get('classifications', None)
            has_above_threshold_classification = False

            if classifications is not None:
                for cls in classifications:
                    cls_id = str(cls[0])
                    cls_conf = cls[1]
                    if cls_conf >= classification_confidence_threshold:
                        cls_name = classification_category_id_to_name.get(cls_id, None)
                        if cls_name is not None:
                            cls_name_lower = cls_name.lower()
                            has_above_threshold_classification = True
                            if cls_name_lower not in predicted_categories or \
                                    predicted_categories[cls_name_lower] < cls_conf:
                                predicted_categories[cls_name_lower] = cls_conf
                            predicted_counts[cls_name_lower] += 1

            if not has_above_threshold_classification:
                if 'unknown' not in predicted_categories:
                    predicted_categories['unknown'] = 1.0
                predicted_counts['unknown'] += 1

    # ...for each detection

    if not has_above_threshold_detection:
        return {'empty': 1.0}, {'empty': 1}

    return predicted_categories, dict(predicted_counts)

# ...def _get_image_predicted_categories(...)


def _render_single_image(im, render_constants):
    """
    Renders a single image with bounding boxes to the output folder.

    Args:
        im (dict): image entry from MD results, with an added '_output_path' key
        render_constants (dict): rendering parameters

    Returns:
        dict: result with 'success', 'output_path', 'file' keys
    """

    image_base_dir = render_constants['image_base_dir']
    output_dir = render_constants['output_dir']
    detection_label_map = render_constants['detection_label_map']
    classification_label_map = render_constants['classification_label_map']
    detection_threshold = render_constants['detection_threshold']
    classification_confidence_threshold = render_constants['classification_confidence_threshold']
    output_image_width = render_constants['output_image_width']

    result = {
        'success': False,
        'output_path': None,
        'file': im['file']
    }

    # Build output filename
    fn_clean = flatten_path(im['file']).replace(' ', '_')
    output_path = os.path.join(output_dir, fn_clean)

    # Skip if already rendered
    if os.path.isfile(output_path):
        result['success'] = True
        result['output_path'] = output_path
        return result

    input_path = os.path.join(image_base_dir, im['file'])

    if not os.path.isfile(input_path):
        print('Warning: image not found: {}'.format(input_path))
        return result

    try:
        image = vis_utils.open_image(input_path)
        image = vis_utils.resize_image(image, output_image_width)

        vis_utils.render_detection_bounding_boxes(
            im['detections'], image,
            label_map=detection_label_map,
            classification_label_map=classification_label_map,
            confidence_threshold=detection_threshold,
            classification_confidence_threshold=classification_confidence_threshold)

        image.save(output_path)
        result['success'] = True
        result['output_path'] = output_path
    except Exception as e:
        print('Warning: error rendering {}: {}'.format(im['file'], str(e)))

    return result

# ...def _render_single_image(...)


#%% Core functions

def analyze_classification_results(options):
    """
    Perform precision-recall analysis on classification results.

    Args:
        options (ClassificationAnalysisOptions): options object defining filenames
            and analysis parameters.

    Returns:
        AnalysisResults: results of the classification analysis
    """

    ## Setup and defaults

    if options.detection_category_mapping is None:
        detection_category_mapping = {k.lower(): v.lower()
                                      for k, v in default_detection_category_mapping.items()}
    else:
        detection_category_mapping = {k.lower(): v.lower()
                                      for k, v in options.detection_category_mapping.items()}

    ## Load files

    print('Loading results from {}'.format(options.results_file))
    results_data = load_md_or_speciesnet_file(options.results_file)

    detection_threshold = options.detection_threshold
    if detection_threshold is None:
        detection_threshold = get_typical_confidence_threshold_from_results(results_data)
        print('Using auto-detected confidence threshold: {}'.format(detection_threshold))

    print('Loading ground truth from {}'.format(options.gt_file))
    with open(options.gt_file, 'r') as f:
        gt_data = json.load(f)

    # Build category maps
    detection_category_id_to_name = results_data.get('detection_categories', {})
    classification_category_id_to_name = results_data.get('classification_categories', {})

    gt_category_id_to_name = {}
    for c in gt_data['categories']:
        gt_category_id_to_name[c['id']] = c['name']

    # Normalize filenames (backslash -> forward slash)
    for im in results_data['images']:
        im['file'] = im['file'].replace('\\', '/')

    for im in gt_data['images']:
        im['file_name'] = im['file_name'].replace('\\', '/')

    ## Validate file compatibility

    results_fn_to_im = {}
    for im in results_data['images']:
        results_fn_to_im[im['file']] = im

    gt_fn_to_im = {}
    for im in gt_data['images']:
        gt_fn_to_im[im['file_name']] = im

    common_filenames = set(results_fn_to_im.keys()) & set(gt_fn_to_im.keys())
    results_only = set(results_fn_to_im.keys()) - set(gt_fn_to_im.keys())
    gt_only = set(gt_fn_to_im.keys()) - set(results_fn_to_im.keys())

    print('{} images in common between results and ground truth'.format(len(common_filenames)))

    if len(results_only) > 0:
        print('Warning: {} images (of {}) in results but not in ground truth'.format(
            len(results_only),len(gt_fn_to_im)))
    if len(gt_only) > 0:
        print('Warning: {} images (of {}) in ground truth but not in results'.format(
            len(gt_only),len(results_fn_to_im)))

    assert len(common_filenames) > 0, \
        'No images in common between results and ground truth'

    ## Build GT category assignments

    gt_image_id_to_annotations = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_image_id_to_annotations[ann['image_id']].append(ann)

    filename_to_gt_categories = {}
    n_images_without_annotations = 0

    for im in gt_data['images']:
        fn = im['file_name']
        if fn not in common_filenames:
            continue

        annotations = gt_image_id_to_annotations.get(im['id'], [])
        if len(annotations) == 0:
            n_images_without_annotations += 1
            continue

        gt_cats = set()
        for ann in annotations:
            cat_name = gt_category_id_to_name[ann['category_id']].lower()
            gt_cats.add(cat_name)

        filename_to_gt_categories[fn] = gt_cats

    if n_images_without_annotations > 0:
        print('Warning: {} images in GT with no annotations (excluded)'.format(
            n_images_without_annotations))

    print('{} images with ground truth annotations'.format(len(filename_to_gt_categories)))

    ## Build predicted category assignments

    filename_to_pred_categories = {}
    filename_to_pred_counts = {}

    for fn in filename_to_gt_categories:
        im = results_fn_to_im[fn]
        pred_cats, pred_counts = _get_image_predicted_categories(
            im, detection_threshold,
            options.classification_confidence_threshold,
            detection_category_id_to_name,
            classification_category_id_to_name,
            detection_category_mapping)
        filename_to_pred_categories[fn] = pred_cats
        filename_to_pred_counts[fn] = pred_counts

    ## Filter ignored categories

    categories_to_ignore = set()
    if options.categories_to_ignore is not None:
        categories_to_ignore = set(c.lower() for c in options.categories_to_ignore)

    if len(categories_to_ignore) > 0:

        print('Ignoring {} categories: {}'.format(
            len(categories_to_ignore), sorted(categories_to_ignore)))

        # Remove ignored categories from GT
        fns_to_remove = []
        for fn in filename_to_gt_categories:
            filename_to_gt_categories[fn] -= categories_to_ignore
            if len(filename_to_gt_categories[fn]) == 0:
                fns_to_remove.append(fn)

        for fn in fns_to_remove:
            del filename_to_gt_categories[fn]
            del filename_to_pred_categories[fn]
            del filename_to_pred_counts[fn]

        if len(fns_to_remove) > 0:
            print('  {} images excluded because all their GT categories were ignored'.format(
                len(fns_to_remove)))

        # Remove ignored categories from predictions
        for fn in filename_to_pred_categories:
            for cat in categories_to_ignore:
                filename_to_pred_categories[fn].pop(cat, None)
                filename_to_pred_counts[fn].pop(cat, None)

        print('{} images remaining after filtering'.format(len(filename_to_gt_categories)))

    ## Sequence-level aggregation

    if options.sequence_level_analysis:

        # Build filename -> seq_id map from GT
        filename_to_seq_id = {}
        for im in gt_data['images']:
            fn = im['file_name']
            if fn in filename_to_gt_categories:
                assert 'seq_id' in im, \
                    'Image {} has no seq_id; sequence-level analysis requires seq_id'.format(fn)
                filename_to_seq_id[fn] = im['seq_id']

        # Group filenames by seq_id
        seq_id_to_filenames = defaultdict(list)
        for fn, seq_id in filename_to_seq_id.items():
            seq_id_to_filenames[seq_id].append(fn)

        # Aggregate GT and predictions per sequence
        seq_filename_to_gt_categories = {}
        seq_filename_to_pred_categories = {}
        seq_filename_to_pred_counts = {}

        for seq_id, filenames in seq_id_to_filenames.items():
            gt_union = set()
            pred_union = {}
            pred_count_union = defaultdict(int)
            for fn in filenames:
                gt_union |= filename_to_gt_categories[fn]
                for cat, conf in filename_to_pred_categories[fn].items():
                    if cat not in pred_union or pred_union[cat] < conf:
                        pred_union[cat] = conf
                for cat, count in filename_to_pred_counts[fn].items():
                    pred_count_union[cat] += count

            seq_filename_to_gt_categories[seq_id] = gt_union
            seq_filename_to_pred_categories[seq_id] = pred_union
            seq_filename_to_pred_counts[seq_id] = dict(pred_count_union)

        # Replace image-level dicts with sequence-level dicts
        filename_to_gt_categories = seq_filename_to_gt_categories
        filename_to_pred_categories = seq_filename_to_pred_categories
        filename_to_pred_counts = seq_filename_to_pred_counts

        print('{} sequences for analysis'.format(len(filename_to_gt_categories)))

    ## Collapse ground truth to single label per image/sequence

    if options.single_label_per_image:

        for entity_id in filename_to_gt_categories:
            gt_cats = filename_to_gt_categories[entity_id]
            if len(gt_cats) <= 1:
                continue

            if options.sequence_level_analysis:
                # Break ties by count (how many images have this category), then
                # alphabetically.
                seq_id = entity_id
                filenames_in_seq = seq_id_to_filenames[seq_id]
                cat_counts = defaultdict(int)
                for fn in filenames_in_seq:
                    # Use the original per-image GT (before sequence aggregation),
                    # but we already replaced that dict.  We can recount from
                    # gt_image_id_to_annotations.
                    gt_im = gt_fn_to_im[fn]
                    anns = gt_image_id_to_annotations.get(gt_im['id'], [])
                    for ann in anns:
                        cat_name = gt_category_id_to_name[ann['category_id']].lower()
                        if cat_name not in categories_to_ignore:
                            cat_counts[cat_name] += 1
                # Pick by count (descending), then alphabetically
                winner = max(gt_cats, key=lambda c: (cat_counts.get(c, 0), -ord(c[0])))
                # More robust: sort by (-count, name)
                winner = sorted(gt_cats, key=lambda c: (-cat_counts.get(c, 0), c))[0]
            else:
                # Image-level: break ties alphabetically
                winner = sorted(gt_cats)[0]

            filename_to_gt_categories[entity_id] = {winner}

        print('Collapsed ground truth to single label per {}'.format(
            'sequence' if options.sequence_level_analysis else 'image'))

    ## Collapse predictions to single label per image/sequence

    if options.single_prediction_per_image:

        for entity_id in filename_to_pred_categories:
            pred_cats = filename_to_pred_categories[entity_id]
            if len(pred_cats) <= 1:
                continue

            counts = filename_to_pred_counts[entity_id]

            # Pick by count (descending), then max confidence (descending),
            # then alphabetically
            winner = sorted(
                pred_cats.keys(),
                key=lambda c: (-counts.get(c, 0), -pred_cats[c], c))[0]

            winner_conf = pred_cats[winner]
            filename_to_pred_categories[entity_id] = {winner: winner_conf}
            filename_to_pred_counts[entity_id] = {winner: counts.get(winner, 1)}

        print('Collapsed predictions to single label per {}'.format(
            'sequence' if options.sequence_level_analysis else 'image'))

    ## Determine active categories

    all_gt_categories = set()
    for cats in filename_to_gt_categories.values():
        all_gt_categories |= cats

    all_pred_categories = set()
    for cats in filename_to_pred_categories.values():
        all_pred_categories |= set(cats.keys())

    active_categories = sorted(all_gt_categories | all_pred_categories)

    print('{} active categories'.format(len(active_categories)))

    category_to_index = {cat: i for i, cat in enumerate(active_categories)}

    ## Build confusion matrix

    n_categories = len(active_categories)
    confusion_matrix = np.zeros((n_categories, n_categories), dtype=int)

    # Maps (true_cat, pred_cat) -> list of filenames/seq_ids
    true_pred_to_filenames = defaultdict(list)

    for entity_id in filename_to_gt_categories:
        gt_cats = filename_to_gt_categories[entity_id]
        pred_cats = filename_to_pred_categories[entity_id]

        for true_cat in gt_cats:
            for pred_cat in pred_cats:

                # For off-diagonal entries, skip cases where both categories are
                # correctly present, i.e. the predicted category is in GT and the
                # true category was also predicted.  In this case the cross-product
                # would create a spurious "misprediction" entry.
                if true_cat != pred_cat:
                    if pred_cat in gt_cats and true_cat in pred_cats:
                        continue

                true_idx = category_to_index[true_cat]
                pred_idx = category_to_index[pred_cat]
                confusion_matrix[true_idx, pred_idx] += 1
                true_pred_to_filenames[(true_cat, pred_cat)].append(entity_id)

    ## Compute metrics

    per_category_results = {}
    total_tp = 0
    total_predicted = 0
    total_gt = 0

    for cat in active_categories:
        idx = category_to_index[cat]
        tp = confusion_matrix[idx, idx]
        fp = int(confusion_matrix[:, idx].sum()) - tp
        fn = int(confusion_matrix[idx, :].sum()) - tp

        n_gt = tp + fn
        n_pred = tp + fp

        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_category_results[cat] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_ground_truth': int(n_gt),
            'n_predicted': int(n_pred),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }

        total_tp += tp
        total_predicted += n_pred
        total_gt += n_gt

    # Compute per-image micro precision and recall.
    #
    # Note: when computed from the confusion matrix, micro precision always
    # equals micro recall (because total column sums == total row sums ==
    # total matrix entries).  Instead, we compute per-image set-based metrics:
    #
    #   micro_precision = sum(|GT ∩ pred|) / sum(|pred|)
    #   micro_recall    = sum(|GT ∩ pred|) / sum(|GT|)
    #
    # These can differ when images have multiple GT and/or predicted categories.

    sum_intersection = 0
    sum_gt_size = 0
    sum_pred_size = 0

    for entity_id in filename_to_gt_categories:
        gt_cats = filename_to_gt_categories[entity_id]
        pred_cats_set = set(filename_to_pred_categories[entity_id].keys())
        sum_intersection += len(gt_cats & pred_cats_set)
        sum_gt_size += len(gt_cats)
        sum_pred_size += len(pred_cats_set)

    micro_precision = sum_intersection / sum_pred_size if sum_pred_size > 0 else 0.0
    micro_recall = sum_intersection / sum_gt_size if sum_gt_size > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0.0

    f1_values = [per_category_results[cat]['f1'] for cat in active_categories
                 if per_category_results[cat]['n_ground_truth'] > 0]
    macro_f1 = np.mean(f1_values) if len(f1_values) > 0 else 0.0

    matrix_sum = confusion_matrix.sum()
    diagonal_sum = np.trace(confusion_matrix)
    accuracy = diagonal_sum / matrix_sum if matrix_sum > 0 else 0.0

    print('\nOverall metrics:')
    print('  Accuracy: {:.4f}'.format(accuracy))
    print('  Macro F1: {:.4f}'.format(macro_f1))
    print('  Micro precision: {:.4f}'.format(micro_precision))
    print('  Micro recall: {:.4f}'.format(micro_recall))
    print('  Micro F1: {:.4f}'.format(micro_f1))

    ## Generate HTML output

    html_output_file = None

    if options.html_output_dir is not None:

        assert options.image_base_dir is not None, \
            'image_base_dir is required when html_output_dir is specified'

        os.makedirs(options.html_output_dir, exist_ok=True)
        preview_images_folder = os.path.join(options.html_output_dir, 'images')
        os.makedirs(preview_images_folder, exist_ok=True)

        # Determine which images to render

        # For sequence-level analysis, we need to map back to individual filenames
        if options.sequence_level_analysis:
            seq_id_to_filenames_map = defaultdict(list)
            for im in gt_data['images']:
                fn = im['file_name']
                if fn in results_fn_to_im and 'seq_id' in im:
                    seq_id_to_filenames_map[im['seq_id']].append(fn)

        # Collect non-empty cells and their filenames
        non_empty_cells = {}
        for (true_cat, pred_cat), entity_ids in true_pred_to_filenames.items():
            if len(entity_ids) > 0:
                non_empty_cells[(true_cat, pred_cat)] = entity_ids

        # Determine how many images to sample per cell.
        #
        # First, cap each cell at max_images_per_cell.  Then, if the total
        # exceeds max_total_images, scale down proportionally.
        cell_desired_counts = {}
        for key, entity_ids in non_empty_cells.items():
            cell_desired_counts[key] = min(len(entity_ids), options.max_images_per_cell)

        total_desired = sum(cell_desired_counts.values())

        if total_desired > options.max_total_images and total_desired > 0:
            scale = options.max_total_images / total_desired
            for key in cell_desired_counts:
                cell_desired_counts[key] = max(1, int(cell_desired_counts[key] * scale))

        # Sample entities per cell
        rng = random.Random(options.random_seed)
        sampled_cells = {}

        for key, entity_ids in non_empty_cells.items():
            n_sample = cell_desired_counts[key]
            if len(entity_ids) <= n_sample:
                sampled_cells[key] = list(entity_ids)
            else:
                sampled_cells[key] = rng.sample(entity_ids, n_sample)

        # Collect all unique filenames that need rendering.
        #
        # For sequence-level analysis, render only the first image in each
        # sequence as an exemplar.
        filenames_to_render = set()
        for entity_ids in sampled_cells.values():
            if options.sequence_level_analysis:
                for seq_id in entity_ids:
                    fns = seq_id_to_filenames_map.get(seq_id, [])
                    if len(fns) > 0:
                        filenames_to_render.add(fns[0])
            else:
                filenames_to_render.update(entity_ids)

        # Build image entries for rendering
        images_to_render = []
        for fn in filenames_to_render:
            if fn in results_fn_to_im:
                images_to_render.append(results_fn_to_im[fn])

        print('\nRendering {} images...'.format(len(images_to_render)))

        # Render images
        render_constants = {
            'image_base_dir': options.image_base_dir,
            'output_dir': preview_images_folder,
            'detection_label_map': detection_category_id_to_name,
            'classification_label_map': classification_category_id_to_name,
            'detection_threshold': detection_threshold,
            'classification_confidence_threshold': options.classification_confidence_threshold,
            'output_image_width': options.output_image_width
        }

        if options.rendering_workers > 1 and len(images_to_render) > 1:

            if options.rendering_pool_type == 'threads':
                pool = ThreadPool(options.rendering_workers)
                worker_string = 'threads'
            else:
                pool = Pool(options.rendering_workers)
                worker_string = 'processes'

            print('Rendering with {} {}'.format(options.rendering_workers, worker_string))

            try:
                rendering_results = list(tqdm(
                    pool.imap(
                        partial(_render_single_image, render_constants=render_constants),
                        images_to_render),
                    total=len(images_to_render)))
            finally:
                pool.close()
                pool.join()
        else:
            rendering_results = []
            for im in tqdm(images_to_render):
                rendering_results.append(_render_single_image(im, render_constants))

        n_success = sum(1 for r in rendering_results if r['success'])
        print('Successfully rendered {} of {} images'.format(n_success, len(images_to_render)))

        # Generate per-cell HTML pages

        for (true_cat, pred_cat), entity_ids in sampled_cells.items():

            html_image_info_list = []

            for entity_id in entity_ids:

                # Get filenames for this entity; for sequence-level analysis,
                # show only the first image as an exemplar.
                if options.sequence_level_analysis:
                    all_fns = seq_id_to_filenames_map.get(entity_id, [])
                    fns = all_fns[:1]
                else:
                    fns = [entity_id]

                for fn in fns:
                    im = results_fn_to_im.get(fn, None)
                    if im is None:
                        continue

                    fn_clean = flatten_path(fn).replace(' ', '_')
                    image_link = 'images/' + fn_clean

                    gt_cats_str = ', '.join(sorted(filename_to_gt_categories.get(
                        entity_id if not options.sequence_level_analysis else entity_id,
                        set())))
                    pred_cats = filename_to_pred_categories.get(
                        entity_id if not options.sequence_level_analysis else entity_id, {})
                    pred_cats_str = ', '.join(
                        ['{} ({:.2f})'.format(c, conf)
                         for c, conf in sorted(pred_cats.items())])

                    title = '<b>File</b>: {}<br/><b>GT</b>: {}<br/><b>Pred</b>: {}'.format(
                        fn, gt_cats_str, pred_cats_str)

                    html_image_info = {
                        'filename': image_link,
                        'title': title,
                        'textStyle':
                            'font-family:verdana,arial,calibri;font-size:80%;'
                            'text-align:left;margin-top:20;margin-bottom:5'
                    }
                    html_image_info_list.append(html_image_info)

            # ...for each entity

            cell_html_filename = 'predicted_{}_true_{}.html'.format(
                pred_cat.replace(' ', '_').replace('/', '_'),
                true_cat.replace(' ', '_').replace('/', '_'))
            cell_html_path = os.path.join(options.html_output_dir, cell_html_filename)

            cell_title = 'True: {}, Predicted: {}'.format(true_cat, pred_cat)
            cell_options = {
                'headerHtml': '<h1>{}</h1>'.format(cell_title)
            }

            write_html_image_list(
                filename=cell_html_path,
                images=html_image_info_list,
                options=cell_options)

        # ...for each cell

        # Generate index.html

        # Helper to build a cell subpage filename
        def _cell_html_filename(pred_cat, true_cat):
            return 'predicted_{}_true_{}.html'.format(
                pred_cat.replace(' ', '_').replace('/', '_'),
                true_cat.replace(' ', '_').replace('/', '_'))

        # Build per-category misprediction summaries (top-N from off-diagonal)
        n_mispred = options.n_mispredictions_for_table

        # "Mispredicted as this category" = FP column: other true categories
        # predicted as this one (off-diagonal entries in this category's column)
        mispredicted_as = {}

        # "This was mispredicted as" = FN row: this true category predicted as
        # other categories (off-diagonal entries in this category's row)
        this_mispredicted_as = {}

        for cat in active_categories:
            idx = category_to_index[cat]

            # Column: which true categories were predicted as this one?
            col_entries = []
            for other_cat in active_categories:
                if other_cat == cat:
                    continue
                other_idx = category_to_index[other_cat]
                count = int(confusion_matrix[other_idx, idx])
                if count > 0:
                    col_entries.append((other_cat, count))
            col_entries.sort(key=lambda x: -x[1])
            mispredicted_as[cat] = col_entries[:n_mispred]

            # Row: what was this true category predicted as?
            row_entries = []
            for other_cat in active_categories:
                if other_cat == cat:
                    continue
                other_idx = category_to_index[other_cat]
                count = int(confusion_matrix[idx, other_idx])
                if count > 0:
                    row_entries.append((other_cat, count))
            row_entries.sort(key=lambda x: -x[1])
            this_mispredicted_as[cat] = row_entries[:n_mispred]

        style_header = """<head>
    <style type="text/css">
    a { text-decoration: none; }
    body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
    div.contentdiv { margin-left: 20px; }
    table.result-table { border:1px solid black; border-collapse: collapse; margin-left:50px;}
    td,th { padding:10px; border:1px solid #ddd; }
    th.sortable { cursor: pointer; user-select: none; }
    th.sortable:hover { background-color: #e8e8e8; }
    th.sortable::after { content: " \\2195"; font-size: 80%; color: #999; }
    .rotate {
      padding:0px;
      writing-mode:vertical-lr;
      -webkit-transform: rotate(-180deg);
      -moz-transform: rotate(-180deg);
      -ms-transform: rotate(-180deg);
      -o-transform: rotate(-180deg);
      transform: rotate(-180deg);
    }
    .metrics-table td { text-align: right; padding: 5px 15px; }
    .metrics-table td:first-child { text-align: left; font-weight: bold; }
    .mispred-cell { font-size: 85%; line-height: 1.6; }
    </style>
    <script>
    function sortTable(tableId, colIdx, type) {
      var table = document.getElementById(tableId);
      var tbody = table.tBodies[0];
      var rows = Array.from(tbody.rows);
      var asc = table.getAttribute('data-sort-col') == colIdx &&
                table.getAttribute('data-sort-dir') == 'asc' ? false : true;
      rows.sort(function(a, b) {
        var va = a.cells[colIdx].getAttribute('data-val') || a.cells[colIdx].textContent;
        var vb = b.cells[colIdx].getAttribute('data-val') || b.cells[colIdx].textContent;
        if (type === 'num') { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
        else { va = va.toLowerCase(); vb = vb.toLowerCase(); }
        if (va < vb) return asc ? -1 : 1;
        if (va > vb) return asc ? 1 : -1;
        return 0;
      });
      rows.forEach(function(r) { tbody.appendChild(r); });
      table.setAttribute('data-sort-col', colIdx);
      table.setAttribute('data-sort-dir', asc ? 'asc' : 'desc');
    }
    </script>
    </head>"""

        html = '<html>\n'
        html += style_header + '\n'
        html += '<body>\n'

        html += '<h1>Classification Analysis Results</h1>\n'

        # Summary / metadata
        html += '<div class="contentdiv">\n'
        html += '<p><b>Results file</b>: {}</p>\n'.format(os.path.basename(options.results_file))
        html += '<p><b>Ground truth file</b>: {}</p>\n'.format(os.path.basename(options.gt_file))
        html += '<p><b>Detection threshold</b>: {}</p>\n'.format(detection_threshold)
        html += '<p><b>Classification confidence threshold</b>: {}</p>\n'.format(
            options.classification_confidence_threshold)

        if options.sequence_level_analysis:
            html += '<p><b>Analysis level</b>: sequence</p>\n'
        else:
            html += '<p><b>Analysis level</b>: image</p>\n'

        analysis_unit = 'sequences' if options.sequence_level_analysis else 'images'
        html += '<p><b>Total {}</b>: {}</p>\n'.format(analysis_unit,
                                                       len(filename_to_gt_categories))

        if len(categories_to_ignore) > 0:
            html += '<p><b>Excluded categories</b>: {}</p>\n'.format(
                ', '.join(sorted(categories_to_ignore)))

        if options.single_label_per_image:
            html += '<p><b>Ground truth collapsed</b>: single label per {}</p>\n'.format(
                analysis_unit.rstrip('s'))

        if options.single_prediction_per_image:
            html += '<p><b>Predictions collapsed</b>: single label per {}</p>\n'.format(
                analysis_unit.rstrip('s'))

        html += '</div>\n'

        # Overall metrics
        html += '<h2>Overall Metrics</h2>\n'
        html += '<div class="contentdiv">\n'
        html += '<table class="metrics-table">\n'
        html += '<tr><td>Accuracy</td><td>{:.4f}</td></tr>\n'.format(accuracy)
        html += '<tr><td>Macro F1</td><td>{:.4f}</td></tr>\n'.format(macro_f1)
        html += '<tr><td>Micro F1</td><td>{:.4f}</td></tr>\n'.format(micro_f1)
        html += '<tr><td>Micro Precision</td><td>{:.4f}</td></tr>\n'.format(micro_precision)
        html += '<tr><td>Micro Recall</td><td>{:.4f}</td></tr>\n'.format(micro_recall)
        html += '</table>\n'
        html += '</div>\n'

        # Per-category statistics table (sortable)
        html += '<h2>Per-Category Statistics</h2>\n'
        html += '<p style="margin-left:50px;font-size:90%;">Click column headers to sort</p>\n'
        html += '<table class="result-table" id="stats-table">\n'
        html += '<thead><tr>'
        html += '<th class="sortable" onclick="sortTable(\'stats-table\',0,\'str\')">Category</th>'
        html += '<th class="sortable" onclick="sortTable(\'stats-table\',1,\'num\')">N (GT)</th>'
        html += '<th class="sortable" onclick="sortTable(\'stats-table\',2,\'num\')">N (Pred)</th>'
        html += '<th class="sortable" onclick="sortTable(\'stats-table\',3,\'num\')">Precision</th>'
        html += '<th class="sortable" onclick="sortTable(\'stats-table\',4,\'num\')">Recall</th>'
        html += '<th class="sortable" onclick="sortTable(\'stats-table\',5,\'num\')">F1</th>'
        html += '<th>Mispredicted as this category</th>'
        html += '<th>This was mispredicted as</th>'
        html += '</tr></thead>\n'
        html += '<tbody>\n'

        for cat in active_categories:
            r = per_category_results[cat]
            html += '<tr>'
            html += '<td>{}</td>'.format(cat)
            html += '<td data-val="{}">{}</td>'.format(r['n_ground_truth'], r['n_ground_truth'])
            html += '<td data-val="{}">{}</td>'.format(r['n_predicted'], r['n_predicted'])
            html += '<td data-val="{:.6f}">{:.3f}</td>'.format(r['precision'], r['precision'])
            html += '<td data-val="{:.6f}">{:.3f}</td>'.format(r['recall'], r['recall'])
            html += '<td data-val="{:.6f}">{:.3f}</td>'.format(r['f1'], r['f1'])

            # "Mispredicted as this category" column (FP: other things predicted as cat)
            fp_entries = mispredicted_as[cat]
            if len(fp_entries) == 0:
                html += '<td></td>'
            else:
                html += '<td class="mispred-cell">'
                fp_parts = []
                for other_cat, count in fp_entries:
                    link = _cell_html_filename(cat, other_cat)
                    fp_parts.append('<a href="{}">{}</a> ({})'.format(
                        link, other_cat, count))
                html += '<br/>'.join(fp_parts)
                html += '</td>'

            # "This was mispredicted as" column (FN: cat predicted as other things)
            fn_entries = this_mispredicted_as[cat]
            if len(fn_entries) == 0:
                html += '<td></td>'
            else:
                html += '<td class="mispred-cell">'
                fn_parts = []
                for other_cat, count in fn_entries:
                    link = _cell_html_filename(other_cat, cat)
                    fn_parts.append('<a href="{}">{}</a> ({})'.format(
                        link, other_cat, count))
                html += '<br/>'.join(fn_parts)
                html += '</td>'

            html += '</tr>\n'

        html += '</tbody>\n'
        html += '</table>\n'

        # Confusion matrix
        html += '<h2>Confusion Matrix</h2>\n'
        html += '<p>Rows = true categories, columns = predicted categories</p>\n'
        html += '<table class="result-table">\n'

        # Header row with rotated predicted category labels
        html += '<tr><td>&nbsp;</td>\n'
        for cat in active_categories:
            html += '<td class="rotate"><p style="margin-left:20px;">{}</p></td>\n'.format(cat)
        html += '</tr>\n'

        # Data rows
        for true_cat in active_categories:
            html += '<tr>\n'
            html += '<td><b>{}</b></td>\n'.format(true_cat)

            for pred_cat in active_categories:
                true_idx = category_to_index[true_cat]
                pred_idx = category_to_index[pred_cat]
                count = int(confusion_matrix[true_idx, pred_idx])

                if count == 0:
                    html += '<td></td>\n'
                else:
                    cell_filename = _cell_html_filename(pred_cat, true_cat)

                    # Highlight diagonal (correct predictions) with background color
                    if true_cat == pred_cat:
                        html += '<td style="background-color:#d4edda;">'
                    else:
                        html += '<td>'

                    html += '<a href="{}">{}</a></td>\n'.format(cell_filename, count)

            html += '</tr>\n'

        html += '</table>\n'

        html += '</body>\n'
        html += '</html>\n'

        html_output_file = os.path.join(options.html_output_dir, 'index.html')
        with open(html_output_file, 'w') as f:
            f.write(html)

        print('\nHTML output written to {}'.format(html_output_file))

    # ...if html_output_dir

    ## Build and return results

    results = AnalysisResults()
    results.per_category_results = per_category_results
    results.macro_f1 = float(macro_f1)
    results.micro_f1 = float(micro_f1)
    results.micro_precision = float(micro_precision)
    results.micro_recall = float(micro_recall)
    results.accuracy = float(accuracy)
    results.confusion_matrix = confusion_matrix
    results.active_categories = active_categories
    results.html_output_file = html_output_file

    return results

# ...def analyze_classification_results(...)


#%% Interactive driver

if False:

    #%%

    options = ClassificationAnalysisOptions()

    options.results_file = '/home/user/tmp/classification-analysis/subset_results.json'
    options.gt_file = '/home/user/tmp/classification-analysis/subset_gt.json'
    options.image_base_dir = '/home/user/tmp/images'
    options.html_output_dir = '/home/user/tmp/classification-analysis/html_output'
    options.sequence_level_analysis = False
    options.rendering_pool_type = 'processes'
    options.categories_to_ignore = ['vehicle','no cv result','setup_pickup']
    options.single_prediction_per_image = True
    options.single_label_per_image = True

    results = analyze_classification_results(options)

    from megadetector.utils.path_utils import open_file
    open_file(results.html_output_file)


#%% Command-line driver

def main():
    """
    Command-line driver for analyze_classification_results
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Analyze classification results against ground truth, '
                    'computing precision/recall/F1 and generating an HTML report.')

    parser.add_argument(
        'results_file', type=str,
        help='MD-formatted results file (.json)')

    parser.add_argument(
        'gt_file', type=str,
        help='Ground truth file in COCO Camera Traps format (.json)')

    parser.add_argument(
        '--image_base_dir', type=str, default=None,
        help='Folder where images live; required if --html_output_dir is specified')

    parser.add_argument(
        '--html_output_dir', type=str, default=None,
        help='Folder for HTML output with confusion matrix and image galleries')

    parser.add_argument(
        '--detection_threshold', type=float, default=None,
        help='Detection confidence threshold (auto-detected if not specified)')

    parser.add_argument(
        '--classification_confidence_threshold', type=float, default=0.5,
        help='Classification confidence threshold')

    parser.add_argument(
        '--max_total_images', type=int, default=8000,
        help='Maximum total number of images to render')

    parser.add_argument(
        '--max_images_per_cell', type=int, default=50,
        help='Maximum number of images per confusion matrix cell')

    parser.add_argument(
        '--random_seed', type=int, default=0,
        help='Random seed for image sampling')

    parser.add_argument(
        '--sequence_level', action='store_true',
        help='Perform analysis at the sequence level instead of image level')

    parser.add_argument(
        '--rendering_workers', type=int, default=10,
        help='Number of workers for image rendering')

    parser.add_argument(
        '--rendering_pool_type', type=str, default='threads',
        choices=['threads', 'processes'],
        help='Type of worker pool for rendering')

    parser.add_argument(
        '--output_image_width', type=int, default=1000,
        help='Width of rendered output images (-1 for original size)')

    parser.add_argument(
        '--n_mispredictions_for_table', type=int, default=3,
        help='Number of top misprediction categories to show in the per-category table')

    parser.add_argument(
        '--categories_to_ignore', type=str, default=None,
        help='Comma-separated list of category names to exclude from analysis')

    parser.add_argument(
        '--single_prediction_per_image', action='store_true',
        help='Collapse predictions to a single category per image/sequence')

    parser.add_argument(
        '--single_label_per_image', action='store_true',
        help='Collapse ground truth to a single category per image/sequence')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    options = ClassificationAnalysisOptions()
    options.results_file = args.results_file
    options.gt_file = args.gt_file
    options.image_base_dir = args.image_base_dir
    options.html_output_dir = args.html_output_dir
    options.detection_threshold = args.detection_threshold
    options.classification_confidence_threshold = args.classification_confidence_threshold
    options.max_total_images = args.max_total_images
    options.max_images_per_cell = args.max_images_per_cell
    options.random_seed = args.random_seed
    options.sequence_level_analysis = args.sequence_level
    options.rendering_workers = args.rendering_workers
    options.rendering_pool_type = args.rendering_pool_type
    options.output_image_width = args.output_image_width
    options.n_mispredictions_for_table = args.n_mispredictions_for_table
    if args.categories_to_ignore is not None:
        options.categories_to_ignore = [c.strip() for c in args.categories_to_ignore.split(',')]
    options.single_prediction_per_image = args.single_prediction_per_image
    options.single_label_per_image = args.single_label_per_image

    results = analyze_classification_results(options)

    print('\nResults:')
    print('  Accuracy: {:.4f}'.format(results.accuracy))
    print('  Macro F1: {:.4f}'.format(results.macro_f1))
    print('  Micro F1: {:.4f}'.format(results.micro_f1))

    return results


if __name__ == '__main__':
    main()
