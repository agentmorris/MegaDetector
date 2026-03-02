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

    # ...def __init__(...)

# ...class ClassificationAnalysisOptions

class AnalysisResultOptions:

    def __init__(self):

        #: Dictionary mapping category names to dicts, where each item has
        #: at least the keys "precision", "recall", "n_ground_truth", "n_predicted"
        self.precision_recall_results = None

        self.macro_f1 = None
        self.micro_f1 = None
        self.accuracy = None

# ...class AnalysisResultOptions


#%% Core functions

def analyze_classification_results(options):
    """
    Perform precision-recall analysis on classification results.

    Args:
        options (ClassificationAnalysisOptions): options object defining filenames
            and analysis parameters.

    Returns:
        ClassificationAnalysisResults: results of the classification analysis
    """

    pass


#%% Command-line driver
