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

        #: Confidence threshold to apply to classification (not detection) results
        self.classification_confidence_threshold = 0.5

    # ...def __init__(...)

# ...class ClassificationAnalysisOptions

class AnalysisResultOptions:

    def __init__(self):

        #: Dictionary mapping category names to dicts, where each item has
        #: at least the keys "precision", "recall", "n_ground_truth", "n_predicted"
        self.precision_recall_results = None

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
