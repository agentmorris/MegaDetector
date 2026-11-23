"""

annotation_constants.py

Defines the default categories for MegaDetector output boxes.  This is the one and only place
where these defaults are defined.

Models are not required to use these categories; detectors read their categories from the
model file where possible (see run_detector.get_detector_category_info()).  But all publicly
released MegaDetector models use these categories, and things like box colors throughout this
repo assume that "animal" is category "1".

Do not change unless you are Dan or Siyu.  In fact, do not change unless you are both Dan *and*
Siyu.

"""

#%% Constants

#: Default MegaDetector output categories, in the format used for the "detection_categories"
#: field in the MD output format (string integer IDs mapped to names).  Category "0" is
#: implicitly reserved for "empty", so these IDs start at 1.
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'
}

#: Number of default detector categories; this is used for choosing colors (e.g., classification
#: categories are colored starting after the detector categories).
NUM_DETECTOR_CATEGORIES = len(DEFAULT_DETECTOR_LABEL_MAP)
