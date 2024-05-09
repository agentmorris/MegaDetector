"""

annotation_constants.py

Defines default categories for MegaDetector output boxes.

Used throughout the repo; do not change unless you are Dan or Siyu.  In fact, do not change unless 
you are both Dan *and* Siyu.

We use integer IDs here; this is different from the MD .json file format,
where indices are string integers.

"""

#%% Constants

# MegaDetector output categories (the "empty" category is implicit)
detector_bbox_categories = [
    {'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'animal'},
    {'id': 2, 'name': 'person'},
    {'id': 3, 'name': 'vehicle'}
]

# This is used for choosing colors, so it ignores the "empty" class.
NUM_DETECTOR_CATEGORIES = len(detector_bbox_categories) - 1

detector_bbox_category_id_to_name = {}
detector_bbox_category_name_to_id = {}

for cat in detector_bbox_categories:
    detector_bbox_category_id_to_name[cat['id']] = cat['name']
    detector_bbox_category_name_to_id[cat['name']] = cat['id']

