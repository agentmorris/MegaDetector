########
#
# ct_utils.py
#
# Utility functions that don't depend on other things in this repo.
#
########

#%% Imports and constants

import argparse
import inspect
import json
import math
import os

import jsonpickle
import numpy as np

# List of file extensions we'll consider images; comparisons will be case-insensitive
# (i.e., no need to include both .jpg and .JPG on this list).
image_extensions = ['.jpg', '.jpeg', '.gif', '.png']


#%% Functions

def truncate_float_array(xs, precision=3):
    """
    Vectorized version of truncate_float(...)

    Args:
    xs        (list of float) List of floats to truncate
    precision (int)           The number of significant digits to preserve, should be
                              greater or equal 1
    """

    return [truncate_float(x, precision=precision) for x in xs]


def truncate_float(x, precision=3):
    """
    Truncates a floating-point value to a specific number of significant digits.
    
    For example: truncate_float(0.0003214884) --> 0.000321
    
    This function is primarily used to achieve a certain float representation
    before exporting to JSON.

    Args:
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserve, should be
                      greater or equal 1
    """

    assert precision > 0

    if np.isclose(x, 0):
        
        return 0
    
    else:
        
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit.
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor.
        return math.floor(x * factor)/factor


def args_to_object(args: argparse.Namespace, obj: object) -> None:
    """
    Copies all fields from a Namespace (i.e., the output from parse_args) to an
    object. Skips fields starting with _. Does not check existence in the target
    object.

    Args:
        args: argparse.Namespace
        obj: class or object whose whose attributes will be updated
    """
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)


def pretty_print_object(obj, b_print=True):
    """
    Prints an arbitrary object as .json
    """

    # _ = pretty_print_object(obj)

    # Sloppy that I'm making a module-wide change here...
    jsonpickle.set_encoder_options('json', sort_keys=True, indent=2)
    a = jsonpickle.encode(obj)
    s = '{}'.format(a)
    if b_print:
        print(s)
    return s


def is_list_sorted(L,reverse=False):
    """
    Returns true if the list L appears to be sorted, otherwise False.
    
    Calling is_list_sorted(L,reverse=True) is the same as calling
    is_list_sorted(L.reverse(),reverse=False).
    """
    
    if reverse:
        return all(L[i] >= L[i + 1] for i in range(len(L)-1))
    else:
        return all(L[i] <= L[i + 1] for i in range(len(L)-1))
        

def write_json(path, content, indent=1):
    """
    Standardized wrapper for json.dump
    """
    
    with open(path, 'w') as f:
        json.dump(content, f, indent=indent)


def is_image_file(s):
    """
    Checks a file's extension against a hard-coded set of image file extensions; 
    return True if it appears to be an image.
    """

    ext = os.path.splitext(s)[1]
    return ext.lower() in image_extensions


def convert_yolo_to_xywh(yolo_box):
    """
    Converts a YOLO format bounding box to [x_min, y_min, width_of_box, height_of_box].

    Args:
        yolo_box: bounding box of format [x_center, y_center, width_of_box, height_of_box].

    Returns:
        bbox with coordinates represented as [x_min, y_min, width_of_box, height_of_box].
    """
    
    x_center, y_center, width_of_box, height_of_box = yolo_box
    x_min = x_center - width_of_box / 2.0
    y_min = y_center - height_of_box / 2.0
    return [x_min, y_min, width_of_box, height_of_box]


def convert_xywh_to_tf(api_box):
    """
    Converts an xywh bounding box to an [y_min, x_min, y_max, x_max] box that the TensorFlow
    Object Detection API uses

    Args:
        api_box: bbox output by the batch processing API [x_min, y_min, width_of_box, height_of_box]

    Returns:
        bbox with coordinates represented as [y_min, x_min, y_max, x_max]
    """
    
    x_min, y_min, width_of_box, height_of_box = api_box
    x_max = x_min + width_of_box
    y_max = y_min + height_of_box
    return [y_min, x_min, y_max, x_max]


def convert_xywh_to_xyxy(api_bbox):
    """
    Converts an xywh bounding box to an xyxy bounding box.

    Note that this is also different from the TensorFlow Object Detection API coords format.
    Args:
        api_bbox: bbox output by the batch processing API [x_min, y_min, width_of_box, height_of_box]

    Returns:
        bbox with coordinates represented as [x_min, y_min, x_max, y_max]
    """

    x_min, y_min, width_of_box, height_of_box = api_bbox
    x_max, y_max = x_min + width_of_box, y_min + height_of_box
    return [x_min, y_min, x_max, y_max]


def get_iou(bb1, bb2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Adapted from:
        
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Args:
        bb1: [x_min, y_min, width_of_box, height_of_box]
        bb2: [x_min, y_min, width_of_box, height_of_box]

    Returns:
        intersection_over_union, a float in [0, 1]
    """

    bb1 = convert_xywh_to_xyxy(bb1)
    bb2 = convert_xywh_to_xyxy(bb2)

    assert bb1[0] < bb1[2], 'Malformed bounding box (x2 >= x1)'
    assert bb1[1] < bb1[3], 'Malformed bounding box (y2 >= y1)'

    assert bb2[0] < bb2[2], 'Malformed bounding box (x2 >= x1)'
    assert bb2[1] < bb2[3], 'Malformed bounding box (y2 >= y1)'

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0, 'Illegal IOU < 0'
    assert iou <= 1.0, 'Illegal IOU > 1'
    return iou


def _get_max_conf_from_detections(detections):
    """
    Internal function used by get_max_conf(); don't call this directly.
    """
    
    max_conf = 0.0
    if detections is not None and len(detections) > 0:
        confidences = [det['conf'] for det in detections]
        max_conf = max(confidences)
    return max_conf


def get_max_conf(im):
    """
    Given an image dict in the format used by the batch API, compute the maximum detection
    confidence for any class.  Returns 0.0 (not None) if there was a failure and 'detections'
    isn't present.
    """
    
    max_conf = 0.0
    if 'detections' in im and im['detections'] is not None and len(im['detections']) > 0:
        max_conf = _get_max_conf_from_detections(im['detections'])
    return max_conf
