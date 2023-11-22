########
#
# ct_utils.py
#
# Numeric/geometry utility functions
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
    Copies all fields from a Namespace (typically the output from parse_args) to an
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


def point_dist(p1,p2):
    """
    Distance between two points, represented as length-two tuples.
    """
    return math.sqrt( ((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2) )


def rect_distance(r1, r2, format='x0y0x1y1'):
    """
    Minimum distance between two axis-aligned rectangles, each represented as 
    (x0,y0,x1,y1) by default.
    
    Can also specify "format" as x0y0wh for MD-style bbox formatting (x0,y0,w,h).    
    """
    
    assert format in ('x0y0x1y1','x0y0wh')
    
    if format == 'x0y0wh':
        # Convert to x0y0x1y1 without modifying the original rectangles
        r1 = [r1[0],r1[1],r1[0]+r1[2],r1[1]+r1[3]]
        r2 = [r2[0],r2[1],r2[0]+r2[2],r2[1]+r2[3]]
        
    # https://stackoverflow.com/a/26178015
    x1, y1, x1b, y1b = r1
    x2, y2, x2b, y2b = r2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return point_dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return point_dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return point_dist((x1b, y1), (x2, y2b))
    elif right and top:
        return point_dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:
        return 0.0


def list_is_sorted(l):
    """
    Returns True if the list [l] is sorted, else False.
    """
    
    return all(l[i] <= l[i+1] for i in range(len(l)-1))


def split_list_into_fixed_size_chunks(L,n):
    """
    Split the list or tuple L into chunks of size n (allowing chunks of size n-1 if necessary,
    i.e. len(L) does not have to be a multiple of n.
    """
    
    return [L[i * n:(i + 1) * n] for i in range((len(L) + n - 1) // n )]


def split_list_into_n_chunks(L, n):
    """
    Splits the list or tuple L into n equally-sized chunks (some chunks may be one 
    element smaller than others, i.e. len(L) does not have to be a multiple of n.
    """
    
    k, m = divmod(len(L), n)
    return list(L[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def sort_dictionary_by_value(d,sort_values=None,reverse=False):
    """
    Sorts the dictionary [d] by value.  If sort_values is None, uses d.values(),
    otherwise uses the dictionary sort_values as the sorting criterion.
    """
    
    if sort_values is None:
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    else:
        d = {k: v for k, v in sorted(d.items(), key=lambda item: sort_values[item[0]], reverse=reverse)}
    return d


#%% Test drivers

if False:
    
    pass
    
    #%% Test a few rectangle distances
    
    r1 = [0,0,1,1]; r2 = [0,0,1,1]; assert rect_distance(r1,r2)==0
    r1 = [0,0,1,1]; r2 = [0,0,1,100]; assert rect_distance(r1,r2)==0
    r1 = [0,0,1,1]; r2 = [1,1,2,2]; assert rect_distance(r1,r2)==0
    r1 = [0,0,1,1]; r2 = [1.1,0,0,1.1]; assert abs(rect_distance(r1,r2)-.1) < 0.00001
    
    r1 = [0.4,0.8,10,22]; r2 = [100, 101, 200, 210.4]; assert abs(rect_distance(r1,r2)-119.753) < 0.001
    r1 = [0.4,0.8,10,22]; r2 = [101, 101, 200, 210.4]; assert abs(rect_distance(r1,r2)-120.507) < 0.001    
    r1 = [0.4,0.8,10,22]; r2 = [120, 120, 200, 210.4]; assert abs(rect_distance(r1,r2)-147.323) < 0.001
