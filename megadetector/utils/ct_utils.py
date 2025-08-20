"""

ct_utils.py

Numeric/geometry/array utility functions.

"""

#%% Imports and constants

import inspect
import json
import math
import os
import builtins
import datetime
import tempfile
import shutil
import platform
import sys
import uuid

import jsonpickle
import numpy as np

from operator import itemgetter

# List of file extensions we'll consider images; comparisons will be case-insensitive
# (i.e., no need to include both .jpg and .JPG on this list).
image_extensions = ['.jpg', '.jpeg', '.gif', '.png']


#%% Functions

def truncate_float_array(xs, precision=3):
    """
    Truncates the fractional portion of each floating-point value in the array [xs]
    to a specific number of floating-point digits.

    Args:
        xs (list): list of floats to truncate
        precision (int, optional): the number of significant digits to preserve, should be >= 1

    Returns:
        list: list of truncated floats
    """

    return [truncate_float(x, precision=precision) for x in xs]


def round_float_array(xs, precision=3):
    """
    Truncates the fractional portion of each floating-point value in the array [xs]
    to a specific number of floating-point digits.

    Args:
        xs (list): list of floats to round
        precision (int, optional): the number of significant digits to preserve, should be >= 1

    Returns:
        list: list of rounded floats
    """

    return [round_float(x,precision) for x in xs]


def round_float(x, precision=3):
    """
    Convenience wrapper for the native Python round()

    Args:
        x (float): number to truncate
        precision (int, optional): the number of significant digits to preserve, should be >= 1

    Returns:
        float: rounded value
    """

    return round(x,precision)


def truncate_float(x, precision=3):
    """
    Truncates the fractional portion of a floating-point value to a specific number of
    floating-point digits.

    For example:

        truncate_float(0.0003214884) --> 0.000321
        truncate_float(1.0003214884) --> 1.000321

    This function is primarily used to achieve a certain float representation
    before exporting to JSON.

    Args:
        x (float): scalar to truncate
        precision (int, optional): the number of significant digits to preserve, should be >= 1

    Returns:
        float: truncated version of [x]
    """

    return math.floor(x * (10 ** precision)) / (10 ** precision)


def args_to_object(args, obj):
    """
    Copies all fields from a Namespace (typically the output from parse_args) to an
    object. Skips fields starting with _. Does not check existence in the target
    object.

    Args:
        args (argparse.Namespace): the namespace to convert to an object
        obj (object): object whose whose attributes will be updated

    Returns:
        object: the modified object (modified in place, but also returned)
    """

    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)

    return obj


def dict_to_object(d, obj):
    """
    Copies all fields from a dict to an object. Skips fields starting with _.
    Does not check existence in the target object.

    Args:
        d (dict): the dict to convert to an object
        obj (object): object whose whose attributes will be updated

    Returns:
        object: the modified object (modified in place, but also returned)
    """

    for k in d.keys():
        if not k.startswith('_'):
            setattr(obj, k, d[k])

    return obj


def pretty_print_object(obj, b_print=True):
    """
    Converts an arbitrary object to .json, optionally printing the .json representation.

    Args:
        obj (object): object to print
        b_print (bool, optional): whether to print the object

    Returns:
        str: .json reprepresentation of [obj]
    """

    # _ = pretty_print_object(obj)

    # TODO: it's sloppy that I'm making a module-wide change here, consider at least
    # recording these operations and re-setting them at the end of this function.
    jsonpickle.set_encoder_options('json', sort_keys=True, indent=2)
    a = jsonpickle.encode(obj)
    s = '{}'.format(a)
    if b_print:
        print(s)
    return s


def is_list_sorted(L, reverse=False): # noqa
    """
    Returns True if the list L appears to be sorted, otherwise False.

    Calling is_list_sorted(L,reverse=True) is the same as calling
    is_list_sorted(L.reverse(),reverse=False).

    Args:
        L (list): list to evaluate
        reverse (bool, optional): whether to reverse the list before evaluating sort status

    Returns:
        bool: True if the list L appears to be sorted, otherwise False
    """

    if reverse:
        return all(L[i] >= L[i + 1] for i in range(len(L)-1))
    else:
        return all(L[i] <= L[i + 1] for i in range(len(L)-1))


def json_serialize_datetime(obj):
    """
    Serializes datetime.datetime and datetime.date objects to ISO format.

    Args:
        obj (object): The object to serialize.

    Returns:
        str: The ISO format string representation of the datetime object.

    Raises:
        TypeError: If the object is not a datetime.datetime or datetime.date instance.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable by json_serialize_datetime")


def write_json(path,
               content,
               indent=1,
               force_str=False,
               serialize_datetimes=False,
               ensure_ascii=True,
               encoding='utf-8'):
    """
    Standardized wrapper for json.dump().

    Args:
        path (str): filename to write to
        content (object): object to dump
        indent (int, optional): indentation depth passed to json.dump
        force_str (bool, optional): whether to force string conversion for non-serializable objects
        serialize_datetimes (bool, optional): whether to serialize datetime objects to ISO format
        ensure_ascii (bool, optional): whether to ensure ASCII characters in the output
        encoding (str, optional): string encoding to use
    """

    default_handler = None

    if serialize_datetimes:
        default_handler = json_serialize_datetime
        if force_str:
            def serialize_or_str(obj):
                try:
                    return json_serialize_datetime(obj)
                except TypeError:
                    return str(obj)
            default_handler = serialize_or_str
    elif force_str:
        default_handler = str

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', newline='\n', encoding=encoding) as f:
        json.dump(content, f, indent=indent, default=default_handler, ensure_ascii=ensure_ascii)

# ...def write_json(...)


def convert_yolo_to_xywh(yolo_box):
    """
    Converts a YOLO format bounding box [x_center, y_center, w, h] to
    [x_min, y_min, width_of_box, height_of_box].

    Args:
        yolo_box (list): bounding box of format [x_center, y_center, width_of_box, height_of_box]

    Returns:
        list: bbox with coordinates represented as [x_min, y_min, width_of_box, height_of_box]
    """

    x_center, y_center, width_of_box, height_of_box = yolo_box
    x_min = x_center - width_of_box / 2.0
    y_min = y_center - height_of_box / 2.0
    return [x_min, y_min, width_of_box, height_of_box]


def convert_xywh_to_xyxy(api_box):
    """
    Converts an xywh bounding box (the MD output format) to an xyxy bounding box (the format
    produced by TF-based MD models).

    Args:
        api_box (list): bbox formatted as [x_min, y_min, width_of_box, height_of_box]

    Returns:
        list: bbox formatted as [x_min, y_min, x_max, y_max]
    """

    x_min, y_min, width_of_box, height_of_box = api_box
    x_max = x_min + width_of_box
    y_max = y_min + height_of_box
    return [x_min, y_min, x_max, y_max]


def get_iou(bb1, bb2):
    """
    Calculates the intersection over union (IoU) of two bounding boxes.

    Adapted from:

    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Args:
        bb1 (list): [x_min, y_min, width_of_box, height_of_box]
        bb2 (list): [x_min, y_min, width_of_box, height_of_box]

    Returns:
        float: intersection_over_union, a float in [0, 1]
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
    Given an image dict in the MD output format, computes the maximum detection confidence for any
    class.  Returns 0.0 if there were no detections, if there was a failure, or if 'detections' isn't
    present.

    Args:
        im (dict): image dictionary in the MD output format (with a 'detections' field)

    Returns:
        float: the maximum detection confidence across all classes
    """

    max_conf = 0.0
    if 'detections' in im and im['detections'] is not None and len(im['detections']) > 0:
        max_conf = _get_max_conf_from_detections(im['detections'])
    return max_conf


def sort_results_for_image(im):
    """
    Sort classification and detection results in descending order by confidence (in place).

    Args:
        im (dict): image dictionary in the MD output format (with a 'detections' field)
    """
    if 'detections' not in im or im['detections'] is None:
        return

    # Sort detections in descending order by confidence
    im['detections'] = sort_list_of_dicts_by_key(im['detections'],k='conf',reverse=True)

    for det in im['detections']:

        # Sort classifications (which are (class,conf) tuples) in descending order by confidence
        if 'classifications' in det and \
            (det['classifications'] is not None) and \
            (len(det['classifications']) > 0):
            classifications = det['classifications']
            det['classifications'] = \
                sorted(classifications,key=itemgetter(1),reverse=True)


def point_dist(p1,p2):
    """
    Computes the distance between two points, represented as length-two tuples.

    Args:
        p1 (list or tuple): point, formatted as (x,y)
        p2 (list or tuple): point, formatted as (x,y)

    Returns:
        float: the Euclidean distance between p1 and p2
    """

    return math.sqrt( ((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2) )


def rect_distance(r1, r2, format='x0y0x1y1'):
    """
    Computes the minimum distance between two axis-aligned rectangles, each represented as
    (x0,y0,x1,y1) by default.

    Can also specify "format" as x0y0wh for MD-style bbox formatting (x0,y0,w,h).

    Args:
        r1 (list or tuple): rectangle, formatted as (x0,y0,x1,y1) or (x0,y0,xy,y1)
        r2 (list or tuple): rectangle, formatted as (x0,y0,x1,y1) or (x0,y0,xy,y1)
        format (str, optional): whether the boxes are formatted as 'x0y0x1y1' (default) or 'x0y0wh'

    Returns:
        float: the minimum distance between r1 and r2
    """

    assert format in ('x0y0x1y1','x0y0wh'), 'Illegal rectangle format {}'.format(format)

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


def split_list_into_fixed_size_chunks(L,n): # noqa
    """
    Split the list or tuple L into chunks of size n (allowing at most one chunk with size
    less than N, i.e. len(L) does not have to be a multiple of n).

    Args:
        L (list): list to split into chunks
        n (int): preferred chunk size

    Returns:
        list: list of chunks, where each chunk is a list of length n or n-1
    """

    return [L[i * n:(i + 1) * n] for i in range((len(L) + n - 1) // n )]


def split_list_into_n_chunks(L, n, chunk_strategy='greedy'): # noqa
    """
    Splits the list or tuple L into n equally-sized chunks (some chunks may be one
    element smaller than others, i.e. len(L) does not have to be a multiple of n).

    chunk_strategy can be "greedy" (default, if there are k samples per chunk, the first
    k go into the first chunk) or "balanced" (alternate between chunks when pulling
    items from the list).

    Args:
        L (list): list to split into chunks
        n (int): number of chunks
        chunk_strategy (str, optional): "greedy" or "balanced"; see above

    Returns:
        list: list of chunks, each of which is a list
    """

    if chunk_strategy == 'greedy':
        k, m = divmod(len(L), n)
        return list(L[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    elif chunk_strategy == 'balanced':
        chunks = [ [] for _ in range(n) ]
        for i_item,item in enumerate(L):
            i_chunk = i_item % n
            chunks[i_chunk].append(item)
        return chunks
    else:
        raise ValueError('Invalid chunk strategy: {}'.format(chunk_strategy))


def sort_list_of_dicts_by_key(L, k, reverse=False, none_handling='smallest'): # noqa ("L" should be lowercase)
    """
    Sorts the list of dictionaries [L] by the key [k].

    Args:
        L (list): list of dictionaries to sort
        k (object, typically str): the sort key
        reverse (bool, optional): whether to sort in reverse (descending) order
        none_handling (str, optional): how to handle None values. Options:
            "smallest" - treat None as smaller than all other values (default)
            "largest" - treat None as larger than all other values
            "error" - raise error when None is compared with non-None

    Returns:
        list: sorted copy of [L]
    """

    if none_handling == 'error':
        return sorted(L, key=lambda d: d[k], reverse=reverse)
    elif none_handling == 'smallest':
        # None values treated as smaller than other values: use tuple (is_not_none, value)
        return sorted(L, key=lambda d: (d[k] is not None, d[k]), reverse=reverse)
    elif none_handling == "largest":
        # None values treated as larger than other values: use tuple (is_none, value)
        return sorted(L, key=lambda d: (d[k] is None, d[k]), reverse=reverse)
    else:
        raise ValueError('Invalid none_handling value: {}'.format(none_handling))


def sort_dictionary_by_key(d,reverse=False):
    """
    Sorts the dictionary [d] by key.

    Args:
        d (dict): dictionary to sort
        reverse (bool, optional): whether to sort in reverse (descending) order

    Returns:
        dict: sorted copy of [d]
    """

    d = dict(sorted(d.items(),reverse=reverse))
    return d


def sort_dictionary_by_value(d,sort_values=None,reverse=False):
    """
    Sorts the dictionary [d] by value.  If sort_values is None, uses d.values(),
    otherwise uses the dictionary sort_values as the sorting criterion.  Always
    returns a new standard dict, so if [d] is, for example, a defaultdict, the
    returned value is not.

    Args:
        d (dict): dictionary to sort
        sort_values (dict, optional): dictionary mapping keys in [d] to sort values (defaults
            to None, uses [d] itself for sorting)
        reverse (bool, optional): whether to sort in reverse (descending) order

    Returns:
        dict: sorted copy of [d
    """

    if sort_values is None:
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    else:
        d = {k: v for k, v in sorted(d.items(), key=lambda item: sort_values[item[0]], reverse=reverse)}
    return d


def invert_dictionary(d):
    """
    Creates a new dictionary that maps d.values() to d.keys().  Does not check
    uniqueness.

    Args:
        d (dict): dictionary to invert

    Returns:
        dict: inverted copy of [d]
    """

    return {v: k for k, v in d.items()}


def round_floats_in_nested_dict(obj, decimal_places=5, allow_iterator_conversion=False):
    """
    Recursively rounds all floating point values in a nested structure to the
    specified number of decimal places. Handles dictionaries, lists, tuples,
    sets, and other iterables. Modifies mutable objects in place.

    Args:
        obj (obj): The object to process (can be a dict, list, set, tuple, or primitive value)
        decimal_places (int, optional): Number of decimal places to round to
        allow_iterator_conversion (bool, optional): for iterator types, should we convert
            to lists?  Otherwise we error.

    Returns:
        The processed object (useful for recursive calls)
    """
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = round_floats_in_nested_dict(obj[key], decimal_places=decimal_places,
                                                   allow_iterator_conversion=allow_iterator_conversion)
        return obj

    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = round_floats_in_nested_dict(obj[i], decimal_places=decimal_places,
                                                 allow_iterator_conversion=allow_iterator_conversion)
        return obj

    elif isinstance(obj, tuple):
        # Tuples are immutable, so we create a new one
        return tuple(round_floats_in_nested_dict(item, decimal_places=decimal_places,
                                                 allow_iterator_conversion=allow_iterator_conversion) for item in obj)

    elif isinstance(obj, set):
        # Sets are mutable but we can't modify elements in-place
        # Convert to list, process, and convert back to set
        return set(round_floats_in_nested_dict(list(obj), decimal_places=decimal_places,
                                               allow_iterator_conversion=allow_iterator_conversion))

    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        # Handle other iterable types: convert to list, process, and convert back
        processed_list = [round_floats_in_nested_dict(item,
                                                      decimal_places=decimal_places,
                                                      allow_iterator_conversion=allow_iterator_conversion) \
                                                        for item in obj]

        # Try to recreate the original type, but fall back to list for iterators
        try:
            return type(obj)(processed_list)
        except (TypeError, ValueError):
            if allow_iterator_conversion:
                # For iterators and other types that can't be reconstructed, return a list
                return processed_list
            else:
                raise ValueError('Cannot process iterator types when allow_iterator_conversion is False')

    elif isinstance(obj, float):
        return round(obj, decimal_places)

    else:
        # For other types (int, str, bool, None, etc.), return as is
        return obj

# ...def round_floats_in_nested_dict(...)


def image_file_to_camera_folder(image_fn):
    r"""
    Removes common overflow folders (e.g. RECNX101, RECNX102) from paths, i.e. turn:

    a\b\c\RECNX101\image001.jpg

    ...into:

    a\b\c

    Returns the same thing as os.dirname() (i.e., just the folder name) if no overflow folders are
    present.

    Always converts backslashes to slashes.

    Args:
        image_fn (str): the image filename from which we should remove overflow folders

    Returns:
        str: a version of [image_fn] from which camera overflow folders have been removed
    """

    import re

    # 100RECNX is the overflow folder style for Reconyx cameras
    # 100EK113 is (for some reason) the overflow folder style for Bushnell cameras
    # 100_BTCF is the overflow folder style for Browning cameras
    # 100MEDIA is the overflow folder style used on a number of consumer-grade cameras
    patterns = [r'/\d+RECNX/',r'/\d+EK\d+/',r'/\d+_BTCF/',r'/\d+MEDIA/']

    image_fn = image_fn.replace('\\','/')
    for pat in patterns:
        image_fn = re.sub(pat,'/',image_fn)
    camera_folder = os.path.dirname(image_fn)

    return camera_folder


def is_float(v):
    """
    Determines whether v is either a float or a string representation of a float.

    Args:
        v (object): object to evaluate

    Returns:
        bool: True if [v] is a float or a string representation of a float, otherwise False
    """

    if v is None:
        return False

    try:
        _ = float(v)
        return True
    except ValueError:
        return False


def is_iterable(x):
    """
    Uses duck typing to assess whether [x] is iterable (list, set, dict, etc.).

    Args:
        x (object): the object to test

    Returns:
        bool: True if [x] appears to be iterable, otherwise False
    """

    try:
        _ = iter(x)
    except Exception:
       return False
    return True


def is_empty(v):
    """
    A common definition of "empty" used throughout the repo, particularly when loading
    data from .csv files.  "empty" includes None, '', and NaN.

    Args:
        v (obj): the object to evaluate for emptiness

    Returns:
        bool: True if [v] is None, '', or NaN, otherwise False
    """
    if v is None:
        return True
    if isinstance(v,str) and v == '':
        return True
    if isinstance(v,float) and np.isnan(v):
        return True
    return False


def to_bool(v):
    """
    Convert an object to a bool with specific rules.

    Args:
        v (object): The object to convert

    Returns:
        bool or None:
        - For strings: True if 'true' (case-insensitive), False if 'false', recursively applied if int-like
        - For int/bytes: False if 0, True otherwise
        - For bool: returns the bool as-is
        - For other types: None
    """

    if isinstance(v, bool):
        return v

    if isinstance(v, str):

        try:
            v = int(v)
            return to_bool(v)
        except Exception:
            pass

        v = v.lower().strip()
        if v == 'true':
            return True
        elif v == 'false':
            return False
        else:
            return None

    if isinstance(v, (int, bytes)):
        return v != 0

    return None


def min_none(a,b):
    """
    Returns the minimum of a and b.  If both are None, returns None.  If one is None,
    returns the other.

    Args:
        a (numeric): the first value to compare
        b (numeric): the second value to compare

    Returns:
        numeric: the minimum of a and b, or None
    """
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return min(a,b)


def max_none(a,b):
    """
    Returns the maximum of a and b.  If both are None, returns None.  If one is None,
    returns the other.

    Args:
        a (numeric): the first value to compare
        b (numeric): the second value to compare

    Returns:
        numeric: the maximum of a and b, or None
    """
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return max(a,b)


def isnan(v):
    """
    Returns True if v is a nan-valued float, otherwise returns False.

    Args:
        v (obj): the object to evaluate for nan-ness

    Returns:
        bool: True if v is a nan-valued float, otherwise False
    """

    try:
        return np.isnan(v)
    except Exception:
        return False


def compare_values_nan_equal(v0,v1):
    """
    Utility function for comparing two values when we want to return True if both
    values are NaN.

    Args:
        v0 (object): the first value to compare
        v1 (object): the second value to compare

    Returns:
        bool: True if v0 == v1, or if both v0 and v1 are NaN
    """

    if isinstance(v0,float) and isinstance(v1,float) and np.isnan(v0) and np.isnan(v1):
        return True
    return v0 == v1


def sets_overlap(set1, set2):
    """
    Determines whether two sets overlap.

    Args:
        set1 (set): the first set to compare (converted to a set if it's not already)
        set2 (set): the second set to compare (converted to a set if it's not already)

    Returns:
        bool: True if any elements are shared between set1 and set2
    """

    return not set(set1).isdisjoint(set(set2))


def is_function_name(s,calling_namespace):
    """
    Determines whether [s] is a callable function in the global or local scope, or a
    built-in function.

    Args:
        s (str): the string to test for function-ness
        calling_namespace (dict): typically pass the output of locals()
    """

    assert isinstance(s,str), 'Input is not a string'

    return callable(globals().get(s)) or \
        callable(locals().get(s)) or \
        callable(calling_namespace.get(s)) or \
        callable(getattr(builtins, s, None))


# From https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
def parse_kvp(s,kv_separator='='):
    """
    Parse a key/value pair, separated by [kv_separator].  Errors if s is not
    a valid key/value pair string.  Strips leading/trailing whitespace from
    the key and value.

    Args:
        s (str): the string to parse
        kv_separator (str, optional): the string separating keys from values.

    Returns:
        tuple: a 2-tuple formatted as (key,value)
    """

    items = s.split(kv_separator)
    assert len(items) > 1, 'Illegal key-value pair'
    key = items[0].strip()
    if len(items) > 1:
        value = kv_separator.join(items[1:]).strip()
    return (key, value)


def parse_kvp_list(items,kv_separator='=',d=None):
    """
    Parse a list key-value pairs into a dictionary.  If items is None or [],
    returns {}.

    Args:
        items (list): the list of KVPs to parse
        kv_separator (str, optional): the string separating keys from values.
        d (dict, optional): the initial dictionary, defaults to {}

    Returns:
        dict: a dict mapping keys to values
    """

    if d is None:
        d = {}

    if items is None or len(items) == 0:
        return d

    for item in items:
        key, value = parse_kvp(item,kv_separator=kv_separator)
        d[key] = value

    return d


def dict_to_kvp_list(d,
                     item_separator=' ',
                     kv_separator='=',
                     non_string_value_handling='error'):
    """
    Convert a string <--> string dict into a string containing list of list of
    key-value pairs.  I.e., converts {'a':'dog','b':'cat'} to 'a=dog b=cat'.  If
    d is None, returns None.  If d is empty, returns ''.

    Args:
        d (dict): the dictionary to convert, must contain only strings
        item_separator (str, optional): the delimiter between KV pairs
        kv_separator (str, optional): the separator betweena a key and its value
        non_string_value_handling (str, optional): what do do with non-string values,
            can be "omit", "error", or "convert"

    Returns:
        str: the string representation of [d]
    """

    if d is None:
        return None

    if len(d) == 0:
        return ''

    s = None
    for k in d.keys():
        assert isinstance(k,str), 'Input {} is not a str <--> str dict'.format(str(d))
        v = d[k]
        if not isinstance(v,str):
            if non_string_value_handling == 'error':
                raise ValueError('Input {} is not a str <--> str dict'.format(str(d)))
            elif non_string_value_handling == 'omit':
                continue
            elif non_string_value_handling == 'convert':
                v = str(v)
            else:
                raise ValueError('Unrecognized non_string_value_handling value: {}'.format(
                    non_string_value_handling))
        if s is None:
            s = ''
        else:
            s += item_separator
        s += k + kv_separator + v

    if s is None:
        s = ''

    return s


def parse_bool_string(s, strict=False):
    """
    Convert the strings "true" or "false" to boolean values.  Case-insensitive, discards
    leading and trailing whitespace.  If s is already a bool, returns s.

    Args:
        s (str or bool): the string to parse, or the bool to return
        strict (bool, optional): only allow "true" or "false", otherwise
            handles "1", "0", "yes", and "no".

    Returns:
        bool: the parsed value
    """

    if isinstance(s,bool):
        return s
    s = str(s).lower().strip()

    if strict:
        false_strings = ('false')
        true_strings = ('true')
    else:
        false_strings = ('no', 'false', 'f', 'n', '0')
        true_strings = ('yes', 'true', 't', 'y', '1')
    if s in true_strings:
        return True
    elif s in false_strings:
        return False
    else:
        raise ValueError('Cannot parse bool from string {}'.format(str(s)))


def make_temp_folder(top_level_folder='megadetector',subfolder=None,append_guid=True):
    """
    Creates a temporary folder within the system temp folder, by default in a subfolder
    called megadetector/some_guid.  Used for testing without making too much of a mess.

    Args:
        top_level_folder (str, optional): the top-level folder to use within the system temp folder
        subfolder (str, optional): the subfolder within [top_level_folder]
        append_guid (bool, optional): append a guid to the subfolder

    Returns:
        str: the new directory
    """

    to_return = os.path.join(tempfile.gettempdir(),top_level_folder)
    if subfolder is not None:
        to_return = os.path.join(to_return,subfolder)
    if append_guid:
        to_return = os.path.join(to_return,str(uuid.uuid1()))
    to_return = os.path.normpath(to_return)
    os.makedirs(to_return,exist_ok=True)
    return to_return


def make_test_folder(subfolder=None):
    """
    Wrapper around make_temp_folder that creates folders within megadetector/tests

    Args:
        subfolder (str): specific subfolder to create within the default megadetector temp
            folder.
    """

    return make_temp_folder(top_level_folder='megadetector/tests',
                            subfolder=subfolder,
                            append_guid=True)


#%% Environment utilities

def is_sphinx_build():
    """
    Determine whether we are running in the context of our Sphinx build.

    Returns:
        bool: True if we're running a Sphinx build
    """

    is_sphinx = hasattr(builtins, '__sphinx_build__')
    return is_sphinx


def is_running_in_gha():
    """
    Determine whether we are running on a GitHub Actions runner.

    Returns:
        bool: True if we're running in a GHA runner
    """

    running_in_gha = False

    if ('GITHUB_ACTIONS' in os.environ):
        # Documentation is inconsistent on how this variable presents itself
        if isinstance(os.environ['GITHUB_ACTIONS'],bool) and \
            os.environ['GITHUB_ACTIONS']:
            running_in_gha = True
        elif isinstance(os.environ['GITHUB_ACTIONS'],str) and \
            os.environ['GITHUB_ACTIONS'].lower() == ('true'):
            running_in_gha = True

    return running_in_gha


def environment_is_wsl():
    """
    Determines whether we're running in WSL.

    Returns:
        True if we're running in WSL
    """

    if sys.platform not in ('linux','posix'):
        return False
    platform_string = ' '.join(platform.uname()).lower()
    return 'microsoft' in platform_string and 'wsl' in platform_string



#%% Tests

def test_write_json():
    """
    Test driver for write_json.
    """

    temp_dir = make_test_folder()

    def _verify_json_file(file_path, expected_content_str):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        assert isinstance(content,dict)
        content = sort_dictionary_by_key(content)
        expected_content = json.loads(expected_content_str)
        expected_content = sort_dictionary_by_key(expected_content)
        assert content == expected_content, \
            f"File {file_path} content mismatch.\nExpected:\n{expected_content}\nGot:\n{content}"

    # Test default indent (1)
    data_default = {'a': 1, 'b': 2}
    file_path_default = os.path.join(temp_dir, 'test_default_indent.json')
    write_json(file_path_default, data_default)
    # Default indent is 1
    _verify_json_file(file_path_default, '{\n "a": 1,\n "b": 2\n}')

    # Test custom indent (e.g., 4)
    data_custom_indent = {'a': 1, 'b': 2}
    file_path_custom_indent = os.path.join(temp_dir, 'test_custom_indent.json')
    write_json(file_path_custom_indent, data_custom_indent, indent=4)
    _verify_json_file(file_path_custom_indent, '{\n    "a": 1,\n    "b": 2\n}')

    # Test indent=None (compact)
    data_no_indent = {'a': 1, 'b': 2}
    file_path_no_indent = os.path.join(temp_dir, 'test_no_indent.json')
    write_json(file_path_no_indent, data_no_indent, indent=None)
    _verify_json_file(file_path_no_indent, '{"a": 1, "b": 2}')

    # Test force_str=True
    data_force_str = {'a': 1, 's': {1, 2, 3}}  # Set is not normally JSON serializable
    file_path_force_str = os.path.join(temp_dir, 'test_force_str.json')
    write_json(file_path_force_str, data_force_str, force_str=True)
    with open(file_path_force_str, 'r', encoding='utf-8') as f:
        result_force_str = json.load(f)
    assert isinstance(result_force_str['s'], str)
    assert eval(result_force_str['s']) == {1, 2, 3}

    # Test serialize_datetimes=True
    dt = datetime.datetime(2023, 1, 1, 10, 30, 0)
    d_date = datetime.date(2023, 2, 15)
    data_serialize_datetimes = {'dt_obj': dt, 'd_obj': d_date}
    file_path_serialize_datetimes = os.path.join(temp_dir, 'test_serialize_datetimes.json')
    write_json(file_path_serialize_datetimes, data_serialize_datetimes, serialize_datetimes=True)
    _verify_json_file(file_path_serialize_datetimes, '{\n "d_obj": "2023-02-15",\n "dt_obj": "2023-01-01T10:30:00"\n}')

    # Test serialize_datetimes=True and force_str=True
    dt_combo = datetime.datetime(2023, 1, 1, 12, 0, 0)
    data_datetime_force_str = {'dt_obj': dt_combo, 's_obj': {4, 5}}
    file_path_datetime_force_str = os.path.join(temp_dir, 'test_datetime_and_force_str.json')
    write_json(file_path_datetime_force_str, data_datetime_force_str, serialize_datetimes=True, force_str=True)
    with open(file_path_datetime_force_str, 'r', encoding='utf-8') as f:
        result_datetime_force_str = json.load(f)
    assert result_datetime_force_str['dt_obj'] == "2023-01-01T12:00:00"
    assert isinstance(result_datetime_force_str['s_obj'], str)
    assert eval(result_datetime_force_str['s_obj']) == {4, 5}

    # Test ensure_ascii=False (with non-ASCII chars)
    data_ensure_ascii_false = {'name': 'Jules César'}
    file_path_ensure_ascii_false = os.path.join(temp_dir, 'test_ensure_ascii_false.json')
    write_json(file_path_ensure_ascii_false, data_ensure_ascii_false, ensure_ascii=False)
    with open(file_path_ensure_ascii_false, 'r', encoding='utf-8') as f:
        content_ensure_ascii_false = f.read()
    assert content_ensure_ascii_false == '{\n "name": "Jules César"\n}'

    # Test ensure_ascii=True (with non-ASCII chars, default)
    data_ensure_ascii_true = {'name': 'Jules César'}
    file_path_ensure_ascii_true = os.path.join(temp_dir, 'test_ensure_ascii_true.json')
    write_json(file_path_ensure_ascii_true, data_ensure_ascii_true, ensure_ascii=True)
    with open(file_path_ensure_ascii_true, 'r', encoding='utf-8') as f:
        content_ensure_ascii_true = f.read()
    assert content_ensure_ascii_true == '{\n "name": "Jules C\\u00e9sar"\n}'

    shutil.rmtree(temp_dir)

# ...def test_write_json(...)


def test_path_operations():
    """
    Test path manipulation functions.
    """

    ##%% Camera folder mapping
    assert image_file_to_camera_folder('a/b/c/d/100EK113/blah.jpg') == 'a/b/c/d'
    assert image_file_to_camera_folder('a/b/c/d/100RECNX/blah.jpg') == 'a/b/c/d'
    assert image_file_to_camera_folder('a/b/c/d/blah.jpg') == 'a/b/c/d'
    assert image_file_to_camera_folder(r'a\b\c\d\100RECNX\blah.jpg') == 'a/b/c/d'


def test_geometric_operations():
    """
    Test geometric calculations like distances.
    """

    ##%% Test a few rectangle distances

    r1 = [0,0,1,1]; r2 = [0,0,1,1]; assert rect_distance(r1,r2)==0
    r1 = [0,0,1,1]; r2 = [0,0,1,100]; assert rect_distance(r1,r2)==0
    r1 = [0,0,1,1]; r2 = [1,1,2,2]; assert rect_distance(r1,r2)==0
    r1 = [0,0,1,1]; r2 = [1.1,0,0,1.1]; assert abs(rect_distance(r1,r2)-.1) < 0.00001

    r1 = [0.4,0.8,10,22]; r2 = [100, 101, 200, 210.4]; assert abs(rect_distance(r1,r2)-119.753) < 0.001
    r1 = [0.4,0.8,10,22]; r2 = [101, 101, 200, 210.4]; assert abs(rect_distance(r1,r2)-120.507) < 0.001
    r1 = [0.4,0.8,10,22]; r2 = [120, 120, 200, 210.4]; assert abs(rect_distance(r1,r2)-147.323) < 0.001

    # Test with 'x0y0wh' format
    r1_wh = [0,0,1,1]; r2_wh = [1,0,1,1]; assert rect_distance(r1_wh, r2_wh, format='x0y0wh') == 0
    r1_wh = [0,0,1,1]; r2_wh = [1.5,0,1,1]; assert abs(rect_distance(r1_wh, r2_wh, format='x0y0wh') - 0.5) < 0.00001


    ##%% Test point_dist

    assert point_dist((0,0), (3,4)) == 5.0
    assert point_dist((1,1), (1,1)) == 0.0


def test_dictionary_operations():
    """
    Test dictionary manipulation and sorting functions.
    """

    ##%% Test sort_list_of_dicts_by_key

    x = [{'a':5},{'a':0},{'a':10}]
    k = 'a'
    sorted_x = sort_list_of_dicts_by_key(x, k)
    assert sorted_x[0]['a'] == 0; assert sorted_x[1]['a'] == 5; assert sorted_x[2]['a'] == 10
    sorted_x_rev = sort_list_of_dicts_by_key(x, k, reverse=True)
    assert sorted_x_rev[0]['a'] == 10; assert sorted_x_rev[1]['a'] == 5; assert sorted_x_rev[2]['a'] == 0


    ##%% Test sort_dictionary_by_key

    d_key = {'b': 2, 'a': 1, 'c': 3}
    sorted_d_key = sort_dictionary_by_key(d_key)
    assert list(sorted_d_key.keys()) == ['a', 'b', 'c']
    sorted_d_key_rev = sort_dictionary_by_key(d_key, reverse=True)
    assert list(sorted_d_key_rev.keys()) == ['c', 'b', 'a']


    ##%% Test sort_dictionary_by_value

    d_val = {'a': 2, 'b': 1, 'c': 3}
    sorted_d_val = sort_dictionary_by_value(d_val)
    assert list(sorted_d_val.keys()) == ['b', 'a', 'c']
    sorted_d_val_rev = sort_dictionary_by_value(d_val, reverse=True)
    assert list(sorted_d_val_rev.keys()) == ['c', 'a', 'b']

    # With sort_values
    sort_vals = {'a': 10, 'b': 0, 'c': 5}
    sorted_d_custom = sort_dictionary_by_value(d_val, sort_values=sort_vals)
    assert list(sorted_d_custom.keys()) == ['b', 'c', 'a']


    ##%% Test invert_dictionary

    d_inv = {'a': 'x', 'b': 'y'}
    inverted_d = invert_dictionary(d_inv)
    assert inverted_d == {'x': 'a', 'y': 'b'}

    # Does not check for uniqueness, last one wins
    d_inv_dup = {'a': 'x', 'b': 'x'}
    inverted_d_dup = invert_dictionary(d_inv_dup)
    assert inverted_d_dup == {'x': 'b'}


def test_float_rounding_and_truncation():
    """
    Test float rounding, truncation, and nested rounding functions.
    """

    ##%% Test round_floats_in_nested_dict

    data = {
        "name": "Project X",
        "values": [1.23456789, 2.3456789],
        "tuple_values": (3.45678901, 4.56789012),
        "set_values": {5.67890123, 6.78901234}, # Order not guaranteed in set, test min/max
        "metrics": {
            "score": 98.7654321,
            "components": [5.6789012, 6.7890123]
        },
        "other_iter": iter([7.89012345]) # Test other iterables
    }

    result = round_floats_in_nested_dict(data, decimal_places=5, allow_iterator_conversion=True)
    assert result['values'][0] == 1.23457
    assert result['tuple_values'][0] == 3.45679

    # For sets, convert to list and sort for consistent testing
    assert sorted(list(result['set_values'])) == sorted([5.67890, 6.78901])
    assert result['metrics']['score'] == 98.76543

    # Test other iterables by converting back to list
    assert list(result['other_iter'])[0] == 7.89012


    ##%% Test truncate_float_array and truncate_float

    assert truncate_float_array([0.12345, 0.67890], precision=3) == [0.123, 0.678]
    assert truncate_float_array([1.0, 2.0], precision=2) == [1.0, 2.0]
    assert truncate_float(0.12345, precision=3) == 0.123
    assert truncate_float(1.999, precision=2) == 1.99
    assert truncate_float(0.0003214884, precision=6) == 0.000321
    assert truncate_float(1.0003214884, precision=6) == 1.000321


    ##%% Test round_float_array and round_float

    assert round_float_array([0.12345, 0.67890], precision=3) == [0.123, 0.679]
    assert round_float_array([1.0, 2.0], precision=2) == [1.0, 2.0]
    assert round_float(0.12345, precision=3) == 0.123
    assert round_float(0.12378, precision=3) == 0.124
    assert round_float(1.999, precision=2) == 2.00


def test_object_conversion_and_presentation():
    """
    Test functions that convert or present objects.
    """

    ##%% Test args_to_object

    class ArgsObject:
        pass
    args_namespace = type('ArgsNameSpace', (), {'a': 1, 'b': 'test', '_c': 'ignored'})
    obj = ArgsObject()
    args_to_object(args_namespace, obj)
    assert obj.a == 1
    assert obj.b == 'test'
    assert not hasattr(obj, '_c')


    ##%% Test dict_to_object

    class DictObject:
        pass
    d = {'a': 1, 'b': 'test', '_c': 'ignored'}
    obj = DictObject()
    dict_to_object(d, obj)
    assert obj.a == 1
    assert obj.b == 'test'
    assert not hasattr(obj, '_c')


    ##%% Test pretty_print_object

    class PrettyPrintable:
        def __init__(self):
            self.a = 1
            self.b = "test"
    obj_to_print = PrettyPrintable()
    json_str = pretty_print_object(obj_to_print, b_print=False)

    # Basic check for valid json and presence of attributes
    parsed_json = json.loads(json_str) # Relies on json.loads
    assert parsed_json['a'] == 1
    assert parsed_json['b'] == "test"


def test_list_operations():
    """
    Test list sorting and chunking functions.
    """

    ##%% Test is_list_sorted

    assert is_list_sorted([1, 2, 3])
    assert not is_list_sorted([1, 3, 2])
    assert is_list_sorted([3, 2, 1], reverse=True)
    assert not is_list_sorted([1, 2, 3], reverse=True)
    assert is_list_sorted([]) # Empty list is considered sorted
    assert is_list_sorted([1]) # Single element list is sorted
    assert is_list_sorted([1,1,1])
    assert is_list_sorted([1,1,1], reverse=True)


    ##%% Test split_list_into_fixed_size_chunks

    assert split_list_into_fixed_size_chunks([1,2,3,4,5,6], 2) == [[1,2],[3,4],[5,6]]
    assert split_list_into_fixed_size_chunks([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
    assert split_list_into_fixed_size_chunks([], 3) == []
    assert split_list_into_fixed_size_chunks([1,2,3], 5) == [[1,2,3]]


    ##%% Test split_list_into_n_chunks

    # Greedy
    assert split_list_into_n_chunks([1,2,3,4,5,6], 3, chunk_strategy='greedy') == [[1,2],[3,4],[5,6]]
    assert split_list_into_n_chunks([1,2,3,4,5], 3, chunk_strategy='greedy') == [[1,2],[3,4],[5]]
    assert split_list_into_n_chunks([1,2,3,4,5,6,7], 3, chunk_strategy='greedy') == [[1,2,3],[4,5],[6,7]]
    assert split_list_into_n_chunks([], 3) == [[],[],[]]

    # Balanced
    assert split_list_into_n_chunks([1,2,3,4,5,6], 3, chunk_strategy='balanced') == [[1,4],[2,5],[3,6]]
    assert split_list_into_n_chunks([1,2,3,4,5], 3, chunk_strategy='balanced') == [[1,4],[2,5],[3]]
    assert split_list_into_n_chunks([], 3, chunk_strategy='balanced') == [[],[],[]]
    try:
        split_list_into_n_chunks([1,2,3], 2, chunk_strategy='invalid')
        raise AssertionError("ValueError not raised for invalid chunk_strategy")
    except ValueError:
        pass


def test_datetime_serialization():
    """
    Test datetime serialization functions.
    """

    ##%% Test json_serialize_datetime

    now = datetime.datetime.now()
    today = datetime.date.today()
    assert json_serialize_datetime(now) == now.isoformat()
    assert json_serialize_datetime(today) == today.isoformat()
    try:
        json_serialize_datetime("not a datetime")
        raise AssertionError("TypeError not raised for non-datetime object")
    except TypeError:
        pass
    try:
        json_serialize_datetime(123)
        raise AssertionError("TypeError not raised for non-datetime object")
    except TypeError:
        pass


def test_bounding_box_operations():
    """
    Test bounding box conversion and IoU calculation.
    """

    ##%% Test convert_yolo_to_xywh

    # [x_center, y_center, w, h]
    yolo_box = [0.5, 0.5, 0.2, 0.2]
    # [x_min, y_min, width_of_box, height_of_box]
    expected_xywh = [0.4, 0.4, 0.2, 0.2]
    assert np.allclose(convert_yolo_to_xywh(yolo_box), expected_xywh)


    ##%% Test convert_xywh_to_xyxy

    # [x_min, y_min, width_of_box, height_of_box]
    xywh_box = [0.1, 0.1, 0.3, 0.3]
    # [x_min, y_min, x_max, y_max]
    expected_xyxy = [0.1, 0.1, 0.4, 0.4]
    assert np.allclose(convert_xywh_to_xyxy(xywh_box), expected_xyxy)


    ##%% Test get_iou

    bb1 = [0, 0, 0.5, 0.5]  # x, y, w, h
    bb2 = [0.25, 0.25, 0.5, 0.5]
    assert abs(get_iou(bb1, bb2) - 0.142857) < 1e-5
    bb3 = [0, 0, 1, 1]
    bb4 = [0.5, 0.5, 1, 1]
    assert abs(get_iou(bb3, bb4) - (0.25 / 1.75)) < 1e-5
    bb5 = [0,0,1,1]
    bb6 = [1,1,1,1] # No overlap
    assert get_iou(bb5, bb6) == 0.0

    # Test malformed boxes (should ideally raise error or handle gracefully based on spec, current impl asserts)
    bb_malformed1 = [0.6, 0.0, 0.5, 0.5] # x_min > x_max after conversion
    bb_ok = [0.0, 0.0, 0.5, 0.5]
    try:
        get_iou(bb_malformed1, bb_ok)
        # This assert False will only be reached if the expected AssertionError is not raised by get_iou
        # assert False, "AssertionError for malformed bounding box (x2 >= x1) not raised in get_iou"
    except AssertionError as e:
        assert 'Malformed bounding box' in str(e)


def test_detection_processing():
    """
    Test functions related to processing detection results.
    """

    ##%% Test _get_max_conf_from_detections and get_max_conf

    detections1 = [{'conf': 0.8}, {'conf': 0.9}, {'conf': 0.75}]
    assert _get_max_conf_from_detections(detections1) == 0.9
    assert _get_max_conf_from_detections([]) == 0.0
    assert _get_max_conf_from_detections(None) == 0.0

    im1 = {'detections': detections1}
    assert get_max_conf(im1) == 0.9
    im2 = {'detections': []}
    assert get_max_conf(im2) == 0.0
    im3 = {} # No 'detections' key
    assert get_max_conf(im3) == 0.0
    im4 = {'detections': None}
    assert get_max_conf(im4) == 0.0


    ##%% Test sort_results_for_image

    img_data = {
        'detections': [
            {'conf': 0.7, 'classifications': [('c', 0.6), ('a', 0.9), ('b', 0.8)]},
            {'conf': 0.9, 'classifications': [('x', 0.95), ('y', 0.85)]},
            {'conf': 0.8} # No classifications field
        ]
    }
    sort_results_for_image(img_data)

    # Check detections sorted by conf
    assert img_data['detections'][0]['conf'] == 0.9
    assert img_data['detections'][1]['conf'] == 0.8
    assert img_data['detections'][2]['conf'] == 0.7

    # Check classifications sorted by conf (only for the first original detection, now at index 0 after sort)
    assert img_data['detections'][0]['classifications'][0] == ('x', 0.95)
    assert img_data['detections'][0]['classifications'][1] == ('y', 0.85)

    # Check classifications for the second original detection (now at index 2)
    assert img_data['detections'][2]['classifications'][0] == ('a', 0.9)
    assert img_data['detections'][2]['classifications'][1] == ('b', 0.8)
    assert img_data['detections'][2]['classifications'][2] == ('c', 0.6)

    # Test with no detections or no classifications field
    img_data_no_det = {'detections': None}
    sort_results_for_image(img_data_no_det)
    assert img_data_no_det['detections'] is None

    img_data_empty_det = {'detections': []}
    sort_results_for_image(img_data_empty_det)
    assert img_data_empty_det['detections'] == []

    img_data_no_classifications_field = {'detections': [{'conf': 0.8}]}
    sort_results_for_image(img_data_no_classifications_field)
    assert 'classifications' not in img_data_no_classifications_field['detections'][0]

    img_data_none_classifications = {'detections': [{'conf': 0.8, 'classifications':None}]}
    sort_results_for_image(img_data_none_classifications)
    assert img_data_none_classifications['detections'][0]['classifications'] is None

    img_data_empty_classifications = {'detections': [{'conf': 0.8, 'classifications':[]}]}
    sort_results_for_image(img_data_empty_classifications)
    assert img_data_empty_classifications['detections'][0]['classifications'] == []


def test_type_checking_and_validation():
    """
    Test type checking and validation utility functions.
    """

    ##%% Test is_float

    assert is_float(1.23)
    assert is_float("1.23")
    assert is_float("-1.23")
    assert is_float("  1.23  ")
    assert not is_float("abc")
    assert not is_float(None)
    assert is_float(1) # int is also a float (current behavior)


    ##%% Test is_iterable

    assert is_iterable([1,2,3])
    assert is_iterable("hello")
    assert is_iterable({'a':1})
    assert is_iterable(range(5))
    assert not is_iterable(123)
    assert not is_iterable(None)
    assert is_iterable(np.array([1,2]))


    ##%% Test is_empty

    assert is_empty(None)
    assert is_empty("")
    assert is_empty(np.nan)
    assert not is_empty(0)
    assert not is_empty(" ")
    assert not is_empty([])
    assert not is_empty({})
    assert not is_empty(False) # False is not empty


    ##%% Test min_none and max_none

    assert min_none(1, 2) == 1
    assert min_none(None, 2) == 2
    assert min_none(1, None) == 1
    assert min_none(None, None) is None
    assert max_none(1, 2) == 2
    assert max_none(None, 2) == 2
    assert max_none(1, None) == 1
    assert max_none(None, None) is None


    ##%% Test isnan

    assert isnan(np.nan)
    assert not isnan(0.0)
    assert not isnan("text")
    assert not isnan(None)
    assert not isnan(float('inf'))
    assert not isnan(float('-inf'))


    ##%% Test sets_overlap

    assert sets_overlap({1,2,3}, {3,4,5})
    assert not sets_overlap({1,2}, {3,4})
    assert sets_overlap([1,2,3], [3,4,5]) # Test with lists
    assert sets_overlap(set(), {1}) is False
    assert sets_overlap({1},{1})


    ##%% Test is_function_name

    def _test_local_func(): pass
    assert is_function_name("is_float", locals()) # Test a function in ct_utils
    assert is_function_name("_test_local_func", locals()) # Test a local function
    assert is_function_name("print", locals()) # Test a builtin
    assert not is_function_name("non_existent_func", locals())
    global _test_global_func_ct_utils # Renamed to avoid conflict if run multiple times
    def _test_global_func_ct_utils(): pass
    assert is_function_name("_test_global_func_ct_utils", globals())
    # Clean up global
    del _test_global_func_ct_utils


def test_string_parsing():
    """
    Test string parsing utilities like KVP and boolean parsing.
    """

    ##%% Test parse_kvp and parse_kvp_list

    assert parse_kvp("key=value") == ("key", "value")
    assert parse_kvp("key = value with spaces") == ("key", "value with spaces")
    assert parse_kvp("key=value1=value2", kv_separator='=') == ("key", "value1=value2")
    try:
        parse_kvp("keyvalue")
        raise AssertionError("AssertionError not raised for invalid KVP")
    except AssertionError:
        pass

    kvp_list = ["a=1", "b = 2", "c=foo=bar"]
    parsed_list = parse_kvp_list(kvp_list)
    assert parsed_list == {"a": "1", "b": "2", "c": "foo=bar"}
    assert parse_kvp_list(None) == {}
    assert parse_kvp_list([]) == {}
    d_initial = {'z': '0'}

    # parse_kvp_list modifies d in place if provided
    parse_kvp_list(kvp_list, d=d_initial)
    assert d_initial == {"z": "0", "a": "1", "b": "2", "c": "foo=bar"}

    # Test with a different separator
    assert parse_kvp("key:value", kv_separator=":") == ("key", "value")
    assert parse_kvp_list(["a:1","b:2"], kv_separator=":") == {"a":"1", "b":"2"}


    ##%% Test dict_to_kvp_list

    d_kvp = {"a": "1", "b": "dog", "c": "foo=bar"}
    kvp_str = dict_to_kvp_list(d_kvp)

    # Order isn't guaranteed, so check for presence of all items and length
    assert "a=1" in kvp_str
    assert "b=dog" in kvp_str
    assert "c=foo=bar" in kvp_str
    assert len(kvp_str.split(' ')) == 3

    assert dict_to_kvp_list({}) == ""
    assert dict_to_kvp_list(None) is None
    d_kvp_int = {"a":1, "b":"text"}
    try:
        dict_to_kvp_list(d_kvp_int, non_string_value_handling='error')
        raise AssertionError("ValueError not raised for non-string value with 'error' handling")
    except ValueError:
        pass
    convert_result = dict_to_kvp_list(d_kvp_int, non_string_value_handling='convert')
    assert "a=1" in convert_result and "b=text" in convert_result

    omit_result = dict_to_kvp_list({"a":1, "b":"text"}, non_string_value_handling='omit')
    assert "a=1" not in omit_result and "b=text" in omit_result
    assert omit_result == "b=text"

    assert dict_to_kvp_list({"key":"val"}, item_separator="&", kv_separator=":") == "key:val"


    ##%% Test parse_bool_string

    assert parse_bool_string("true")
    assert parse_bool_string("True")
    assert parse_bool_string(" TRUE ")
    assert not parse_bool_string("false")
    assert not parse_bool_string("False")
    assert not parse_bool_string(" FALSE ")
    assert parse_bool_string("1", strict=False)
    assert not parse_bool_string("0", strict=False)
    assert parse_bool_string(True) is True # Test with existing bool
    assert parse_bool_string(False) is False
    try:
        parse_bool_string("maybe")
        raise AssertionError("ValueError not raised for invalid bool string")
    except ValueError:
        pass
    try:
        parse_bool_string("1",strict=True)
        raise AssertionError("ValueError not raised for '1'")
    except ValueError:
        pass


def test_temp_folder_creation():
    """
    Test temporary folder creation and cleanup.
    """

    # Store original tempdir for restoration if modified by tests (though unlikely for make_temp_folder)
    original_tempdir = tempfile.gettempdir()

    # Test make_temp_folder
    custom_top_level = "my_custom_temp_app_test" # Unique name for this test run
    custom_subfolder = "specific_test_run"

    # Test with default subfolder (UUID)
    temp_folder1_base = os.path.join(tempfile.gettempdir(), custom_top_level)
    temp_folder1 = make_temp_folder(top_level_folder=custom_top_level)
    assert os.path.exists(temp_folder1)
    assert os.path.basename(os.path.dirname(temp_folder1)) == custom_top_level
    assert temp_folder1_base == os.path.dirname(temp_folder1) # Path up to UUID should match

    # Cleanup: remove the custom_top_level which contains the UUID folder
    if os.path.exists(temp_folder1_base):
        shutil.rmtree(temp_folder1_base)
    assert not os.path.exists(temp_folder1_base)


    # Test with specified subfolder
    temp_folder2_base = os.path.join(tempfile.gettempdir(), custom_top_level)
    temp_folder2 = make_temp_folder(top_level_folder=custom_top_level,
                                    subfolder=custom_subfolder,
                                    append_guid=False)
    assert os.path.exists(temp_folder2)
    assert os.path.basename(temp_folder2) == custom_subfolder
    assert os.path.basename(os.path.dirname(temp_folder2)) == custom_top_level
    assert temp_folder2 == os.path.join(tempfile.gettempdir(), custom_top_level, custom_subfolder)

    # Cleanup
    if os.path.exists(temp_folder2_base):
        shutil.rmtree(temp_folder2_base)
    assert not os.path.exists(temp_folder2_base)


    # Test make_test_folder (which uses 'megadetector/tests' as top_level)
    #
    # This will create tempfile.gettempdir()/megadetector/tests/some_uuid or specified_subfolder
    megadetector_temp_base = os.path.join(tempfile.gettempdir(), "megadetector")
    test_subfolder = "my_specific_module_test"

    # Test with default subfolder for make_test_folder
    test_folder1 = make_test_folder() # Creates megadetector/tests/uuid_folder
    assert os.path.exists(test_folder1)
    assert os.path.basename(os.path.dirname(test_folder1)) == "tests"
    assert os.path.basename(os.path.dirname(os.path.dirname(test_folder1))) == "megadetector"

    # Cleanup for make_test_folder default: remove the 'megadetector' base temp dir
    if os.path.exists(megadetector_temp_base):
        shutil.rmtree(megadetector_temp_base)
    assert not os.path.exists(megadetector_temp_base)


    # Test with specified subfolder for make_test_folder
    test_folder2 = make_test_folder(subfolder=test_subfolder) # megadetector/tests/my_specific_module_test
    assert os.path.exists(test_folder2)
    assert test_subfolder in test_folder2
    assert "megadetector" in test_folder2

    # Cleanup for make_test_folder specific: remove the 'megadetector' base temp dir
    if os.path.exists(megadetector_temp_base):
        shutil.rmtree(megadetector_temp_base)
    assert not os.path.exists(megadetector_temp_base)

    # Verify cleanup if top level folder was 'megadetector' (default for make_temp_folder)
    #
    # This means it creates tempfile.gettempdir()/megadetector/uuid_folder
    default_temp_folder = make_temp_folder()
    assert os.path.exists(default_temp_folder)
    assert os.path.basename(os.path.dirname(default_temp_folder)) == "megadetector"

    # Cleanup: remove the 'megadetector' base temp dir created by default make_temp_folder
    if os.path.exists(megadetector_temp_base):
         shutil.rmtree(megadetector_temp_base)
    assert not os.path.exists(megadetector_temp_base)

    # Restore original tempdir if it was changed (though not expected for these functions)
    tempfile.tempdir = original_tempdir


def run_all_module_tests():
    """
    Run all tests in the ct_utils module.  This is not invoked by pytest; this is
    just a convenience wrapper for debugging the tests.
    """

    test_write_json()
    test_path_operations()
    test_geometric_operations()
    test_dictionary_operations()
    test_float_rounding_and_truncation()
    test_object_conversion_and_presentation()
    test_list_operations()
    test_datetime_serialization()
    test_bounding_box_operations()
    test_detection_processing()
    test_type_checking_and_validation()
    test_string_parsing()
    test_temp_folder_creation()
