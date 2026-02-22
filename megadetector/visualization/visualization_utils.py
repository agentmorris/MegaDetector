"""

visualization_utils.py

Rendering functions shared across visualization scripts

"""

#%% Constants and imports

import time
import numpy as np
import requests
import os
import cv2

from io import BytesIO
from PIL import Image, ImageFile, ImageFont, ImageDraw, ImageFilter
from PIL.ExifTags import TAGS
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial

from megadetector.utils.path_utils import find_images
from megadetector.data_management.annotations import annotation_constants
from megadetector.data_management.annotations.annotation_constants import \
    detector_bbox_category_id_to_name
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Maps EXIF standard rotation identifiers to degrees.  The value "1" indicates no
# rotation; this will be ignored.  The values 2, 4, 5, and 7 are mirrored rotations,
# which are not supported (we'll assert() on this when we apply rotations).
EXIF_IMAGE_NO_ROTATION = 1
EXIF_IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}

TEXTALIGN_LEFT = 0
TEXTALIGN_RIGHT = 1
TEXTALIGN_CENTER = 2

VTEXTALIGN_TOP = 0
VTEXTALIGN_BOTTOM = 1

# Convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

# Constants controlling retry behavior when fetching images from URLs
n_retries = 10
retry_sleep_time = 0.01

# If we try to open an image from a URL, and we encounter any error in this list,
# we'll retry, otherwise it's just an error.
error_names_for_retry = ['ConnectionError']

DEFAULT_BOX_THICKNESS = 4
DEFAULT_LABEL_FONT_SIZE = 16

# Default color map for mapping integer category IDs to colors when rendering bounding
# boxes
DEFAULT_COLORS = [
    'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua', 'Azure',
    'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
    'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'Aquamarine', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

pil_tag_name_to_id = {v: k for k, v in TAGS.items()}


#%% Functions

def open_image(input_file, ignore_exif_rotation=False):
    """
    Opens an image in binary format using PIL.Image and converts to RGB mode.

    Supports local files or URLs.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.  load_image() is the non-lazy version of this function.

    Args:
        input_file (str or BytesIO): can be a path to an image file (anything
            that PIL can open), a URL, or an image as a stream of bytes
        ignore_exif_rotation (bool, optional): don't rotate the loaded pixels,
            even if we are loading a JPEG and that JPEG says it should be rotated

    Returns:
        PIL.Image.Image: A PIL Image object in RGB mode
    """

    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        try:
            response = requests.get(input_file)
        except Exception as e:
            print(f'Error retrieving image {input_file}: {e}')
            success = False
            if e.__class__.__name__ in error_names_for_retry:
                for i_retry in range(0,n_retries):
                    try:
                        time.sleep(retry_sleep_time)
                        response = requests.get(input_file)
                    except Exception as e:
                        print(f'Error retrieving image {input_file} on retry {i_retry}: {e}')
                        continue
                    print('Succeeded on retry {}'.format(i_retry))
                    success = True
                    break
            if not success:
                raise
        try:
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise

    else:
        image = Image.open(input_file)

    # Convert to RGB if necessary
    if image.mode not in ('RGBA', 'RGB', 'L', 'I;16'):
        raise AttributeError(
            f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    if not ignore_exif_rotation:
        # Alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
        #
        # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
        # https://www.media.mit.edu/pia/Research/deepview/exif.html
        #
        try:
            exif = image._getexif()
            orientation: int = exif.get(274, None)
            if (orientation is not None) and (orientation != EXIF_IMAGE_NO_ROTATION):
                assert orientation in EXIF_IMAGE_ROTATIONS, \
                    'Mirrored rotations are not supported'
                image = image.rotate(EXIF_IMAGE_ROTATIONS[orientation], expand=True)
        except Exception:
            pass

    return image

# ...def open_image(...)


def _remove_exif_tags(pil_image, tags_to_remove):
    """
    Remove a set of tags by name from [pil_image]
    """

    exif = pil_image.getexif()
    if exif is not None:
        for tag_name in tags_to_remove:
            if tag_name in pil_tag_name_to_id:
                exif.pop(pil_tag_name_to_id[tag_name], None)
    return exif

# ..._remove_exif_tags


def exif_preserving_save(pil_image,
                         output_file,
                         quality='keep',
                         default_quality=85,
                         verbose=False,
                         tags_to_exclude=None):
    """
    Saves [pil_image] to [output_file], making a moderate attempt to preserve EXIF
    data and JPEG quality.  Neither is guaranteed.

    Also see:

    https://discuss.dizzycoding.com/determining-jpg-quality-in-python-pil/

    ...for more ways to preserve jpeg quality if quality='keep' doesn't do the trick.

    Args:
        pil_image (Image): the PIL Image object to save
        output_file (str): the destination file
        quality (str or int, optional): can be "keep" (default), or an integer from 0 to 100.
            This is only used if PIL thinks the the source image is a JPEG.  If you load a JPEG
            and resize it in memory, for example, it's no longer a JPEG.
        default_quality (int, optional): determines output quality when quality == 'keep' and we are
            saving a non-JPEG source to a JPEG file
        verbose (bool, optional): enable additional debug console output
        tags_to_exclude (list, optional): tags to exclude from the output file
    """

    # Read EXIF metadata
    # exif = pil_image.info['exif'] if ('exif' in pil_image.info) else None
    exif = pil_image.getexif()

    if isinstance(tags_to_exclude,str):
        tags_to_exclude = [tags_to_exclude]

    # Optionally remove some tags
    if (exif is not None) and (tags_to_exclude is not None):
        exif = _remove_exif_tags(pil_image,
                                 tags_to_remove=tags_to_exclude)

    # Quality preservation is only supported for JPEG sources.
    if pil_image.format != "JPEG":
        if quality == 'keep':
            if verbose:
                print('Warning: quality "keep" passed when saving a non-JPEG source (during save to {})'.format(
                    output_file))
            quality = default_quality

    # Some output formats don't support the quality parameter, so we try once with,
    # and once without.  This is a horrible cascade of if's, but it's a consequence of
    # the fact that "None" is not supported for either "exif" or "quality".

    try:

        if exif is not None:
            pil_image.save(output_file, exif=exif, quality=quality)
        else:
            pil_image.save(output_file, quality=quality)

    except Exception:

        if verbose:
            print('Warning: failed to write {}, trying again without quality parameter'.format(output_file))
        if exif is not None:
            pil_image.save(output_file, exif=exif)
        else:
            pil_image.save(output_file)

# ...def exif_preserving_save(...)


def load_image(input_file, ignore_exif_rotation=False):
    """
    Loads an image file.  This is the non-lazy version of open_file(); i.e.,
    it forces image decoding before returning.

    Args:
        input_file (str or BytesIO): can be a path to an image file (anything
            that PIL can open), a URL, or an image as a stream of bytes
        ignore_exif_rotation (bool, optional): don't rotate the loaded pixels,
            even if we are loading a JPEG and that JPEG says it should be rotated

    Returns:
        PIL.Image.Image: a PIL Image object in RGB mode
    """

    image = open_image(input_file, ignore_exif_rotation=ignore_exif_rotation)
    image.load()
    return image


def resize_image(image,
                 target_width=-1,
                 target_height=-1,
                 output_file=None,
                 no_enlarge_width=False,
                 verbose=False,
                 quality='keep'):
    """
    Resizes a PIL Image object to the specified width and height; does not resize
    in place. If either width or height are -1, resizes with aspect ratio preservation.

    If target_width and target_height are both -1, does not modify the image, but
    will write to output_file if supplied.

    If no resizing is required, and an Image object is supplied, returns the original Image
    object (i.e., does not copy).

    Args:
        image (Image or str): PIL Image object or a filename (local file or URL)
        target_width (int, optional): width to which we should resize this image, or -1
            to let target_height determine the size
        target_height (int, optional): height to which we should resize this image, or -1
            to let target_width determine the size
        output_file (str, optional): file to which we should save this image; if None,
            just returns the image without saving
        no_enlarge_width (bool, optional): if [no_enlarge_width] is True, and
            [target width] is larger than the original image width, does not modify the image,
            but will write to output_file if supplied
        verbose (bool, optional): enable additional debug output
        quality (str or int, optional): passed to exif_preserving_save, see docs for more detail

    Returns:
        PIL.Image.Image: the resized image, which may be the original image if no resizing is
            required
    """

    image_fn = 'in_memory'
    if isinstance(image,str):
        image_fn = image
        image = load_image(image)

    if target_width is None:
        target_width = -1

    if target_height is None:
        target_height = -1

    resize_required = True

    # No resize was requested, this is always a no-op
    if target_width == -1 and target_height == -1:

        resize_required = False

    # Does either dimension need to scale according to the other?
    elif target_width == -1 or target_height == -1:

        # Aspect ratio as width over height
        # ar = w / h
        aspect_ratio = image.size[0] / image.size[1]

        if target_width != -1:
            # h = w / ar
            target_height = int(target_width / aspect_ratio)
        else:
            # w = ar * h
            target_width = int(aspect_ratio * target_height)

    # If we're not enlarging images and this would be an enlarge operation
    if (no_enlarge_width) and (target_width > image.size[0]):

        if verbose:
            print('Bypassing image enlarge for {} --> {}'.format(
                image_fn,str(output_file)))
        resize_required = False

    # If the target size is the same as the original size
    if (target_width == image.size[0]) and (target_height == image.size[1]):

        resize_required = False

    if not resize_required:

        if output_file is not None:
            if verbose:
                print('No resize required for resize {} --> {}'.format(
                    image_fn,str(output_file)))
            exif_preserving_save(image,output_file,quality=quality,verbose=verbose)
        return image

    assert target_width > 0 and target_height > 0, \
        'Invalid image resize target {},{}'.format(target_width,target_height)

    # The antialiasing parameter changed between Pillow versions 9 and 10, and for a bit,
    # I'd like to support both.
    try:
        resized_image = image.resize((target_width, target_height), Image.ANTIALIAS)
    except Exception:
        resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    if output_file is not None:
        exif_preserving_save(resized_image,output_file,quality=quality,verbose=verbose)

    return resized_image

# ...def resize_image(...)


def crop_image(detections, image, confidence_threshold=0.15, expansion=0):
    """
    Crops detections above [confidence_threshold] from the PIL image [image],
    returning a list of PIL Images, preserving the order of [detections].

    Args:
        detections (list): a list of dictionaries with keys 'conf' and 'bbox';
            boxes are length-four arrays formatted as [x,y,w,h], normalized,
            upper-left origin (this is the standard MD detection format)
        image (Image or str): the PIL Image object from which we should crop detections,
            or an image filename
        confidence_threshold (float, optional): only crop detections above this threshold
        expansion (int, optional): a number of pixels to include on each side of a cropped
            detection

    Returns:
        list: a possibly-empty list of PIL Image objects
    """

    ret_images = []

    if isinstance(image,str):
        image = load_image(image)

    for detection in detections:

        score = float(detection['conf'])

        if (confidence_threshold is None) or (score >= confidence_threshold):

            x1, y1, w_box, h_box = detection['bbox']
            ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box

            # Convert to pixels so we can use the PIL crop() function
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            if expansion > 0:
                left -= expansion
                right += expansion
                top -= expansion
                bottom += expansion

            # PIL's crop() does surprising things if you provide values outside of
            # the image, clip inputs
            left = max(left,0); right = max(right,0)
            top = max(top,0); bottom = max(bottom,0)

            left = min(left,im_width-1); right = min(right,im_width-1)
            top = min(top,im_height-1); bottom = min(bottom,im_height-1)

            ret_images.append(image.crop((left, top, right, bottom)))

        # ...if this detection is above threshold

    # ...for each detection

    return ret_images

# ...def crop_image(...)


def blur_detections(image,detections,blur_radius=40):
    """
    Blur the regions in [image] corresponding to the MD-formatted list [detections].
    [image] is modified in place.

    Args:
        image (PIL.Image.Image): image in which we should blur specific regions
        detections (list): list of detections in the MD output format, see render
            detection_bounding_boxes for more detail.
        blur_radius (int, optional): radius of blur kernel in pixels
    """

    img_width, img_height = image.size

    for d in detections:

        bbox = d['bbox']
        x_norm, y_norm, width_norm, height_norm = bbox

        # Calculate absolute pixel coordinates
        x = int(x_norm * img_width)
        y = int(y_norm * img_height)
        width = int(width_norm * img_width)
        height = int(height_norm * img_height)

        # Calculate box boundaries
        left = max(0, x)
        top = max(0, y)
        right = min(img_width, x + width)
        bottom = min(img_height, y + height)

        # Crop the region, blur it, and paste it back
        region = image.crop((left, top, right, bottom))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_region, (left, top))

    # ...for each detection

# ...def blur_detections(...)


def render_detection_bounding_boxes(detections,
                                    image,
                                    label_map='show_categories',
                                    classification_label_map='show_categories',
                                    confidence_threshold=0.0,
                                    thickness=DEFAULT_BOX_THICKNESS,
                                    expansion=0,
                                    classification_confidence_threshold=0.3,
                                    max_classifications=3,
                                    colormap=None,
                                    textalign=TEXTALIGN_LEFT,
                                    vtextalign=VTEXTALIGN_TOP,
                                    label_font_size=DEFAULT_LABEL_FONT_SIZE,
                                    custom_strings=None,
                                    box_sort_order='confidence',
                                    verbose=False):
    """
    Renders bounding boxes (with labels and confidence values) on an image for all
    detections above a threshold.

    Renders classification labels if present.

    [image] is modified in place.

    Args:
        detections (list): list of detections in the MD output format, for example:

            .. code-block::none

                [
                    {
                        "category": "2",
                        "conf": 0.996,
                        "bbox": [
                            0.0,
                            0.2762,
                            0.1234,
                            0.2458
                        ]
                    }
                ]

                ...where the bbox coordinates are [x, y, box_width, box_height].

                (0, 0) is the upper-left.  Coordinates are normalized.

            Supports classification results, in the standard format:

            .. code-block::none

                [
                    {
                        "category": "2",
                        "conf": 0.996,
                        "bbox": [
                            0.0,
                            0.2762,
                            0.1234,
                            0.2458
                        ]
                        "classifications": [
                            ["3", 0.901],
                            ["1", 0.071],
                            ["4", 0.025]
                        ]
                    }
                ]

        image (PIL.Image.Image): image on which we should render detections
        label_map (dict, optional): optional, mapping the numeric label to a string name. The type of the
            numeric label (typically strings) needs to be consistent with the keys in label_map; no casting is
            carried out. If [label_map] is None, no labels are shown (not even numbers and confidence values).
            If you want category numbers and confidence values without class labels, use the default value,
            the string 'show_categories'.
        classification_label_map (dict, optional): optional, mapping of the string class labels to the actual
            class names. The type of the numeric label (typically strings) needs to be consistent with the keys
            in label_map; no casting is carried out. If [label_map] is None, no labels are shown (not even numbers
            and confidence values).  If you want category numbers and confidence values without class labels, use
            the default value, the string 'show_categories'.
        confidence_threshold (float or dict, optional): threshold above which boxes are rendered.  Can also be a
            dictionary mapping category IDs to thresholds.
        thickness (int or float, optional): line thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        expansion (int or float, optional): number of pixels to expand bounding boxes on each side.  If
            this is a float less than 1.0, it's treated as a fraction of the image width.
        classification_confidence_threshold (float, optional): confidence above which classification results
            are displayed
        max_classifications (int, optional): maximum number of classification results rendered for one image
        colormap (list, optional): list of color names, used to choose colors for categories by
            indexing with the values in [classes]; defaults to a reasonable set of colors
        textalign (int, optional): TEXTALIGN_LEFT, TEXTALIGN_CENTER, or TEXTALIGN_RIGHT
        vtextalign (int, optional): VTEXTALIGN_TOP or VTEXTALIGN_BOTTOM
        label_font_size (float, optional): font size in pixels.  If this is less than one, it's treated
            as a fraction of the image width.
        custom_strings (list of str, optional): optional set of strings to append to detection labels, should
            have the same length as [detections].  Appended before any classification labels.
        box_sort_order (str, optional): sorting scheme for detection boxes, can be None, "confidence", or
            "reverse_confidence".  "confidence" puts the highest-confidence boxes on top.
        verbose (bool, optional): enable additional debug output
    """

    # Input validation
    if (label_map is not None) and (isinstance(label_map,str)) and (label_map == 'show_categories'):
        label_map = {}

    if custom_strings is not None:
        assert len(custom_strings) == len(detections), \
            '{} custom strings provided for {} detections'.format(
                len(custom_strings),len(detections))

    display_boxes = []

    # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    display_strs = []

    # for color selection
    classes = []

    if box_sort_order is not None:

        if box_sort_order == 'confidence':
            detections = sort_list_of_dicts_by_key(detections,k='conf',reverse=False)
        elif box_sort_order == 'reverse_confidence':
            detections = sort_list_of_dicts_by_key(detections,k='conf',reverse=True)
        else:
            raise ValueError('Unrecognized sorting scheme {}'.format(box_sort_order))

    for i_detection,detection in enumerate(detections):

        score = detection['conf']

        if isinstance(confidence_threshold,dict):
            rendering_threshold = confidence_threshold[detection['category']]
        else:
            rendering_threshold = confidence_threshold

        # Always render objects with a confidence of "None", this is typically used
        # for ground truth data.
        if (score is None) or (rendering_threshold is None) or (score >= rendering_threshold):

            x1, y1, w_box, h_box = detection['bbox']
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])

            # The class index to use for coloring this box, which may be based on the detection
            # category or on the most confident classification category.
            clss = detection['category']

            # This will be a list of strings that should be rendered above/below this box
            displayed_label = []

            if label_map is not None:
                label = label_map[clss] if clss in label_map else clss
                if score is not None:
                    displayed_label = ['{}: {}%'.format(label, round(100 * score))]
                else:
                    displayed_label = ['{}'.format(label)]
            else:
                displayed_label = ['']

            if custom_strings is not None:
                custom_string = custom_strings[i_detection]
                if custom_string is not None and len(custom_string) > 0:
                    assert len(displayed_label) == 1
                    displayed_label[0] += ' ' + custom_string

            if ('classifications' in detection) and len(detection['classifications']) > 0:

                classifications = detection['classifications']

                if len(classifications) > max_classifications:
                    classifications = classifications[0:max_classifications]

                max_classification_category = 0
                max_classification_conf = -100

                # Should we render classification categories?
                if classification_label_map is not None:

                    show_category_names = True

                    if isinstance(classification_label_map,str):
                        assert classification_label_map == 'show_categories', \
                            'Unknown value for classification_label_map: {}'.format(
                                classification_label_map)
                        show_category_names = False

                    for classification in classifications:

                        classification_conf = classification[1]
                        if classification_conf is None or \
                        classification_conf < classification_confidence_threshold:
                            continue

                        class_key = classification[0]

                        # Is this the most confident classification for this detection?
                        if classification_conf > max_classification_conf:
                            max_classification_conf = classification_conf
                            max_classification_category = int(class_key)

                        if (show_category_names) and (class_key in classification_label_map):
                            class_name = classification_label_map[class_key]
                        else:
                            class_name = class_key
                        if classification_conf is not None:
                            displayed_label += ['{}: {}%'.format(
                                class_name.lower(), round(100.0 * classification_conf))]
                        else:
                            displayed_label += ['{}'.format(class_name.lower())]

                    # ...for each classification

                # ...if we're supposed to show classification categories

                # To avoid duplicate colors with detection-only visualization, offset
                # the classification class index by the number of detection classes
                clss = annotation_constants.NUM_DETECTOR_CATEGORIES + max_classification_category

            # ...if we have classification results

            # display_strs is a list of labels for each box
            display_strs.append(displayed_label)
            classes.append(clss)

        # ...if the confidence of this detection is above threshold

    # ...for each detection

    display_boxes = np.array(display_boxes)

    if verbose:
        print('Rendering {} of {} detections'.format(len(display_boxes),len(detections)))

    draw_bounding_boxes_on_image(image,
                                 display_boxes,
                                 classes,
                                 display_strs=display_strs,
                                 thickness=thickness,
                                 expansion=expansion,
                                 colormap=colormap,
                                 textalign=textalign,
                                 vtextalign=vtextalign,
                                 label_font_size=label_font_size)

# ...render_detection_bounding_boxes(...)


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 classes,
                                 thickness=DEFAULT_BOX_THICKNESS,
                                 expansion=0,
                                 display_strs=None,
                                 colormap=None,
                                 textalign=TEXTALIGN_LEFT,
                                 vtextalign=VTEXTALIGN_TOP,
                                 text_rotation=None,
                                 label_font_size=DEFAULT_LABEL_FONT_SIZE):
    """
    Draws bounding boxes on an image.  Modifies the image in place.

    Args:
        image (PIL.Image): the image on which we should draw boxes
        boxes (np.array): a two-dimensional numpy array of size [N, 4], where N is the
            number of boxes, and each row is (ymin, xmin, ymax, xmax).  Coordinates should be
            normalized to image height/width.
        classes (list): a list of ints or string-formatted ints corresponding to the
             class labels of the boxes. This is only used for color selection.  Should have the same
             length as [boxes].
        thickness (int or float, optional): line thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        expansion (int or float, optional): number of pixels to expand bounding boxes on each side.  If
            this is a float less than 1.0, it's treated as a fraction of the image width.
        display_strs (list, optional): list of list of strings (the outer list should have the
            same length as [boxes]).  Typically this is used to show (possibly multiple) detection
            or classification categories and/or confidence values.
        colormap (list, optional): list of color names, used to choose colors for categories by
            indexing with the values in [classes]; defaults to a reasonable set of colors
        textalign (int, optional): TEXTALIGN_LEFT, TEXTALIGN_CENTER, or TEXTALIGN_RIGHT
        vtextalign (int, optional): VTEXTALIGN_TOP or VTEXTALIGN_BOTTOM
        text_rotation (float, optional): rotation to apply to text
        label_font_size (float, optional): font size in pixels.  If this is less than one, it's treated
            as a fraction of the image width.
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        return
    for i in range(boxes_shape[0]):
        display_str_list = None
        if display_strs:
            display_str_list = display_strs[i]
        draw_bounding_box_on_image(image,
                                   boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                                   classes[i],
                                   thickness=thickness, expansion=expansion,
                                   display_str_list=display_str_list,
                                   colormap=colormap,
                                   textalign=textalign,
                                   vtextalign=vtextalign,
                                   text_rotation=text_rotation,
                                   label_font_size=label_font_size)

# ...draw_bounding_boxes_on_image(...)


def get_text_size(font,s):
    """
    Get the expected width and height when rendering the string [s] in the font
    [font].

    Args:
        font (PIL.ImageFont): the font whose size we should query
        s (str): the string whose size we should query

    Returns:
        tuple: (w,h), both floats in pixel coordinates
    """

    # This is what we did w/Pillow 9
    # w,h = font.getsize(s)

    # I would *think* this would be the equivalent for Pillow 10
    # l,t,r,b = font.getbbox(s); w = r-l; h=b-t

    # ...but this actually produces the most similar results to Pillow 9
    # l,t,r,b = font.getbbox(s); w = r; h=b

    try:
        l,t,r,b = font.getbbox(s); w = r; h=b # noqa
    except Exception:
        w,h = font.getsize(s)

    return w,h


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               clss=None,
                               thickness=DEFAULT_BOX_THICKNESS,
                               expansion=0,
                               display_str_list=None,
                               use_normalized_coordinates=True,
                               label_font_size=DEFAULT_LABEL_FONT_SIZE,
                               colormap=None,
                               textalign=TEXTALIGN_LEFT,
                               vtextalign=VTEXTALIGN_TOP,
                               text_rotation=None):
    """
    Adds a bounding box to an image.  Modifies the image in place.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Adapted from:

    https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

    Args:
        image (PIL.Image.Image): the image on which we should draw a box
        ymin (float): ymin of bounding box
        xmin (float): xmin of bounding box
        ymax (float): ymax of bounding box
        xmax (float): xmax of bounding box
        clss (int, optional): the class index of the object in this bounding box, used for choosing
            a color; should be either an integer or a string-formatted integer
        thickness (int or float, optional): line thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        expansion (int or float, optional): number of pixels to expand bounding boxes on each side.  If
            this is a float less than 1.0, it's treated as a fraction of the image width.
        display_str_list (list, optional): list of strings to display above the box (each to be shown on its
            own line)
        use_normalized_coordinates (bool, optional): if True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image, otherwise coordinates as absolute pixel values
        label_font_size (float, optional): font size in pixels.  If this is less than one, it's treated
            as a fraction of the image width.
        colormap (list, optional): list of color names, used to choose colors for categories by
            indexing with the values in [classes]; defaults to a reasonable set of colors
        textalign (int, optional): TEXTALIGN_LEFT, TEXTALIGN_CENTER, or TEXTALIGN_RIGHT
        vtextalign (int, optional): VTEXTALIGN_TOP or VTEXTALIGN_BOTTOM
        text_rotation (float, optional): rotation to apply to text
    """

    if colormap is None:
        colormap = DEFAULT_COLORS

    if display_str_list is None:
        display_str_list = []

    if clss is None:
        # Default to the MegaDetector animal class ID (1)
        color = colormap[1]
    else:
        color = colormap[int(clss) % len(colormap)]

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    # Resolve fractional (image-width-relative) values for thickness, expansion,
    # and label_font_size.  If any of these is between 0 and 1 (exclusive), interpret
    # it as a fraction of image width.
    assert thickness != 0.0, 'thickness cannot be zero'
    assert label_font_size != 0.0, 'label_font_size cannot be zero'

    # Clamp thickness and label_font_size to a minimum of 1 pixel, since a value
    # of zero would be meaningless for line thickness and font size.  Expansion is
    # not clamped, since 0 is a valid value (no expansion).
    if 0 < thickness < 1:
        thickness = max(1, round(thickness * im_width))
    if 0 < expansion < 1:
        expansion = round(expansion * im_width)
    if 0 < label_font_size < 1:
        label_font_size = max(1, round(label_font_size * im_width))

    # Ensure these are ints regardless of whether fractional resolution occurred,
    # since PIL requires int values for line thickness and font size.
    thickness = int(thickness)
    expansion = int(expansion)
    label_font_size = int(label_font_size)

    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    if expansion > 0:

        left -= expansion
        right += expansion
        top -= expansion
        bottom += expansion

        # Deliberately trimming to the width of the image only in the case where
        # box expansion is turned on.  There's not an obvious correct behavior here,
        # but the thinking is that if the caller provided an out-of-range bounding
        # box, they meant to do that, but at least in the eyes of the person writing
        # this comment, if you expand a box for visualization reasons, you don't want
        # to end up with part of a box.
        #
        # A slightly more sophisticated might check whether it was in fact the expansion
        # that made this box larger than the image, but this is the case 99.999% of the time
        # here, so that doesn't seem necessary.
        left = max(left,0); right = max(right,0)
        top = max(top,0); bottom = max(bottom,0)

        left = min(left,im_width-1); right = min(right,im_width-1)
        top = min(top,im_height-1); bottom = min(bottom,im_height-1)

    # ...if we need to expand boxes

    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    if display_str_list is not None:

        try:
            font = ImageFont.truetype('arial.ttf', label_font_size)
        except OSError:
            font = ImageFont.load_default()

        display_str_heights = [get_text_size(font,ds)[1] for ds in display_str_list]

        # Each display_str has a top and bottom margin of 0.05x the total label height.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        # Reverse list and print from bottom to top
        for i_str,display_str in enumerate(display_str_list[::-1]):

            # Skip empty strings
            if len(display_str) == 0:
                continue

            text_width, text_height = get_text_size(font,display_str)
            margin = int(np.ceil(0.05 * text_height))

            if text_rotation is not None and text_rotation != 0:

                assert text_rotation == -90, \
                    'Only -90-degree text rotation is supported'

                image_tmp = Image.new('RGB',(text_width+2*margin,text_height+2*margin))
                image_tmp_draw = ImageDraw.Draw(image_tmp)
                image_tmp_draw.rectangle([0,0,text_width+2*margin,text_height+2*margin],fill=color)
                image_tmp_draw.text( (margin,margin), display_str, font=font, fill='black')
                rotated_text = image_tmp.rotate(text_rotation,expand=1)

                if textalign == TEXTALIGN_RIGHT:
                    text_left = right
                else:
                    text_left = left
                text_left = int(text_left + (text_height) * i_str)

                if vtextalign == VTEXTALIGN_BOTTOM:
                    text_top = bottom - text_width
                else:
                    text_top = top
                text_left = int(text_left)
                text_top = int(text_top)

                image.paste(rotated_text,[text_left,text_top])

            else:

                # If the total height of the display strings added to the top of the bounding
                # box exceeds the top of the image, stack the strings below the bounding box
                # instead of above, and vice-versa if we're bottom-aligning.
                #
                # If neither outside-the-box position fits within the image (e.g. the box
                # is nearly the full height of the image), render the label just inside the
                # box, respecting the caller's original alignment preference.
                if vtextalign == VTEXTALIGN_TOP:
                    text_bottom = top
                    if (text_bottom - total_display_str_height) < 0:
                        text_bottom = bottom + total_display_str_height
                        if text_bottom > im_height:
                            text_bottom = top + total_display_str_height
                else:
                    assert vtextalign == VTEXTALIGN_BOTTOM, \
                        'Unrecognized vertical text alignment {}'.format(vtextalign)
                    text_bottom = bottom + total_display_str_height
                    if text_bottom > im_height:
                        text_bottom = top
                        if (text_bottom - total_display_str_height) < 0:
                            text_bottom = bottom

                text_bottom = int(text_bottom) - i_str * (int(text_height + (2 * margin)))

                text_left = left

                if textalign == TEXTALIGN_RIGHT:
                    text_left = right - text_width
                elif textalign == TEXTALIGN_CENTER:
                    text_left = ((right + left) / 2.0) - (text_width / 2.0)
                text_left = int(text_left)

                draw.rectangle(
                    [(text_left, (text_bottom - text_height) - (2 * margin)),
                     (text_left + text_width, text_bottom)],
                    fill=color)

                draw.text(
                    (text_left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill='black',
                    font=font)

            # ...if we're rotating text

    # ...if we're rendering text

# ...def draw_bounding_box_on_image(...)


def render_megadb_bounding_boxes(boxes_info, image):
    """
    Render bounding boxes to an image, where those boxes are in the mostly-deprecated
    MegaDB format, which looks like:

    .. code-block::none

        {
            "category": "animal",
            "bbox": [
                0.739,
                0.448,
                0.187,
                0.198
            ]
        }

    Args:
        boxes_info (list): list of dicts, each dict represents a single detection
            where bbox coordinates are normalized [x_min, y_min, width, height]
        image (PIL.Image.Image): image to modify

    :meta private:
    """

    display_boxes = []
    display_strs = []
    classes = []  # ints, for selecting colors

    for b in boxes_info:
        x_min, y_min, w_rel, h_rel = b['bbox']
        y_max = y_min + h_rel
        x_max = x_min + w_rel
        display_boxes.append([y_min, x_min, y_max, x_max])
        display_strs.append([b['category']])
        classes.append(annotation_constants.detector_bbox_category_name_to_id[b['category']])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)

# ...def render_iMerit_boxes(...)


def render_db_bounding_boxes(boxes,
                             classes,
                             image,
                             original_size=None,
                             label_map=None,
                             thickness=DEFAULT_BOX_THICKNESS,
                             expansion=0,
                             colormap=None,
                             textalign=TEXTALIGN_LEFT,
                             vtextalign=VTEXTALIGN_TOP,
                             text_rotation=None,
                             label_font_size=DEFAULT_LABEL_FONT_SIZE,
                             tags=None,
                             boxes_are_normalized=False):
    """
    Render bounding boxes (with class labels) on an image.  This is a wrapper for
    draw_bounding_boxes_on_image, allowing the caller to operate on a resized image
    by providing the original size of the image; boxes will be scaled accordingly.

    This function assumes that bounding boxes are in absolute coordinates, typically
    because they come from COCO camera traps .json files, unless boxes_are_normalized
    is True.

    Args:
        boxes (list): list of length-4 tuples, foramtted as (x,y,w,h) (in pixels)
        classes (list): list of ints (or string-formatted ints), used to choose labels (either
            by literally rendering the class labels, or by indexing into [label_map])
        image (PIL.Image.Image): image object to modify
        original_size (tuple, optional): if this is not None, and the size is different than
            the size of [image], we assume that [boxes] refer to the original size, and we scale
            them accordingly before rendering
        label_map (dict, optional): int --> str dictionary, typically mapping category IDs to
            species labels; if None, category labels are rendered verbatim (typically as numbers)
        thickness (int or float, optional): line thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        expansion (int or float, optional): number of pixels to expand bounding boxes on each side.  If
            this is a float less than 1.0, it's treated as a fraction of the image width.
        colormap (list, optional): list of color names, used to choose colors for categories by
            indexing with the values in [classes]; defaults to a reasonable set of colors
        textalign (int, optional): TEXTALIGN_LEFT, TEXTALIGN_CENTER, or TEXTALIGN_RIGHT
        vtextalign (int, optional): VTEXTALIGN_TOP or VTEXTALIGN_BOTTOM
        text_rotation (float, optional): rotation to apply to text
        label_font_size (float, optional): font size in pixels.  If this is less than one, it's treated
            as a fraction of the image width.
        tags (list, optional): list of strings of length len(boxes) that should be appended
            after each class name (e.g. to show scores)
        boxes_are_normalized (bool, optional): whether boxes have already been normalized
    """

    display_boxes = []
    display_strs = []

    if original_size is not None:
        image_size = original_size
    else:
        image_size = image.size

    img_width, img_height = image_size

    for i_box in range(0,len(boxes)):

        box = boxes[i_box]
        clss = classes[i_box]

        x_min_abs, y_min_abs, width_abs, height_abs = box[0:4]

        # Normalize boxes if necessary
        if boxes_are_normalized:

            xmin = x_min_abs
            xmax = x_min_abs + width_abs
            ymin = y_min_abs
            ymax = y_min_abs + height_abs

        else:

            ymin = y_min_abs / img_height
            ymax = ymin + height_abs / img_height

            xmin = x_min_abs / img_width
            xmax = xmin + width_abs / img_width

        display_boxes.append([ymin, xmin, ymax, xmax])

        if (label_map is not None) and (int(clss) in label_map):
            clss = label_map[int(clss)]

        display_str = str(clss)

        # Do we have a tag to append to the class string?
        if tags is not None and tags[i_box] is not None and len(tags[i_box]) > 0:
            display_str += ' ' + tags[i_box]

        # need to be a string here because PIL needs to iterate through chars
        display_strs.append([display_str])

    # ...for each box

    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image,
                                 display_boxes,
                                 classes,
                                 display_strs=display_strs,
                                 thickness=thickness,
                                 expansion=expansion,
                                 colormap=colormap,
                                 textalign=textalign,
                                 vtextalign=vtextalign,
                                 text_rotation=text_rotation,
                                 label_font_size=label_font_size)

# ...def render_db_bounding_boxes(...)


def draw_bounding_boxes_on_file(input_file,
                                output_file,
                                detections,
                                confidence_threshold=0.0,
                                detector_label_map=DEFAULT_DETECTOR_LABEL_MAP,
                                thickness=DEFAULT_BOX_THICKNESS,
                                expansion=0,
                                colormap=None,
                                label_font_size=DEFAULT_LABEL_FONT_SIZE,
                                custom_strings=None,
                                target_size=None,
                                ignore_exif_rotation=False,
                                quality=None):
    """
    Renders detection bounding boxes on an image loaded from file, optionally writing the results to
    a new image file.

    Args:
        input_file (str): filename or URL to load
        output_file (str): filename to which we should write the rendered image
        detections (list): a list of dictionaries with keys 'conf', 'bbox', and 'category';
            boxes are length-four arrays formatted as [x,y,w,h], normalized,
            upper-left origin (this is the standard MD detection format). 'category' is a string-int.
        confidence_threshold (float, optional): only render detections with confidence above this
            threshold
        detector_label_map (dict, optional): a dict mapping category IDs to strings.  If this
            is None, no confidence values or identifiers are shown.  If this is {}, just category
            indices and confidence values are shown.
        thickness (int or float, optional): line thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        expansion (int or float, optional): number of pixels to expand bounding boxes on each side.  If
            this is a float less than 1.0, it's treated as a fraction of the image width.
        colormap (list, optional): list of color names, used to choose colors for categories by
            indexing with the values in [classes]; defaults to a reasonable set of colors
        label_font_size (float, optional): font size in pixels.  If this is less than one, it's treated
            as a fraction of the image width.
        custom_strings (list, optional): set of strings to append to detection labels, should have the
            same length as [detections].  Appended before any classification labels.
        target_size (tuple, optional): tuple of (target_width,target_height).  Either or both can be -1,
            see resize_image() for documentation.  If None or (-1,-1), uses the original image size.
        ignore_exif_rotation (bool, optional): don't rotate the loaded pixels,
            even if we are loading a JPEG and that JPEG says it should be rotated.
        quality (int, optional): jpeg quality to use for output (None to use PIL default)

    Returns:
        PIL.Image.Image: loaded and modified image
    """

    image = open_image(input_file, ignore_exif_rotation=ignore_exif_rotation)

    if target_size is not None:
        image = resize_image(image,target_size[0],target_size[1])

    render_detection_bounding_boxes(
            detections,
            image,
            label_map=detector_label_map,
            confidence_threshold=confidence_threshold,
            thickness=thickness,
            expansion=expansion,
            colormap=colormap,
            custom_strings=custom_strings,
            label_font_size=label_font_size)

    if output_file is not None:
        if quality is None:
            image.save(output_file)
        else:
            image.save(output_file,quality=quality)

    return image

# ...def draw_bounding_boxes_on_file(...)


def draw_db_boxes_on_file(input_file,
                          output_file,
                          boxes,
                          classes=None,
                          label_map=None,
                          thickness=DEFAULT_BOX_THICKNESS,
                          expansion=0,
                          ignore_exif_rotation=False,
                          quality=None):
    """
    Render COCO-formatted bounding boxes (in absolute coordinates) on an image loaded from file,
    writing the results to a new image file.

    Args:
        input_file (str): image file to read
        output_file (str): image file to write
        boxes (list): list of length-4 tuples, foramtted as (x,y,w,h) (in pixels)
        classes (list, optional): list of ints (or string-formatted ints), used to choose
            labels (either by literally rendering the class labels, or by indexing into [label_map])
        label_map (dict, optional): int --> str dictionary, typically mapping category IDs to
            species labels; if None, category labels are rendered verbatim (typically as numbers)
        thickness (int or float, optional): line thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        expansion (int or float, optional): number of pixels to expand bounding boxes on each side.  If
            this is a float less than 1.0, it's treated as a fraction of the image width.
        ignore_exif_rotation (bool, optional): don't rotate the loaded pixels,
            even if we are loading a JPEG and that JPEG says it should be rotated
        quality (int, optional): jpeg quality to use for output (None to use PIL default)

    Returns:
        PIL.Image.Image: the loaded and modified image
    """

    image = open_image(input_file, ignore_exif_rotation=ignore_exif_rotation)

    if classes is None:
        classes = [0] * len(boxes)

    render_db_bounding_boxes(boxes,
                             classes,
                             image,
                             original_size=None,
                             label_map=label_map,
                             thickness=thickness,
                             expansion=expansion)

    if quality is None:
        image.save(output_file)
    else:
        image.save(output_file,quality=quality)

    return image

# ...def draw_bounding_boxes_on_file(...)


def gray_scale_fraction(image,crop_size=(0.1,0.1)):
    """
    Computes the fraction of the pixels in [image] that appear to be grayscale (R==G==B),
    useful for approximating whether this is a night-time image when flash information is not
    available in EXIF data (or for video frames, where this information is often not available
    in structured metadata at all).

    Args:
        image (str or PIL.Image.Image): Image, filename, or URL to analyze
        crop_size (tuple of floats, optional): a 2-element list/tuple, representing the fraction of
            the image to crop at the top and bottom, respectively, before analyzing (to minimize
            the possibility of including color elements in the image overlay)

    Returns:
        float: the fraction of pixels in [image] that appear to be grayscale (R==G==B)
    """

    if isinstance(image,str):
        image = Image.open(image)

    if image.mode == 'L':
        return 1.0

    if len(image.getbands()) == 1:
        return 1.0

    # Crop if necessary
    if crop_size[0] > 0 or crop_size[1] > 0:

        assert (crop_size[0] + crop_size[1]) < 1.0, \
            'Illegal crop size: {}'.format(str(crop_size))

        top_crop_pixels = int(image.height * crop_size[0])
        bottom_crop_pixels = int(image.height * crop_size[1])

        left = 0
        right = image.width

        # Remove pixels from the top
        first_crop_top = top_crop_pixels
        first_crop_bottom = image.height
        first_crop = image.crop((left, first_crop_top, right, first_crop_bottom))

        # Remove pixels from the bottom
        second_crop_top = 0
        second_crop_bottom = first_crop.height - bottom_crop_pixels
        second_crop = first_crop.crop((left, second_crop_top, right, second_crop_bottom))

        image = second_crop

    # It doesn't matter if these are actually R/G/B, they're just names
    r = np.array(image.getchannel(0))
    g = np.array(image.getchannel(1))
    b = np.array(image.getchannel(2))

    gray_pixels = np.logical_and(r == g, r == b)
    n_pixels = gray_pixels.size
    n_gray_pixels = gray_pixels.sum()

    return n_gray_pixels / n_pixels

    # Non-numpy way to do the same thing, briefly keeping this here for posterity
    if False:

        w, h = image.size
        n_pixels = w*h
        n_gray_pixels = 0
        for i in range(w):
            for j in range(h):
                r, g, b = image.getpixel((i,j))
                if r == g and r == b and g == b:
                    n_gray_pixels += 1

# ...def gray_scale_fraction(...)


def _resize_relative_image(fn_relative,
                           input_folder,
                           output_folder,
                           target_width,
                           target_height,
                           no_enlarge_width,
                           verbose,
                           quality,
                           overwrite=True):
    """
    Internal function for resizing an image from one folder to another,
    maintaining relative path.
    """

    input_fn_abs = os.path.join(input_folder,fn_relative)
    output_fn_abs = os.path.join(output_folder,fn_relative)

    if (not overwrite) and (os.path.isfile(output_fn_abs)):
        status = 'skipped'
        error = None
        return {'fn_relative':fn_relative,'status':status,'error':error}

    os.makedirs(os.path.dirname(output_fn_abs),exist_ok=True)
    try:
        _ = resize_image(input_fn_abs,
                         output_file=output_fn_abs,
                         target_width=target_width,
                         target_height=target_height,
                         no_enlarge_width=no_enlarge_width,
                         verbose=verbose,
                         quality=quality)
        status = 'success'
        error = None
    except Exception as e:
        if verbose:
            print('Error resizing {}: {}'.format(fn_relative,str(e)))
        status = 'error'
        error = str(e)

    return {'fn_relative':fn_relative,'status':status,'error':error}

# ...def _resize_relative_image(...)


def _resize_absolute_image(input_output_files,
                           target_width,
                           target_height,
                           no_enlarge_width,
                           verbose,
                           quality):
    """
    Internal wrapper for resize_image used in the context of a batch resize operation.
    """

    input_fn_abs = input_output_files[0]
    output_fn_abs = input_output_files[1]
    os.makedirs(os.path.dirname(output_fn_abs),exist_ok=True)
    try:
        _ = resize_image(input_fn_abs,
                         output_file=output_fn_abs,
                         target_width=target_width,
                         target_height=target_height,
                         no_enlarge_width=no_enlarge_width,
                         verbose=verbose,
                         quality=quality)
        status = 'success'
        error = None
    except Exception as e:
        if verbose:
            print('Error resizing {}: {}'.format(input_fn_abs,str(e)))
        status = 'error'
        error = str(e)

    return {'input_fn':input_fn_abs,
            'output_fn':output_fn_abs,
            'status':status,
            'error':error}

# ..._resize_absolute_image(...)


def resize_images(input_file_to_output_file,
                  target_width=-1,
                  target_height=-1,
                  no_enlarge_width=False,
                  verbose=False,
                  quality='keep',
                  pool_type='process',
                  n_workers=10):
    """
    Resizes all images the dictionary [input_file_to_output_file].

    TODO: This is a little more redundant with resize_image_folder than I would like;
    refactor resize_image_folder to call resize_images.  Not doing that yet because
    at the time I'm writing this comment, a lot of code depends on resize_image_folder
    and I don't want to rock the boat yet.

    Args:
        input_file_to_output_file (dict): dict mapping images that exist to the locations
            where the resized versions should be written
        target_width (int, optional): width to which we should resize this image, or -1
            to let target_height determine the size
        target_height (int, optional): height to which we should resize this image, or -1
            to let target_width determine the size
        no_enlarge_width (bool, optional): if [no_enlarge_width] is True, and
            [target width] is larger than the original image width, does not modify the image,
            but will write to output_file if supplied
        verbose (bool, optional): enable additional debug output
        quality (str or int, optional): passed to exif_preserving_save, see docs for more detail
        pool_type (str, optional): whether use use processes ('process') or threads ('thread') for
            parallelization; ignored if n_workers <= 1
        n_workers (int, optional): number of workers to use for parallel resizing; set to <=1
            to disable parallelization

    Returns:
        list: a list of dicts with keys 'input_fn', 'output_fn', 'status', and 'error'.
        'status' will be 'success' or 'error'; 'error' will be None for successful cases,
        otherwise will contain the image-specific error.
    """

    assert pool_type in ('process','thread'), 'Illegal pool type {}'.format(pool_type)

    input_output_file_pairs = []

    # Reformat input files as (input,output) tuples
    for input_fn in input_file_to_output_file:
        input_output_file_pairs.append((input_fn,input_file_to_output_file[input_fn]))

    if n_workers == 1:

        results = []
        for i_o_file_pair in tqdm(input_output_file_pairs):
            results.append(_resize_absolute_image(i_o_file_pair,
                            target_width=target_width,
                            target_height=target_height,
                            no_enlarge_width=no_enlarge_width,
                            verbose=verbose,
                            quality=quality))

    else:

        pool = None

        try:

            if pool_type == 'thread':
                pool = ThreadPool(n_workers); poolstring = 'threads'
            else:
                assert pool_type == 'process'
                pool = Pool(n_workers); poolstring = 'processes'

            if verbose:
                print('Starting resizing pool with {} {}'.format(n_workers,poolstring))

            p = partial(_resize_absolute_image,
                    target_width=target_width,
                    target_height=target_height,
                    no_enlarge_width=no_enlarge_width,
                    verbose=verbose,
                    quality=quality)

            results = list(tqdm(pool.imap(p, input_output_file_pairs),total=len(input_output_file_pairs)))

        finally:

            if pool is not None:
                pool.close()
                pool.join()
                print('Pool closed and joined for image resizing')

    return results

# ...def resize_images(...)


def resize_image_folder(input_folder,
                        output_folder=None,
                        target_width=-1,
                        target_height=-1,
                        no_enlarge_width=False,
                        verbose=False,
                        quality='keep',
                        pool_type='process',
                        n_workers=10,
                        recursive=True,
                        image_files_relative=None,
                        overwrite=True):
    """
    Resize all images in a folder (defaults to recursive).

    Defaults to in-place resizing (output_folder is optional).

    Args:
        input_folder (str): folder in which we should find images to resize
        output_folder (str, optional): folder in which we should write resized images.  If
            None, resizes images in place.  Otherwise, maintains relative paths in the target
            folder.
        target_width (int, optional): width to which we should resize this image, or -1
            to let target_height determine the size
        target_height (int, optional): height to which we should resize this image, or -1
            to let target_width determine the size
        no_enlarge_width (bool, optional): if [no_enlarge_width] is True, and
            [target width] is larger than the original image width, does not modify the image,
            but will write to output_file if supplied
        verbose (bool, optional): enable additional debug output
        quality (str or int, optional): passed to exif_preserving_save, see docs for more detail
        pool_type (str, optional): whether use use processes ('process') or threads ('thread') for
            parallelization; ignored if n_workers <= 1
        n_workers (int, optional): number of workers to use for parallel resizing; set to <=1
            to disable parallelization
        recursive (bool, optional): whether to search [input_folder] recursively for images.
        image_files_relative (list, optional): if not None, skips any relative paths not
            in this list
        overwrite (bool, optional): whether to overwrite existing target images

    Returns:
        list: a list of dicts with keys 'input_fn', 'output_fn', 'status', and 'error'.
        'status' will be 'success', 'skipped', or 'error'; 'error' will be None for successful
        cases, otherwise will contain the image-specific error.
    """

    assert os.path.isdir(input_folder), '{} is not a folder'.format(input_folder)

    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder,exist_ok=True)

    assert pool_type in ('process','thread'), 'Illegal pool type {}'.format(pool_type)

    if image_files_relative is None:

        if verbose:
            print('Enumerating images')

        image_files_relative = find_images(input_folder,recursive=recursive,
                                           return_relative_paths=True,convert_slashes=True)
        if verbose:
            print('Found {} images'.format(len(image_files_relative)))

    if n_workers == 1:

        if verbose:
            print('Resizing images')

        results = []
        for fn_relative in tqdm(image_files_relative):
            results.append(_resize_relative_image(fn_relative,
                                  input_folder=input_folder,
                                  output_folder=output_folder,
                                  target_width=target_width,
                                  target_height=target_height,
                                  no_enlarge_width=no_enlarge_width,
                                  verbose=verbose,
                                  quality=quality,
                                  overwrite=overwrite))

    else:

        if pool_type == 'thread':
            pool = ThreadPool(n_workers); poolstring = 'threads'
        else:
            assert pool_type == 'process'
            pool = Pool(n_workers); poolstring = 'processes'

        if verbose:
            print('Starting resizing pool with {} {}'.format(n_workers,poolstring))

        p = partial(_resize_relative_image,
                input_folder=input_folder,
                output_folder=output_folder,
                target_width=target_width,
                target_height=target_height,
                no_enlarge_width=no_enlarge_width,
                verbose=verbose,
                quality=quality,
                overwrite=overwrite)

        results = list(tqdm(pool.imap(p, image_files_relative),
                            total=len(image_files_relative)))

    return results

# ...def resize_image_folder(...)


def get_image_size(im,verbose=False):
    """
    Retrieve the size of an image.  Returns None if the image fails to load.

    Args:
        im (str or PIL.Image): filename or PIL image
        verbose (bool, optional): enable additional debug output

    Returns:
        tuple (w,h), or None if the image fails to load.
    """

    image_name = '[in memory]'

    try:
        if isinstance(im,str):
            image_name = im
            im = load_image(im)
        w = im.width
        h = im.height
        if w <= 0 or h <= 0:
            if verbose:
                print('Error reading width from image {}: {},{}'.format(
                    image_name,w,h))
            return None
        return (w,h)
    except Exception as e:
        if verbose:
            print('Error reading width from image {}: {}'.format(
                image_name,str(e)))
        return None

# ...def get_image_size(...)


def parallel_get_image_sizes(filenames,
                             max_workers=16,
                             use_threads=True,
                             recursive=True,
                             verbose=False):
    """
    Retrieve image sizes for a list or folder of images

    Args:
        filenames (list or str): a list of image filenames or a folder.  Non-image files and
            unreadable images will be returned with a file size of None.
        max_workers (int, optional): the number of parallel workers to use; set to <=1 to disable
            parallelization
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallelization
        recursive (bool, optional): if [filenames] is a folder, whether to search recursively
            for images. Ignored if [filenames] is a list.
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: a dict mapping filenames to (w,h) tuples; the value will be None for images that fail
        to load.  Filenames will always be absolute.
    """

    if isinstance(filenames,str) and os.path.isdir(filenames):
        if verbose:
            print('Enumerating images in {}'.format(filenames))
        filenames = find_images(filenames,recursive=recursive,return_relative_paths=False)

    n_workers = min(max_workers,len(filenames))

    if verbose:
        print('Getting image sizes for {} images'.format(len(filenames)))

    if n_workers <= 1:

        results = []
        for filename in filenames:
            results.append(get_image_size(filename,verbose=verbose))

    else:

        if use_threads:
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)

        try:
            results = list(tqdm(pool.imap(
                partial(get_image_size,verbose=verbose),filenames), total=len(filenames)))
        finally:
            pool.close()
            pool.join()
            print('Pool closed and joined for image size retrieval')

    assert len(filenames) == len(results), 'Internal error in parallel_get_image_sizes'

    to_return = {}
    for i_file,filename in enumerate(filenames):
        to_return[filename] = results[i_file]

    return to_return


#%% Image integrity checking functions

def check_image_integrity(filename,modes=None):
    """
    Check whether we can successfully load an image via OpenCV and/or PIL.

    Args:
        filename (str): the filename to evaluate
        modes (list, optional): a list containing one or more of:

            - 'cv'
            - 'pil'
            - 'skimage'
            - 'jpeg_trailer'

            'jpeg_trailer' checks that the binary data ends with ffd9.  It does not check whether
            the image is actually a jpeg, and even if it is, there are lots of reasons the image might not
            end with ffd9.  It's also true the JPEGs that cause "premature end of jpeg segment" issues
            don't end with ffd9, so this may be a useful diagnostic.  High precision, very low recall
            for corrupt jpegs.

            Set to None to use all modes.

    Returns:
        dict: a dict with a key called 'file' (the value of [filename]), one key for each string in
        [modes] (a success indicator for that mode, specifically a string starting with either
        'success' or 'error').
    """

    if modes is None:
        modes = ('cv','pil','skimage','jpeg_trailer')
    else:
        if isinstance(modes,str):
            modes = [modes]
        for mode in modes:
            assert mode in ('cv','pil','skimage'), 'Unrecognized mode {}'.format(mode)

    assert os.path.isfile(filename), 'Could not find file {}'.format(filename)

    result = {}
    result['file'] = filename

    for mode in modes:

        result[mode] = 'unknown'
        if mode == 'pil':
            try:
                pil_im = load_image(filename) # noqa
                assert pil_im is not None
                result[mode] = 'success'
            except Exception as e:
                result[mode] = 'error: {}'.format(str(e))
        elif mode == 'cv':
            try:
                cv_im = cv2.imread(filename)
                assert cv_im is not None, 'Unknown opencv read failure'
                numpy_im = np.asarray(cv_im) # noqa
                result[mode] = 'success'
            except Exception as e:
                result[mode] = 'error: {}'.format(str(e))
        elif mode == 'skimage':
            try:
                # This is not a standard dependency
                from skimage import io as skimage_io # type: ignore # noqa
            except Exception:
                result[mode] = 'could not import skimage, run pip install scikit-image'
                return result
            try:
                skimage_im = skimage_io.imread(filename) # noqa
                assert skimage_im is not None
                result[mode] = 'success'
            except Exception as e:
                result[mode] = 'error: {}'.format(str(e))
        elif mode == 'jpeg_trailer':
            # https://stackoverflow.com/a/48282863/16644970
            try:
                with open(filename, 'rb') as f:
                    check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    result[mode] = 'invalid jpeg trailer: {}'.format(str(check_chars))
                else:
                    result[mode] = 'success'
            except Exception as e:
                result[mode] = 'error: {}'.format(str(e))

    # ...for each mode

    return result

# ...def check_image_integrity(...)


def parallel_check_image_integrity(filenames,
                                   modes=None,
                                   max_workers=16,
                                   use_threads=True,
                                   recursive=True,
                                   verbose=False):
    """
    Check whether we can successfully load a list of images via OpenCV and/or PIL.

    Args:
        filenames (list or str): a list of image filenames or a folder
        modes (list, optional): see check_image_integrity() for documentation on the [modes] parameter
        max_workers (int, optional): the number of parallel workers to use; set to <=1 to disable
            parallelization
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallelization
        recursive (bool, optional): if [filenames] is a folder, whether to search recursively for images.
            Ignored if [filenames] is a list.
        verbose (bool, optional): enable additional debug output

    Returns:
        list: a list of dicts, each with a key called 'file' (the value of [filename]), one key for
        each string in [modes] (a success indicator for that mode, specifically a string starting
        with either 'success' or 'error').
    """

    if isinstance(filenames,str) and os.path.isdir(filenames):
        if verbose:
            print('Enumerating images in {}'.format(filenames))
        filenames = find_images(filenames,recursive=recursive,return_relative_paths=False)

    n_workers = min(max_workers,len(filenames))

    if verbose:
        print('Checking image integrity for {} filenames'.format(len(filenames)))

    if n_workers <= 1:

        results = []
        for filename in filenames:
            results.append(check_image_integrity(filename,modes=modes))

    else:

        if use_threads:
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)

        results = list(tqdm(pool.imap(
            partial(check_image_integrity,modes=modes),filenames), total=len(filenames)))

    return results


#%% Test drivers

if False:

    #%% Text rendering tests

    import os # noqa
    import numpy as np # noqa
    from megadetector.visualization.visualization_utils import \
        draw_bounding_boxes_on_image, exif_preserving_save, load_image, \
        TEXTALIGN_LEFT,TEXTALIGN_RIGHT,VTEXTALIGN_BOTTOM,VTEXTALIGN_TOP, \
        DEFAULT_LABEL_FONT_SIZE

    fn = os.path.expanduser('~/AppData/Local/Temp/md-tests/md-test-images/ena24_7904.jpg')
    output_fn = r'g:\temp\test.jpg'

    image = load_image(fn)

    w = 0.2; h = 0.2
    all_boxes = [[0.05, 0.05, 0.25, 0.25],
                 [0.05, 0.35, 0.25, 0.6],
                 [0.35, 0.05, 0.6, 0.25],
                 [0.35, 0.35, 0.6, 0.6]]

    alignments = [
        [TEXTALIGN_LEFT,VTEXTALIGN_TOP],
        [TEXTALIGN_LEFT,VTEXTALIGN_BOTTOM],
        [TEXTALIGN_RIGHT,VTEXTALIGN_TOP],
        [TEXTALIGN_RIGHT,VTEXTALIGN_BOTTOM]
        ]

    labels = ['left_top','left_bottom','right_top','right_bottom']

    text_rotation = -90
    n_label_copies = 2

    for i_box,box in enumerate(all_boxes):

        boxes = [box]
        boxes = np.array(boxes)
        classes = [i_box]
        display_strs = [[labels[i_box]]*n_label_copies]
        textalign = alignments[i_box][0]
        vtextalign = alignments[i_box][1]
        draw_bounding_boxes_on_image(image,
                                     boxes,
                                     classes,
                                     thickness=2,
                                     expansion=0,
                                     display_strs=display_strs,
                                     colormap=None,
                                     textalign=textalign,
                                     vtextalign=vtextalign,
                                     label_font_size=DEFAULT_LABEL_FONT_SIZE,
                                     text_rotation=text_rotation)

    exif_preserving_save(image,output_fn)
    from megadetector.utils.path_utils import open_file
    open_file(output_fn)


    #%% Recursive resize test

    from megadetector.visualization.visualization_utils import resize_image_folder # noqa

    input_folder = r"C:\temp\resize-test\in"
    output_folder = r"C:\temp\resize-test\out"

    resize_results = resize_image_folder(input_folder,output_folder,
                         target_width=1280,verbose=True,quality=85,no_enlarge_width=True,
                         pool_type='process',n_workers=10)


    #%% Integrity checking test

    from megadetector.utils import md_tests
    options = md_tests.download_test_data()
    folder = options.scratch_dir

    results = parallel_check_image_integrity(folder,max_workers=8)

    modes = ['cv','pil','skimage','jpeg_trailer']

    for r in results:
        for mode in modes:
            if r[mode] != 'success':
                s = r[mode]
                print('Mode {} failed for {}:\n{}\n'.format(mode,r['file'],s))
