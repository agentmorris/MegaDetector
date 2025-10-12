"""

pytorch_detector.py

Module to run YOLO-based MegaDetector models.

"""

#%% Imports and constants

import os
import sys
import math
import zipfile
import tempfile
import shutil
import uuid
import json
import inspect

import cv2
import torch
import numpy as np

from megadetector.detection.run_detector import \
    CONF_DIGITS, COORD_DIGITS, FAILURE_INFER, FAILURE_IMAGE_OPEN, \
    get_detector_version_from_model_file, \
    known_models
from megadetector.utils.ct_utils import parse_bool_string
from megadetector.utils.ct_utils import is_running_in_gha
from megadetector.utils import ct_utils
import torchvision

# We support a few ways of accessing the YOLOv5 dependencies:
#
# * The standard configuration as of 2023.09 expects that the YOLOv5 repo is checked
#   out and on the PYTHONPATH (import utils)
#
# * Supported but non-default (used for PyPI packaging):
#
#   pip install ultralytics-yolov5
#
# * Works, but not supported:
#
#   pip install yolov5
#
# * Unfinished:
#
#   pip install ultralytics

yolo_model_type_imported = None

def _get_model_type_for_model(model_file,
                              prefer_model_type_source='table',
                              default_model_type='yolov5',
                              verbose=False):
    """
    Determine the model type (i.e., the inference library we need to use) for a .pt file.

    Args:
        model_file (str): the model file to read
        prefer_model_type_source (str, optional): how should we handle the (very unlikely)
            case where the metadata in the file indicates one model type, but the global model
            type table says something else.  Should be "table" (trust the table) or "file"
            (trust the file).
        default_model_type (str, optional): return value for the case where we can't find
            appropriate metadata in the file or in the global table.
        verbose (bool, optional): enable additional debug output

    Returns:
        str: the model type indicated for this model
    """

    model_info = read_metadata_from_megadetector_model_file(model_file)

    # Check whether the model file itself specified a model type
    model_type_from_model_file_metadata = None

    if model_info is not None and 'model_type' in model_info:
        model_type_from_model_file_metadata = model_info['model_type']
        if verbose:
            print('Parsed model type {} from model {}'.format(
                model_type_from_model_file_metadata,
                model_file))

    model_type_from_model_version = None

    # Check whether this is a known model version with a specific model type
    model_version_from_file = get_detector_version_from_model_file(model_file)

    if model_version_from_file is not None and model_version_from_file in known_models:
        model_info = known_models[model_version_from_file]
        if 'model_type' in model_info:
            model_type_from_model_version = model_info['model_type']
            if verbose:
                print('Parsed model type {} from global metadata'.format(model_type_from_model_version))
        else:
            model_type_from_model_version = None

    if model_type_from_model_file_metadata is None and \
        model_type_from_model_version is None:
        if verbose:
            print('Could not determine model type for {}, assuming {}'.format(
                model_file,default_model_type))
        model_type = default_model_type

    elif model_type_from_model_file_metadata is not None and \
         model_type_from_model_version is not None:
        if model_type_from_model_version == model_type_from_model_file_metadata:
            model_type = model_type_from_model_file_metadata
        else:
            print('Warning: model type from model version is {}, from file metadata is {}'.format(
                model_type_from_model_version,model_type_from_model_file_metadata))
            if prefer_model_type_source == 'table':
                model_type = model_type_from_model_version
            else:
                model_type = model_type_from_model_file_metadata

    elif model_type_from_model_file_metadata is not None:

        model_type = model_type_from_model_file_metadata

    elif model_type_from_model_version is not None:

        model_type = model_type_from_model_version

    return model_type

# ...def _get_model_type_for_model(...)


def _initialize_yolo_imports_for_model(model_file,
                                       prefer_model_type_source='table',
                                       default_model_type='yolov5',
                                       detector_options=None,
                                       verbose=False):
    """
    Initialize the appropriate YOLO imports for a model file.

    Args:
        model_file (str): The model file for which we're loading support
        prefer_model_type_source (str, optional): how should we handle the (very unlikely)
            case where the metadata in the file indicates one model type, but the global model
            type table says something else.  Should be "table" (trust the table) or "file"
            (trust the file).
        default_model_type (str, optional): return value for the case where we can't find
            appropriate metadata in the file or in the global table.
        detector_options (dict, optional): dictionary of detector options that mean
            different things to different models
        verbose (bool, optional): enable additional debug output

    Returns:
        str: the model type for which we initialized support
    """

    global yolo_model_type_imported

    if detector_options is not None and 'model_type' in detector_options:
        model_type = detector_options['model_type']
        print('Model type {} provided in detector options'.format(model_type))
    else:
        model_type = _get_model_type_for_model(model_file,
                                               prefer_model_type_source=prefer_model_type_source,
                                               default_model_type=default_model_type)

    if yolo_model_type_imported is not None:
        if model_type == yolo_model_type_imported:
            print('Bypassing imports for model type {}'.format(model_type))
            return
        else:
            print('Previously set up imports for model type {}, re-importing as {}'.format(
                yolo_model_type_imported,model_type))

    _initialize_yolo_imports(model_type,verbose=verbose)

    return model_type


def _clean_yolo_imports(verbose=False, aggressive_cleanup=False):
    """
    Remove all YOLO-related imports from sys.modules and sys.path, to allow a clean re-import
    of another YOLO library version.  The reason we jump through all these hoops, rather than
    just, e.g., handling different libraries in different modules, is that we need to make sure
    *pickle* sees the right version of modules during module loading, including modules we don't
    load directly (i.e., every module loaded within a YOLO library), and the only way I know to
    do that is to remove all the "wrong" versions from sys.modules and sys.path.

    Args:
        verbose (bool, optional): enable additional debug output
        aggressive_cleanup (bool, optional): err on the side of removing modules,
            at least by ignoring whether they are/aren't in a site-packages folder.
            By default, only modules in a folder that includes "site-packages" will
            be considered for unloading.
    """

    modules_to_delete = []

    for module_name in sys.modules.keys():

        module = sys.modules[module_name]
        if not hasattr(module,'__file__') or (module.__file__ is None):
            continue
        try:
            module_file = module.__file__.replace('\\','/')
            if not aggressive_cleanup:
                if 'site-packages' not in module_file:
                    continue
            tokens = module_file.split('/')

            # For local path imports, a module filename that should be unloaded might
            # look like:
            #
            # c:/git/yolov9/models/common.py
            #
            # For pip imports, a module filename that should be unloaded might look like:
            #
            # c:/users/user/miniforge3/envs/megadetector/lib/site-packages/yolov9/utils/__init__.py
            first_token_to_check = len(tokens) - 4
            for i_token,token in enumerate(tokens):
                if i_token < first_token_to_check:
                    continue
                # Don't remove anything based on the environment name, which
                # always follows "envs" in the path
                if (i_token > 1) and (tokens[i_token-1] == 'envs'):
                    continue
                if ('yolov5' in token) or ('yolov9' in token) or ('ultralytics' in token):
                    if verbose:
                        print('Module {} ({}) looks deletable'.format(module_name,module_file))
                    modules_to_delete.append(module_name)
                    break
        except Exception as e:
            if verbose:
                print('Exception during module review: {}'.format(str(e)))
            pass

    # ...for each module in the global namespace

    for module_name in modules_to_delete:

        if module_name in sys.modules.keys():
            if verbose:
                try:
                    module = sys.modules[module_name]
                    module_file = module.__file__.replace('\\','/')
                    print('clean_yolo_imports: deleting module {}: {}'.format(module_name,module_file))
                except Exception:
                    pass
            del sys.modules[module_name]

    # ...for each module we want to remove from the global namespace

    paths_to_delete = []

    for p in sys.path:
        if p.endswith('yolov5') or p.endswith('yolov9') or p.endswith('ultralytics'):
            print('clean_yolo_imports: removing {} from path'.format(p))
            paths_to_delete.append(p)

    for p in paths_to_delete:
        sys.path.remove(p)

# ...def _clean_yolo_imports(...)


def _initialize_yolo_imports(model_type='yolov5',
                             allow_fallback_import=True,
                             force_reimport=False,
                             verbose=False):
    """
    Imports required functions from one or more yolo libraries (yolov5, yolov9,
    ultralytics, targeting support for [model_type]).

    Args:
        model_type (str): The model type for which we're loading support
        allow_fallback_import (bool, optional): If we can't import from the package for
            which we're trying to load support, fall back to "import utils".  This is
            typically used when the right support library is on the current PYTHONPATH.
        force_reimport (bool, optional): import the appropriate libraries even if the
            requested model type matches the current initialization state
        verbose (bool, optional): include additional debug output

    Returns:
        str: the model type for which we initialized support
    """

    # When running in pytest, the megadetector 'utils' module is put in the global
    # namespace, which creates conflicts with yolov5; remove it from the global
    # namespsace.
    if ('PYTEST_CURRENT_TEST' in os.environ):
        print('*** pytest detected ***')
        if ('utils' in sys.modules):
            utils_module = sys.modules['utils']
            if hasattr(utils_module, '__file__') and 'megadetector' in str(utils_module.__file__):
                print(f"Removing conflicting utils module: {utils_module.__file__}")
                sys.modules.pop('utils', None)
                # Also remove any submodules
                to_remove = [name for name in sys.modules if name.startswith('utils.')]
                for name in to_remove:
                    sys.modules.pop(name, None)

    global yolo_model_type_imported

    if model_type is None:
        model_type = 'yolov5'

    # The point of this function is to make the appropriate version
    # of the following functions available at module scope
    global non_max_suppression
    global xyxy2xywh
    global letterbox
    global scale_coords

    if yolo_model_type_imported is not None:
        if (yolo_model_type_imported == model_type) and (not force_reimport):
            print('Bypassing imports for YOLO model type {}'.format(model_type))
            return
        else:
            _clean_yolo_imports()

    try_yolov5_import = (model_type == 'yolov5')
    try_yolov9_import = (model_type == 'yolov9')
    try_ultralytics_import = (model_type == 'ultralytics')

    utils_imported = False

    # First try importing from the yolov5 package; this is how the pip
    # package finds YOLOv5 utilities.
    if try_yolov5_import and not utils_imported:

        try:
            # from yolov5.utils.general import non_max_suppression # type: ignore
            from yolov5.utils.general import xyxy2xywh # noqa
            from yolov5.utils.augmentations import letterbox # noqa
            try:
                from yolov5.utils.general import scale_boxes as scale_coords
            except Exception:
                from yolov5.utils.general import scale_coords
            utils_imported = True
            if verbose:
                print('Imported utils from YOLOv5 package')

        except Exception as e: # noqa
            # print('yolov5 module import failed: {}'.format(e))
            # print(traceback.format_exc())
            pass

    # Next try importing from the yolov9 package
    if try_yolov9_import and not utils_imported:

        try:

            # from yolov9.utils.general import non_max_suppression # noqa
            from yolov9.utils.general import xyxy2xywh # noqa
            from yolov9.utils.augmentations import letterbox # noqa
            from yolov9.utils.general import scale_boxes as scale_coords # noqa
            utils_imported = True
            if verbose:
                print('Imported utils from YOLOv9 package')

        except Exception as e: # noqa

            # print('yolov9 module import failed: {}'.format(e))
            # print(traceback.format_exc())
            pass

    # If we haven't succeeded yet, import from the ultralytics package
    if try_ultralytics_import and not utils_imported:

        try:

            import ultralytics # type: ignore # noqa

        except Exception:

            print('It looks like you are trying to run a model that requires the ultralytics package, '
                  'but the ultralytics package is not installed.  For licensing reasons, this '
                  'is not installed by default with the MegaDetector Python package.  Run '
                  '"pip install ultralytics" to install it, and try again.')
            raise

        try:

            # The non_max_suppression() function moved from the ops module to the nms module
            # in mid-2025
            try:
                from ultralytics.utils.ops import non_max_suppression # type: ignore # noqa
            except Exception:
                from ultralytics.utils.nms import non_max_suppression # type: ignore # noqa
            from ultralytics.utils.ops import xyxy2xywh # type: ignore # noqa

            # In the ultralytics package, scale_boxes and scale_coords both exist;
            # we want scale_boxes.
            #
            # from ultralytics.utils.ops import scale_coords # noqa
            from ultralytics.utils.ops import scale_boxes as scale_coords # type: ignore # noqa
            from ultralytics.data.augment import LetterBox # type: ignore # noqa

            # letterbox() became a LetterBox class in the ultralytics package.  Create a
            # backwards-compatible letterbox function wrapper that wraps the class up.
            def letterbox(img,new_shape,auto=False,scaleFill=False, #noqa
                          scaleup=True,center=True,stride=32):

                # Ultralytics changed the "scaleFill" parameter to "scale_fill", we want to support
                # both conventions.
                use_old_scalefill_arg = False
                try:
                    sig = inspect.signature(LetterBox.__init__)
                    if 'scaleFill' in sig.parameters:
                        use_old_scalefill_arg = True
                except Exception:
                    pass

                if use_old_scalefill_arg:
                    if verbose:
                        print('Using old scaleFill calling convention')
                    letterbox_transformer = LetterBox(new_shape,auto=auto,scaleFill=scaleFill,
                                                    scaleup=scaleup,center=center,stride=stride)
                else:
                    letterbox_transformer = LetterBox(new_shape,auto=auto,scale_fill=scaleFill,
                                                  scaleup=scaleup,center=center,stride=stride)

                letterbox_result = letterbox_transformer(image=img)

                if isinstance(new_shape,int):
                    new_shape = [new_shape,new_shape]

                # The letterboxing is done, we just need to reverse-engineer what it did
                shape = img.shape[:2]

                r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
                if not scaleup:
                    r = min(r, 1.0)
                ratio = r, r

                new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
                dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
                if auto:
                    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
                elif scaleFill:
                    dw, dh = 0.0, 0.0
                    new_unpad = (new_shape[1], new_shape[0])
                    ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

                dw /= 2
                dh /= 2
                pad = (dw,dh)

                return [letterbox_result,ratio,pad]

            utils_imported = True
            if verbose:
                print('Imported utils from ultralytics package')

        except Exception as e:

            print('Ultralytics module import failed: {}'.format(str(e)))
            pass

    # If we haven't succeeded yet, assume the YOLOv5 repo is on our PYTHONPATH.
    if (not utils_imported) and allow_fallback_import:

        try:

            # import pre- and post-processing functions from the YOLOv5 repo
            # from utils.general import non_max_suppression # type: ignore
            from utils.general import xyxy2xywh # type: ignore
            from utils.augmentations import letterbox # type: ignore

            # scale_coords() is scale_boxes() in some YOLOv5 versions
            try:
                from utils.general import scale_coords # type: ignore
            except ImportError:
                from utils.general import scale_boxes as scale_coords # type: ignore
            utils_imported = True
            imported_file = sys.modules[scale_coords.__module__].__file__
            if verbose:
                print('Imported utils from {}'.format(imported_file))

        except ModuleNotFoundError as e:

            raise ModuleNotFoundError('Could not import YOLOv5 functions:\n{}'.format(str(e)))

    assert utils_imported, 'YOLO utils import error'

    yolo_model_type_imported = model_type
    if verbose:
        print('Prepared YOLO imports for model type {}'.format(model_type))

    return model_type

# ...def _initialize_yolo_imports(...)


#%% NMS

def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Non-maximum suppression (a wrapper around torchvision.ops.nms())

    Args:
        prediction (torch.Tensor): Model predictions with shape [batch_size, num_anchors, num_classes + 5]
            Format: [x_center, y_center, width, height, objectness, class1_conf, class2_conf, ...]
            Coordinates are normalized to input image size.
        conf_thres (float): Confidence threshold for filtering detections
        iou_thres (float): IoU threshold for NMS
        max_det (int): Maximum number of detections per image

    Returns:
        list: List of tensors, one per image in batch. Each tensor has shape [N, 6] where:
                - N is the number of detections for that image
                - Columns are [x1, y1, x2, y2, confidence, class_id]
                - Coordinates are in absolute pixels relative to input image size
                - class_id is the integer class index (0-based)
    """

    batch_size = prediction.shape[0]
    num_classes = prediction.shape[2] - 5 # noqa
    output = []

    # Process each image in the batch
    for img_idx in range(batch_size):

        x = prediction[img_idx]  # Shape: [num_anchors, num_classes + 5]

        # Filter by objectness confidence
        obj_conf = x[:, 4]
        valid_detections = obj_conf > conf_thres
        x = x[valid_detections]

        if x.shape[0] == 0:
            # No detections for this image
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        # Convert box coordinates from [x_center, y_center, w, h] to [x1, y1, x2, y2]
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2.0  # x1 = center_x - width/2
        box[:, 1] = x[:, 1] - x[:, 3] / 2.0  # y1 = center_y - height/2
        box[:, 2] = x[:, 0] + x[:, 2] / 2.0  # x2 = center_x + width/2
        box[:, 3] = x[:, 1] + x[:, 3] / 2.0  # y2 = center_y + height/2

        # Get class predictions: multiply objectness by class probabilities
        class_conf = x[:, 5:] * x[:, 4:5]  # shape: [N, num_classes]

        # For each detection, take the class with highest confidence (single-label)
        best_class_conf, best_class_idx = class_conf.max(1, keepdim=True)

        # Filter by class confidence threshold
        conf_mask = best_class_conf.view(-1) > conf_thres
        if conf_mask.sum() == 0:
            # No detections pass confidence threshold
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        box = box[conf_mask]
        best_class_conf = best_class_conf[conf_mask]
        best_class_idx = best_class_idx[conf_mask]

        # Prepare for NMS: group detections by class
        unique_classes = best_class_idx.unique()
        final_detections = []

        for class_id in unique_classes:

            class_mask = (best_class_idx == class_id).view(-1)
            class_boxes = box[class_mask]
            class_scores = best_class_conf[class_mask].view(-1)

            if class_boxes.shape[0] == 0:
                continue

            # Apply NMS for this class
            keep_indices = torchvision.ops.nms(class_boxes, class_scores, iou_thres)

            if len(keep_indices) > 0:
                kept_boxes = class_boxes[keep_indices]
                kept_scores = class_scores[keep_indices]
                kept_classes = torch.full((len(keep_indices), 1), class_id.item(),
                                        device=prediction.device, dtype=torch.float)

                # Combine: [x1, y1, x2, y2, conf, class]
                class_detections = torch.cat([kept_boxes, kept_scores.unsqueeze(1), kept_classes], 1)
                final_detections.append(class_detections)

        # ...for each category

        if final_detections:

            # Combine all classes and sort by confidence
            all_detections = torch.cat(final_detections, 0)
            conf_sort_indices = all_detections[:, 4].argsort(descending=True)
            all_detections = all_detections[conf_sort_indices]

            # Limit to max_det
            if all_detections.shape[0] > max_det:
                all_detections = all_detections[:max_det]

            output.append(all_detections)
        else:
            output.append(torch.zeros((0, 6), device=prediction.device))

    # ...for each image in the batch

    return output

# ...def nms(...)


#%% Model metadata functions

def add_metadata_to_megadetector_model_file(model_file_in,
                                            model_file_out,
                                            metadata,
                                            destination_path='megadetector_info.json'):
    """
    Adds a .json file to the specified MegaDetector model file containing metadata used
    by this module.  Always over-writes the output file.

    Args:
        model_file_in (str): The input model filename, typically .pt (.zip is also sensible)
        model_file_out (str): The output model filename, typically .pt (.zip is also sensible).
            May be the same as model_file_in.
        metadata (dict): The metadata dict to add to the output model file
        destination_path (str, optional): The relative path within the main folder of the
            model archive where we should write the metadata.  This is not relative to the root
            of the archive, it's relative to the one and only folder at the root of the archive
            (this is a PyTorch convention).
    """

    tmp_base = os.path.join(tempfile.gettempdir(),'md_metadata')
    os.makedirs(tmp_base,exist_ok=True)
    metadata_tmp_file_relative = 'megadetector_info_' + str(uuid.uuid1()) + '.json'
    metadata_tmp_file_abs = os.path.join(tmp_base,metadata_tmp_file_relative)

    with open(metadata_tmp_file_abs,'w') as f:
        json.dump(metadata,f,indent=1)

    # Copy the input file to the output file
    shutil.copyfile(model_file_in,model_file_out)

    # Write metadata to the output file
    with zipfile.ZipFile(model_file_out, 'a', compression=zipfile.ZIP_DEFLATED) as zipf:

        # Torch doesn't like anything in the root folder of the zipfile, so we put
        # it in the one and only folder.
        names = zipf.namelist()
        root_folders = set()
        for name in names:
            root_folder = name.split('/')[0]
            root_folders.add(root_folder)
        assert len(root_folders) == 1,\
            'This archive does not have exactly one folder at the top level; are you sure it\'s a Torch model file?'
        root_folder = next(iter(root_folders))

        zipf.write(metadata_tmp_file_abs,
                   root_folder + '/' + destination_path,
                   compresslevel=9,
                   compress_type=zipfile.ZIP_DEFLATED)

    try:
        os.remove(metadata_tmp_file_abs)
    except Exception as e:
        print('Warning: error deleting file {}: {}'.format(metadata_tmp_file_abs,str(e)))

# ...def add_metadata_to_megadetector_model_file(...)


def read_metadata_from_megadetector_model_file(model_file,
                                               relative_path='megadetector_info.json',
                                               verbose=False):
    """
    Reads custom MegaDetector metadata from a modified MegaDetector model file.

    Args:
        model_file (str): The model filename to read, typically .pt (.zip is also sensible)
        relative_path (str, optional): The relative path within the main folder of the model
            archive from which we should read the metadata.  This is not relative to the root
            of the archive, it's relative to the one and only folder at the root of the archive
            (this is a PyTorch convention).
        verbose (str, optional): enable additional debug output

    Returns:
        object: whatever we read from the metadata file, always a dict in practice.  Returns
        None if we failed to read the specified metadata file.
    """

    with zipfile.ZipFile(model_file,'r') as zipf:

        # Torch doesn't like anything in the root folder of the zipfile, so we put
        # it in the one and only folder.
        names = zipf.namelist()
        root_folders = set()
        for name in names:
            root_folder = name.split('/')[0]
            root_folders.add(root_folder)
        if len(root_folders) != 1:
            print('Warning: this archive does not have exactly one folder at the top level; ' + \
                  'are you sure it\'s a Torch model file?')
            return None
        root_folder = next(iter(root_folders))

        metadata_file = root_folder + '/' + relative_path
        if metadata_file not in names:
            # This is the case for MDv5a and MDv5b
            if verbose:
                print('Warning: could not find metadata file {} in zip archive {}'.format(
                    metadata_file,os.path.basename(model_file)))
            return None

        try:
            path = zipfile.Path(zipf,metadata_file)
            contents = path.read_text()
            d = json.loads(contents)
        except Exception as e:
            print('Warning: error reading metadata from path {}: {}'.format(metadata_file,str(e)))
            return None

        return d

    # ...with zipfile.Zipfile(...)

# ...def read_metadata_from_megadetector_model_file(...)


#%% Inference classes

default_compatibility_mode = 'classic'

# This is a useful hack when I want to verify that my test driver (md_tests.py) is
# correctly forcing a specific compatibility mode (I use "classic-test" in that case)
require_non_default_compatibility_mode = False

class PTDetector:
    """
    Class that runs a PyTorch-based MegaDetector model.  Also used as a preprocessor
    for images that will later be run through an instance of PTDetector.
    """

    def __init__(self, model_path, detector_options=None, verbose=False):
        """
        PTDetector constructor.  If detector_options['preprocess_only'] exists and is
        True, this instance is being used as a preprocessor, so we don't load model weights.
        """

        if verbose:
            print('Initializing PTDetector (verbose)')

        # Set up the import environment for this model, unloading previous
        # YOLO library versions if necessary.
        _initialize_yolo_imports_for_model(model_path,
                                           detector_options=detector_options,
                                           verbose=verbose)

        # Parse options specific to this detector family
        force_cpu = False
        use_model_native_classes = False
        compatibility_mode = default_compatibility_mode

        if detector_options is not None:

            if 'force_cpu' in detector_options:
                force_cpu = parse_bool_string(detector_options['force_cpu'])
            if 'use_model_native_classes' in detector_options:
                use_model_native_classes = parse_bool_string(detector_options['use_model_native_classes'])
            if 'compatibility_mode' in detector_options:
                if detector_options['compatibility_mode'] is None:
                    compatibility_mode = default_compatibility_mode
                else:
                    compatibility_mode = detector_options['compatibility_mode']

        # This is a global option used only during testing, to make sure I'm hitting
        # the cases where we are not using "classic" preprocessing.
        if require_non_default_compatibility_mode:

            print('### DEBUG: requiring non-default compatibility mode ###')
            assert compatibility_mode != 'classic'
            assert compatibility_mode != 'default'

        preprocess_only = False
        if (detector_options is not None) and \
           ('preprocess_only' in detector_options) and \
           (detector_options['preprocess_only']):
            preprocess_only = True

        if verbose or (not preprocess_only):
            print('Loading PT detector with compatibility mode {}'.format(compatibility_mode))

        self.model_metadata = read_metadata_from_megadetector_model_file(model_path)

        #: Image size passed to the letterbox() function; 1280 means "1280 on the long side,
        #: preserving aspect ratio".
        if self.model_metadata is not None and 'image_size' in self.model_metadata:
            self.default_image_size = self.model_metadata['image_size']
            print('Loaded image size {} from model metadata'.format(self.default_image_size))
        else:
            # This is not the default for most YOLO models, but most of the time, if someone
            # is loading a model here that does not have metadata, it's MDv5[ab].0.0
            print('No image size available in model metadata, defaulting to 1280')
            self.default_image_size = 1280

        #: Either a string ('cpu','cuda:0') or a torch.device()
        self.device = 'cpu'

        #: Have we already printed a warning about using a non-standard image size?
        #:
        #: :meta private:
        self.printed_image_size_warning = False

        #: If this is False, we assume the underlying model is producing class indices in the
        #: set (0,1,2) (and we assert() on this), and we add 1 to get to the backwards-compatible
        #: MD classes (1,2,3) before generating output.  If this is True, we use whatever
        #: indices the model provides
        self.use_model_native_classes = use_model_native_classes

        #: This allows us to maintain backwards compatibility across a set of changes to the
        #: way this class does inference.  Currently should start with either "default" or
        #: "classic".
        self.compatibility_mode = compatibility_mode

        #: Stride size passed to the YOLO letterbox() function
        self.letterbox_stride = 32

        # This is a convenient heuristic to determine the stride size without actually loading
        # the model: the only models in the YOLO family with a stride size of 64 are the
        # YOLOv5*6 and YOLOv5*6u models, which are 1280px models.
        #
        # See:
        #
        # github.com/ultralytics/ultralytics/issues/21544
        #
        # Note to self, though, if I decide later to require loading the model on preprocessing
        # workers so I can more reliably choose a stride, this is the right way to determine the
        # stride:
        #
        # self.letterbox_stride = int(self.model.stride.max())
        if self.default_image_size == 1280:
            self.letterbox_stride = 64

        print('Using model stride: {}'.format(self.letterbox_stride))

        #: Use half-precision inference... fixed by the model, generally don't mess with this
        self.half_precision = False

        if preprocess_only:
            return

        if not force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            try:
                if torch.backends.mps.is_built and torch.backends.mps.is_available():
                    # MPS inference fails on GitHub runners as of 2025.08.  This is
                    # independent of model size.  So, we disable MPS when running in GHA.
                    if is_running_in_gha():
                        print('GitHub actions detected, bypassing MPS backend')
                    else:
                        print('Using MPS device')
                        self.device = 'mps'
            except AttributeError:
                pass

        # AddaxAI depends on this printout, don't remove it
        print('PTDetector using device {}'.format(str(self.device).lower()))

        try:
            self.model = PTDetector._load_model(model_path,
                                                device=self.device,
                                                compatibility_mode=self.compatibility_mode)

        except Exception as e:
            # In a very esoteric scenario where an old version of YOLOv5 is used to run
            # newer models, we run into an issue because the "Model" class became
            # "DetectionModel".  New YOLOv5 code handles this case by just setting them
            # to be the same, so doing that externally doesn't seem *that* rude.
            if "Can't get attribute 'DetectionModel'" in str(e):
                print('Forward-compatibility issue detected, patching')
                from models import yolo # type: ignore
                yolo.DetectionModel = yolo.Model
                self.model = PTDetector._load_model(model_path,
                                                    device=self.device,
                                                    compatibility_mode=self.compatibility_mode,
                                                    verbose=verbose)
            else:
                raise
        if (self.device != 'cpu'):
            if verbose:
                print('Sending model to GPU')
            self.model.to(self.device)


    @staticmethod
    def _load_model(model_pt_path, device, compatibility_mode='', verbose=False):

        if verbose:
            print(f'Using PyTorch version {torch.__version__}')

        # I get quirky errors when loading YOLOv5 models on MPS hardware using
        # map_location, but this is the recommended method, so I'm using it everywhere
        # other than MPS devices.
        use_map_location = (device != 'mps')

        if use_map_location:
            try:
                checkpoint = torch.load(model_pt_path, map_location=device, weights_only=False)
            # For a transitional period, we want to support torch 1.1x, where the weights_only
            # parameter doesn't exist
            except Exception as e:
                if "'weights_only' is an invalid keyword" in str(e):
                    checkpoint = torch.load(model_pt_path, map_location=device)
                else:
                    raise
        else:
            try:
                checkpoint = torch.load(model_pt_path, weights_only=False)
            # For a transitional period, we want to support torch 1.1x, where the weights_only
            # parameter doesn't exist
            except Exception as e:
                if "'weights_only' is an invalid keyword" in str(e):
                    checkpoint = torch.load(model_pt_path)
                else:
                    raise

        # Compatibility fix that allows us to load older YOLOv5 models with
        # newer versions of YOLOv5/PT
        for m in checkpoint['model'].modules():
            t = type(m)
            if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None

        # Calling .to(device) should no longer be necessary now that we're using map_location=device
        # model = checkpoint['model'].float().fuse().eval().to(device)
        model = checkpoint['model'].float().fuse().eval()

        return model

    # ...def _load_model(...)


    def preprocess_image(self,
                         img_original,
                         image_id='unknown',
                         image_size=None,
                         verbose=False):
        """
        Prepare an image for detection, including scaling and letterboxing.

        Args:
            img_original (Image or np.array): the image on which we should run the detector, with
                EXIF rotation already handled
            image_id (str, optional): a path to identify the image; will be in the "file" field
                of the output object
            detection_threshold (float, optional): only detections above this confidence threshold
                will be included in the return value
            image_size (int, optional): image size (long side) to use for inference, or None to
                use the default size specified at the time the model was loaded
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: dict with fields:
                - file (filename)
                - img (the preprocessed np.array)
                - img_original (the input image before preprocessing, as an np.array)
                - img_original_pil (the input image before preprocessing, as a PIL Image)
                - target_shape (the 2D shape to which the image was resized during preprocessing)
                - scaling_shape (the 2D original size, for normalizing coordinates later)
                - letterbox_ratio (letterbox parameter used for normalizing coordinates later)
                - letterbox_pad (letterbox parameter used for normalizing coordinates later)
        """

        # Prepare return dict
        result = {'file': image_id }

        # Store the PIL version of the original image, the caller may want to use
        # it for metadata extraction later.
        img_original_pil = None

        # If we were given a PIL image, rather than a numpy array
        if not isinstance(img_original,np.ndarray):
            img_original_pil = img_original
            img_original = np.asarray(img_original)

        # PIL images are RGB already
        # img_original = img_original[:, :, ::-1]

        # Save the original shape for scaling boxes later
        scaling_shape = img_original.shape

        # If the caller is requesting a specific target size...
        if image_size is not None:

            assert isinstance(image_size,int)

            if not self.printed_image_size_warning:
                print('Using user-supplied image size {}'.format(image_size))
                self.printed_image_size_warning = True

        # Otherwise resize to self.default_image_size
        else:

            image_size = self.default_image_size
            self.printed_image_size_warning = False

        # ...if the caller has specified an image size

        # In "classic mode", we only do the letterboxing resize, we don't do an
        # additional initial resizing operation
        if 'classic' in self.compatibility_mode:

            resize_ratio = 1.0

        # Resize the image so the long side matches the target image size.  This is not
        # letterboxing (i.e., padding) yet, just resizing.
        else:

            use_ceil_for_resize = ('use_ceil_for_resize' in self.compatibility_mode)

            h,w = img_original.shape[:2]
            resize_ratio = image_size / max(h,w)

            # Only resize if we have to
            if resize_ratio != 1:

                # Match what yolov5 does: use linear interpolation for upsizing;
                # area interpolation for downsizing
                if resize_ratio > 1:
                    interpolation_method = cv2.INTER_LINEAR
                else:
                    interpolation_method = cv2.INTER_AREA

                if use_ceil_for_resize:
                    target_w = math.ceil(w * resize_ratio)
                    target_h = math.ceil(h * resize_ratio)
                else:
                    target_w = int(w * resize_ratio)
                    target_h = int(h * resize_ratio)

                img_original = cv2.resize(
                    img_original, (target_w, target_h),
                    interpolation=interpolation_method)

        if 'classic' in self.compatibility_mode:

            letterbox_auto = True
            letterbox_scaleup = True
            target_shape = image_size

        else:

            letterbox_auto = False
            letterbox_scaleup = False

            # The padding to apply as a fraction of the stride size
            pad = 0.5

            # Resize to a multiple of the model stride
            #
            # This is how we would determine the stride if we knew the model had been loaded:
            #
            # model_stride = int(self.model.stride.max())
            #
            # ...but because we do this on preprocessing workers now, we try to avoid loading the model
            # just for preprocessing, and we assume the stride was determined at the time the PTDetector
            # object was created.
            try:
                model_stride = int(self.model.stride.max())
                if model_stride != self.letterbox_stride:
                    print('*** Warning: model stride is {}, stride at construction time was {} ***'.format(
                        model_stride,self.letterbox_stride
                    ))
            except Exception:
                pass

            model_stride = self.letterbox_stride
            max_dimension = max(img_original.shape)
            normalized_shape = [img_original.shape[0] / max_dimension,
                                img_original.shape[1] / max_dimension]
            target_shape = np.ceil(((np.array(normalized_shape) * image_size) / model_stride) + \
                                   pad).astype(int) * model_stride

        # Now we letterbox, which is just padding, since we've already resized
        img,letterbox_ratio,letterbox_pad = letterbox(img_original,
                                                      new_shape=target_shape,
                                                      stride=self.letterbox_stride,
                                                      auto=letterbox_auto,
                                                      scaleFill=False,
                                                      scaleup=letterbox_scaleup)

        result['img_processed'] = img
        result['img_original'] = img_original
        result['img_original_pil'] = img_original_pil
        result['target_shape'] = target_shape
        result['scaling_shape'] = scaling_shape
        result['letterbox_ratio'] = letterbox_ratio
        result['letterbox_pad'] = letterbox_pad
        return result

    # ...def preprocess_image(...)


    def generate_detections_one_batch(self,
                                      img_original,
                                      image_id=None,
                                      detection_threshold=0.00001,
                                      image_size=None,
                                      augment=False,
                                      verbose=False):
        """
        Run a detector on a batch of images.

        Args:
            img_original (list): list of images (Image, np.array, or dict) on which we should run the detector, with
                EXIF rotation already handled, or dicts representing preprocessed images with associated
                letterbox parameters
            image_id (list or None): list of paths to identify the images; will be in the "file" field
                of the output objects. Will be ignored when img_original contains preprocessed dicts.
            detection_threshold (float, optional): only detections above this confidence threshold
                will be included in the return value
            image_size (int, optional): image size (long side) to use for inference, or None to
                use the default size specified at the time the model was loaded
            augment (bool, optional): enable (implementation-specific) image augmentation
            verbose (bool, optional): enable additional debug output

        Returns:
            list: a list of dictionaries, each with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, or None if everything went fine)
        """

        # Validate inputs
        if not isinstance(img_original, list):
            raise ValueError('img_original must be a list for batch processing')

        if len(img_original) == 0:
            return []

        # Check input consistency
        if isinstance(img_original[0], dict):
            # All items in img_original should be preprocessed dicts
            for i, img in enumerate(img_original):
                if not isinstance(img, dict):
                    raise ValueError(f'Mixed input types in batch: item {i} is not a dict, but item 0 is a dict')
        else:
            # All items in img_original should be PIL/numpy images, and image_id should be a list of strings
            if image_id is None:
                raise ValueError('image_id must be a list when img_original contains PIL/numpy images')
            if not isinstance(image_id, list):
                raise ValueError('image_id must be a list for batch processing')
            if len(image_id) != len(img_original):
                raise ValueError(
                    'Length mismatch: img_original has {} items, image_id has {} items'.format(
                    len(img_original),len(image_id)))
            for i_img, img in enumerate(img_original):
                if isinstance(img, dict):
                    raise ValueError(
                        'Mixed input types in batch: item {} is a dict, but item 0 is not a dict'.format(
                            i_img))

        if detection_threshold is None:
            detection_threshold = 0.0

        batch_size = len(img_original)
        results = [None] * batch_size

        # Preprocess all images, handling failures
        preprocessed_images = []
        preprocessing_failed_indices = set()

        for i_img, img in enumerate(img_original):

            try:
                if isinstance(img, dict):
                    # Already preprocessed
                    image_info = img
                    current_image_id = image_info['file']
                else:
                    # Need to preprocess
                    current_image_id = image_id[i_img]
                    image_info = self.preprocess_image(
                        img_original=img,
                        image_id=current_image_id,
                        image_size=image_size,
                        verbose=verbose)

                preprocessed_images.append((i_img, image_info, current_image_id))

            except Exception as e:
                print('Warning: preprocessing failed for image {}: {}'.format(
                    image_id[i_img] if image_id else f'index_{i_img}', str(e)))

                preprocessing_failed_indices.add(i_img)
                current_image_id = image_id[i_img] if image_id else f'index_{i_img}'
                results[i_img] = {
                    'file': current_image_id,
                    'detections': None,
                    'failure': FAILURE_IMAGE_OPEN
                }

        # ...for each image in this batch

        # Group preprocessed images by actual processed image shape for batching
        shape_groups = {}
        for original_idx, image_info, current_image_id in preprocessed_images:
            # Use the actual processed image shape for grouping, not target_shape
            actual_shape = tuple(image_info['img_processed'].shape)
            if actual_shape not in shape_groups:
                shape_groups[actual_shape] = []
            shape_groups[actual_shape].append((original_idx, image_info, current_image_id))

        # Process each shape group as a batch
        for target_shape, group_items in shape_groups.items():

            try:
                self._process_batch_group(group_items, results, detection_threshold, augment, verbose)
            except Exception as e:
                # If inference fails for the entire batch, mark all images in this batch as failed
                print('Warning: batch inference failed for shape {}: {}'.format(target_shape, str(e)))

                for original_idx, image_info, current_image_id in group_items:
                    results[original_idx] = {
                        'file': current_image_id,
                        'detections': None,
                        'failure': FAILURE_INFER
                    }

        # ...for each shape group
        return results

    # ...def generate_detections_one_batch(...)


    def _process_batch_group(self, group_items, results, detection_threshold, augment, verbose):
        """
        Process a group of images with the same target shape as a single batch.

        Args:
            group_items (list): List of (original_idx, image_info, current_image_id) tuples
            results (list): Results list to populate (modified in place)
            detection_threshold (float): Detection confidence threshold
            augment (bool): Enable augmentation
            verbose (bool): Enable verbose output

        Returns:
            list of dict: list of dictionaries the same length as group_items, with fields 'file',
            'detections', 'max_detection_conf'.
        """

        if len(group_items) == 0:
            return

        # Extract batch data
        batch_images = []
        batch_metadata = []

        # For each image in this batch...
        for original_idx, image_info, current_image_id in group_items:

            img = image_info['img_processed']

            # Convert HWC to CHW and prepare tensor
            img_tensor = img.transpose((2, 0, 1))
            img_tensor = np.ascontiguousarray(img_tensor)
            img_tensor = torch.from_numpy(img_tensor)
            batch_images.append(img_tensor)

            metadata = {
                'original_idx': original_idx,
                'current_image_id': current_image_id,
                'scaling_shape': image_info['scaling_shape'],
                'letterbox_pad': image_info['letterbox_pad'],
                'img_original': image_info['img_original']
            }
            batch_metadata.append(metadata)

        # ...for each image in this batch

        # Stack images into a batch tensor
        batch_tensor = torch.stack(batch_images)

        batch_tensor = batch_tensor.float()
        batch_tensor /= 255.0

        batch_tensor = batch_tensor.to(self.device)
        if self.half_precision:
            batch_tensor = batch_tensor.half()

        # Run the model on the batch
        pred = self.model(batch_tensor, augment=augment)[0]

        # Configure NMS parameters
        if 'classic' in self.compatibility_mode:
            nms_iou_thres = 0.45
        else:
            nms_iou_thres = 0.6

        use_library_nms = False

        # Model output format changed in recent ultralytics packages, and the nms implementation
        # in this module hasn't been updated to handle that format yet.
        if (yolo_model_type_imported is not None) and (yolo_model_type_imported == 'ultralytics'):
            use_library_nms = True

        if use_library_nms:
            pred = non_max_suppression(prediction=pred,
                                    conf_thres=detection_threshold,
                                    iou_thres=nms_iou_thres,
                                    agnostic=False,
                                    multi_label=False)
        else:
            pred = nms(prediction=pred,
                    conf_thres=detection_threshold,
                    iou_thres=nms_iou_thres)

        assert isinstance(pred, list)
        assert len(pred) == len(batch_metadata), \
            'Mismatch between prediction length {} and batch size {}'.format(
                   len(pred),len(batch_metadata))

        # Process each image's detections
        for i_image, det in enumerate(pred):

            metadata = batch_metadata[i_image]
            original_idx = metadata['original_idx']
            current_image_id = metadata['current_image_id']
            scaling_shape = metadata['scaling_shape']
            letterbox_pad = metadata['letterbox_pad']
            img_original = metadata['img_original']

            detections = []
            max_conf = 0.0

            if len(det) > 0:

                # Prepare scaling parameters
                gn = torch.tensor(scaling_shape)[[1, 0, 1, 0]]

                if 'classic' in self.compatibility_mode:
                    ratio = None
                    ratio_pad = None
                else:
                    ratio = (img_original.shape[0]/scaling_shape[0],
                             img_original.shape[1]/scaling_shape[1])
                    ratio_pad = (ratio, letterbox_pad)

                # Rescale boxes
                if 'classic' in self.compatibility_mode:
                    det[:, :4] = scale_coords(batch_tensor.shape[2:], det[:, :4], img_original.shape).round()
                else:
                    det[:, :4] = scale_coords(batch_tensor.shape[2:], det[:, :4], scaling_shape, ratio_pad).round()

                # Process each detection
                for *xyxy, conf, cls in reversed(det):
                    if conf < detection_threshold:
                        continue

                    # Convert to YOLO format then to MD format
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    api_box = ct_utils.convert_yolo_to_xywh(xywh)

                    if 'classic' in self.compatibility_mode:
                        api_box = ct_utils.truncate_float_array(api_box, precision=COORD_DIGITS)
                        conf = ct_utils.truncate_float(conf.tolist(), precision=CONF_DIGITS)
                    else:
                        api_box = ct_utils.round_float_array(api_box, precision=COORD_DIGITS)
                        conf = ct_utils.round_float(conf.tolist(), precision=CONF_DIGITS)

                    if not self.use_model_native_classes:
                        cls = int(cls.tolist()) + 1
                        if cls not in (1, 2, 3):
                            raise KeyError(f'{cls} is not a valid class.')
                    else:
                        cls = int(cls.tolist())

                    detections.append({
                        'category': str(cls),
                        'conf': conf,
                        'bbox': api_box
                    })
                    max_conf = max(max_conf, conf)

                # ...for each detection

            # ...if there are > 0 detections

            # Store result for this image
            results[original_idx] = {
                'file': current_image_id,
                'detections': detections,
                'max_detection_conf': max_conf
            }

        # ...for each image

    # ...def _process_batch_group(...)

    def generate_detections_one_image(self,
                                      img_original,
                                      image_id='unknown',
                                      detection_threshold=0.00001,
                                      image_size=None,
                                      augment=False,
                                      verbose=False):
        """
        Run a detector on an image (wrapper around batch function).

        Args:
            img_original (Image, np.array, or dict): the image on which we should run the detector, with
                EXIF rotation already handled, or a dict representing a preprocessed image with associated
                letterbox parameters
            image_id (str, optional): a path to identify the image; will be in the "file" field
                of the output object
            detection_threshold (float, optional): only detections above this confidence threshold
                will be included in the return value
            image_size (int, optional): image size (long side) to use for inference, or None to
                use the default size specified at the time the model was loaded
            augment (bool, optional): enable (implementation-specific) image augmentation
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: a dictionary with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, or None if everything went fine)
        """

        # Prepare batch inputs
        if isinstance(img_original, dict):
            batch_results = self.generate_detections_one_batch(
                img_original=[img_original],
                image_id=None,
                detection_threshold=detection_threshold,
                image_size=image_size,
                augment=augment,
                verbose=verbose)
        else:
            batch_results = self.generate_detections_one_batch(
                img_original=[img_original],
                image_id=[image_id],
                detection_threshold=detection_threshold,
                image_size=image_size,
                augment=augment,
                verbose=verbose)

        # Return the single result
        return batch_results[0]

    # ...def generate_detections_one_image(...)

# ...class PTDetector
