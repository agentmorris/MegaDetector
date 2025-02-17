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
import traceback
import uuid
import json

import cv2
import torch
import numpy as np

from megadetector.detection.run_detector import \
    CONF_DIGITS, COORD_DIGITS, FAILURE_INFER, \
    get_detector_version_from_model_file, \
    known_models
from megadetector.utils.ct_utils import parse_bool_string
from megadetector.utils import ct_utils

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
            print('Waring: model type from model version is {}, from file metadata is {}'.format(
                model_type_from_model_version,model_type_from_model_file_metadata))
            if prefer_model_type_source == 'table':
                model_type = model_type_from_model_file_metadata
            else:
                model_type = model_type_from_model_version
    
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
        verbose (bool, optional): enable additonal debug output
            
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


def _clean_yolo_imports(verbose=False):
    """
    Remove all YOLO-related imports from sys.modules and sys.path, to allow a clean re-import
    of another YOLO library version.  The reason we jump through all these hoops, rather than 
    just, e.g., handling different libraries in different modules, is that we need to make sure
    *pickle* sees the right version of modules during module loading, including modules we don't
    load directly (i.e., every module loaded within a YOLO library), and the only way I know to 
    do that is to remove all the "wrong" versions from sys.modules and sys.path.
    
    Args:
        verbose (bool, optional): enable additional debug output
    """
    
    modules_to_delete = []
    for module_name in sys.modules.keys():        
        module = sys.modules[module_name]
        try:
            module_file = module.__file__.replace('\\','/')
            if 'site-packages' not in module_file:
                continue
            tokens = module_file.split('/')[-4:]
            for token in tokens:
                if 'yolov5' in token or 'yolov9' in token or 'ultralytics' in token:
                    modules_to_delete.append(module_name)
                    break                    
        except Exception:
            pass
        
    for module_name in modules_to_delete:
        if module_name in sys.modules.keys():
            module_file = module.__file__.replace('\\','/')
            if verbose:
                print('clean_yolo_imports: deleting module {}: {}'.format(module_name,module_file))
            del sys.modules[module_name]
    
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
        verbose (bool, optional): include additonal debug output
            
    Returns:
        str: the model type for which we initialized support
    """
    
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
        if yolo_model_type_imported == model_type:
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
            from yolov5.utils.general import non_max_suppression, xyxy2xywh # noqa
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
            
            from yolov9.utils.general import non_max_suppression, xyxy2xywh # noqa
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
            
            import ultralytics # noqa
            
        except Exception:
            
            print('It looks like you are trying to run a model that requires the ultralytics package, '
                  'but the ultralytics package is not installed, but .  For licensing reasons, this '
                  'is not installed by default with the MegaDetector Python package.  Run '
                  '"pip install ultralytics" to install it, and try again.')
            raise
                 
        try:
            
            from ultralytics.utils.ops import non_max_suppression # noqa
            from ultralytics.utils.ops import xyxy2xywh # noqa
            
            # In the ultralytics package, scale_boxes and scale_coords both exist;
            # we want scale_boxes.
            #        
            # from ultralytics.utils.ops import scale_coords # noqa
            from ultralytics.utils.ops import scale_boxes as scale_coords # noqa
            from ultralytics.data.augment import LetterBox
            
            # letterbox() became a LetterBox class in the ultralytics package.  Create a 
            # backwards-compatible letterbox function wrapper that wraps the class up.
            def letterbox(img,new_shape,auto=False,scaleFill=False,scaleup=True,center=True,stride=32): # noqa
                
                L = LetterBox(new_shape,auto=auto,scaleFill=scaleFill,scaleup=scaleup,center=center,stride=stride)
                letterbox_result = L(image=img)
            
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
            
        except Exception:
            
            # print('Ultralytics module import failed')
            pass
    
    # If we haven't succeeded yet, assume the YOLOv5 repo is on our PYTHONPATH.
    if (not utils_imported) and allow_fallback_import:
        
        try:
            
            # import pre- and post-processing functions from the YOLOv5 repo
            from utils.general import non_max_suppression, xyxy2xywh # noqa
            from utils.augmentations import letterbox # noqa
            
            # scale_coords() is scale_boxes() in some YOLOv5 versions
            try:
                from utils.general import scale_coords # noqa
            except ImportError:
                from utils.general import scale_boxes as scale_coords
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
    
    Returns:
        object: Whatever we read from the metadata file, always a dict in practice.  Returns
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
            print('Warning: this archive does not have exactly one folder at the top level; are you sure it\'s a Torch model file?')
            return None
        root_folder = next(iter(root_folders))
        
        metadata_file = root_folder + '/' + relative_path
        if metadata_file not in names:
            # This is the case for MDv5a and MDv5b
            if verbose:
                print('Warning: could not find metadata file {} in zip archive'.format(metadata_file))
            return None
    
        try:
            path = zipfile.Path(zipf,metadata_file)
            contents = path.read_text()
            d = json.loads(contents)
        except Exception as e:
            print('Warning: error reading metadata from path {}: {}'.format(metadata_file,str(e)))
            return None
        
        return d
        
# ...def read_metadata_from_megadetector_model_file(...)


#%% Inference classes

default_compatibility_mode = 'classic'

# This is a useful hack when I want to verify that my test driver (md_tests.py) is 
# correctly forcing a specific compabitility mode (I use "classic-test" in that case)
require_non_default_compatibility_mode = False

class PTDetector:
    
    def __init__(self, model_path, detector_options=None, verbose=False):
        
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
        
        model_metadata = read_metadata_from_megadetector_model_file(model_path)
        
        #: Image size passed to the letterbox() function; 1280 means "1280 on the long side, preserving 
        #: aspect ratio".
        if model_metadata is not None and 'image_size' in model_metadata:
            self.default_image_size = model_metadata['image_size']
            if verbose:
                print('Loaded image size {} from model metadata'.format(self.default_image_size))
        else:
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
        
        #: Stride size passed to YOLOv5's letterbox() function
        self.letterbox_stride = 32
        
        if 'classic' in self.compatibility_mode:
            self.letterbox_stride = 64
        
        #: Use half-precision inference... fixed by the model, generally don't mess with this
        self.half_precision = False
        
        if preprocess_only:
            return
            
        if not force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            try:
                if torch.backends.mps.is_built and torch.backends.mps.is_available():
                    self.device = 'mps'
            except AttributeError:
                pass
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
                from models import yolo
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

        # There are two very slightly different ways to load the model, (1) using the
        # map_location=device parameter to torch.load and (2) calling .to(device) after
        # loading the model.  The former is what we did for a zillion years, but is not
        # supported on Apple silicon at of 2029.09.  Switching to the latter causes
        # very slight changes to the output, which always make me nervous, so I'm not
        # doing a wholesale swap just yet.  Instead, we'll just do this on M1 hardware.
        if 'classic' in compatibility_mode:
            use_map_location = (device != 'mps')        
        else:
            use_map_location = False
        
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
        
        if use_map_location:            
            model = checkpoint['model'].float().fuse().eval()            
        else:
            model = checkpoint['model'].float().fuse().eval().to(device)            
            
        return model

    # ...def _load_model(...)
    

    def generate_detections_one_image(self, 
                                      img_original, 
                                      image_id='unknown', 
                                      detection_threshold=0.00001, 
                                      image_size=None,
                                      skip_image_resizing=False,
                                      augment=False,
                                      preprocess_only=False,
                                      verbose=False):
        """
        Applies the detector to an image.

        Args:
            img_original (Image): the PIL Image object (or numpy array) on which we should run the 
                detector, with EXIF rotation already handled
            image_id (str, optional): a path to identify the image; will be in the "file" field 
                of the output object
            detection_threshold (float, optional): only detections above this confidence threshold 
                will be included in the return value
            image_size (tuple, optional): image size to use for inference, only mess with this if 
                (a) you're using a model other than MegaDetector or (b) you know what you're getting into
            skip_image_resizing (bool, optional): whether to skip internal image resizing (and rely on 
                external resizing), only mess with this if (a) you're using a model other than MegaDetector 
                or (b) you know what you're getting into
            augment (bool, optional): enable (implementation-specific) image augmentation
            preprocess_only (bool, optional): only run preprocessing, and return the preprocessed image
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: a dictionary with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, or None if everything went fine)
        """

        result = {'file': image_id }
        detections = []
        max_conf = 0.0

        if preprocess_only:
            assert 'classic' in self.compatibility_mode, \
                'Standalone preprocessing only supported in "classic" mode'
            assert not skip_image_resizing, \
                'skip_image_resizing and preprocess_only are exclusive' 
            
        if detection_threshold is None:
            
            detection_threshold = 0
            
        try:
            
            # If the caller wants us to skip all the resizing operations...
            if skip_image_resizing:
                
                if isinstance(img_original,dict):
                    image_info = img_original
                    img = image_info['img_processed']
                    scaling_shape = image_info['scaling_shape']
                    letterbox_pad = image_info['letterbox_pad']
                    letterbox_ratio = image_info['letterbox_ratio']
                    img_original = image_info['img_original']
                    img_original_pil = image_info['img_original_pil']
                else:
                    img = img_original
            
            else:
                
                img_original_pil = None
                # If we were given a PIL image
                
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
                    
                    model_stride = int(self.model.stride.max())
                    
                    max_dimension = max(img_original.shape)
                    normalized_shape = [img_original.shape[0] / max_dimension,
                                        img_original.shape[1] / max_dimension]
                    target_shape = np.ceil(np.array(normalized_shape) * image_size / model_stride + \
                                           pad).astype(int) * model_stride
                    
                # Now we letterbox, which is just padding, since we've already resized.
                img,letterbox_ratio,letterbox_pad = letterbox(img_original, 
                                                              new_shape=target_shape,
                                                              stride=self.letterbox_stride, 
                                                              auto=letterbox_auto,
                                                              scaleFill=False,
                                                              scaleup=letterbox_scaleup)
            
                if preprocess_only:
            
                    assert 'file' in result
                    result['img_processed'] = img
                    result['img_original'] = img_original
                    result['img_original_pil'] = img_original_pil
                    result['target_shape'] = target_shape
                    result['scaling_shape'] = scaling_shape
                    result['letterbox_ratio'] = letterbox_ratio
                    result['letterbox_pad'] = letterbox_pad
                    return result
                    
            # ...are we doing resizing here, or were images already resized?
            
            # Convert HWC to CHW (which is what the model expects).  The PIL Image is RGB already,
            # so we don't need to mess with the color channels.
            #
            # TODO, this could be moved into the preprocessing loop
            
            img = img.transpose((2, 0, 1)) # [::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)
            img = img.half() if self.half_precision else img.float()
            img /= 255

            # In practice this is always true 
            if len(img.shape) == 3:  
                img = torch.unsqueeze(img, 0)

            # Run the model
            pred = self.model(img,augment=augment)[0]

            if 'classic' in self.compatibility_mode:
                nms_conf_thres = detection_threshold
                nms_iou_thres = 0.45
                nms_agnostic = False
                nms_multi_label = False
            else:
                nms_conf_thres = detection_threshold # 0.01
                nms_iou_thres = 0.6
                nms_agnostic = False    
                nms_multi_label = True
                
            # As of PyTorch 1.13.0.dev20220824, nms is not implemented for MPS.
            #
            # Send predictions back to the CPU for NMS.            
            if self.device == 'mps':
                pred_nms = pred.cpu()
            else:
                pred_nms = pred
                
            # NMS
            pred = non_max_suppression(prediction=pred_nms, 
                                       conf_thres=nms_conf_thres,
                                       iou_thres=nms_iou_thres,
                                       agnostic=nms_agnostic,
                                       multi_label=nms_multi_label)

            # In practice this is [w,h,w,h] of the original image
            gn = torch.tensor(scaling_shape)[[1, 0, 1, 0]]

            if 'classic' in self.compatibility_mode:
                
                ratio = None
                ratio_pad = None
                
            else:
                
                # letterbox_pad is a 2-tuple specifying the padding that was added on each axis.
                #
                # ratio is a 2-tuple specifying the scaling that was applied to each dimension.
                #
                # The scale_boxes function expects a 2-tuple with these things combined.
                ratio = (img_original.shape[0]/scaling_shape[0], img_original.shape[1]/scaling_shape[1])
                ratio_pad = (ratio, letterbox_pad)                
                
            # This is a loop over detection batches, which will always be length 1 in our case,
            # since we're not doing batch inference.
            #
            # det = pred[0]
            #
            # det is a torch.Tensor with size [nBoxes,6].  In practice the boxes are sorted 
            # in descending order by confidence.
            #
            # Columns are:
            #
            # x0,y0,x1,y1,confidence,class
            #
            # At this point, these are *non*-normalized values, referring to the size at which we
            # ran inference (img.shape).
            for det in pred:
                
                if len(det) == 0:
                    continue
                                    
                # Rescale boxes from img_size to im0 size, and undo the effect of padded letterboxing
                if 'classic' in self.compatibility_mode:
                    
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()
                    
                else:
                    # After this scaling, each element of det is a box in x0,y0,x1,y1 format, referring to the
                    # original pixel dimension of the image, followed by the class and confidence
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], scaling_shape, ratio_pad).round()

                # Loop over detections
                for *xyxy, conf, cls in reversed(det):
                    
                    if conf < detection_threshold:
                        continue
                    
                    # Convert this box to normalized cx, cy, w, h (i.e., YOLO format)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    # Convert from normalized cx/cy/w/h (i.e., YOLO format) to normalized 
                    # left/top/w/h (i.e., MD format)
                    api_box = ct_utils.convert_yolo_to_xywh(xywh)

                    if 'classic' in self.compatibility_mode:
                        api_box = ct_utils.truncate_float_array(api_box, precision=COORD_DIGITS)
                        conf = ct_utils.truncate_float(conf.tolist(), precision=CONF_DIGITS)
                    else:
                        api_box = ct_utils.round_float_array(api_box, precision=COORD_DIGITS)
                        conf = ct_utils.round_float(conf.tolist(), precision=CONF_DIGITS)                            
                    
                    if not self.use_model_native_classes:
                        # The MegaDetector output format's categories start at 1, but all YOLO-based 
                        # MD models have category numbers starting at 0.                        
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
                    
                # ...for each detection in this batch
                    
            # ...for each detection batch (always one iteration)

        # ...try
        
        except Exception as e:
            
            result['failure'] = FAILURE_INFER
            print('PTDetector: image {} failed during inference: {}\n'.format(image_id, str(e)))
            # traceback.print_exc(e)
            print(traceback.format_exc())

        result['max_detection_conf'] = max_conf
        result['detections'] = detections

        return result

    # ...def generate_detections_one_image(...)

# ...class PTDetector


#%% Command-line driver

# For testing only... you don't really want to run this module directly.

if __name__ == '__main__':

    pass
    
    #%%
    
    import os #noqa
    from megadetector.visualization import visualization_utils as vis_utils
    
    model_file = os.environ['MDV5A']
    im_file = os.path.expanduser('~/git/MegaDetector/images/nacti.jpg')

    detector = PTDetector(model_file)
    image = vis_utils.load_image(im_file)

    res = detector.generate_detections_one_image(image, im_file, detection_threshold=0.00001)
    print(res)
