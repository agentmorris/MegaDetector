"""

pytorch_detector.py

Module to run YOLO-based MegaDetector models.

"""

#%% Imports and constants

import sys
import cv2
import math
import torch
import numpy as np
import traceback

from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS, FAILURE_INFER
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
#
#   If try_ultralytics_import is True, we'll try to import all YOLOv5 dependencies from 
#   ultralytics.utils and ultralytics.data.  But as of 2023.11, this results in a "No 
#   module named 'models'" error when running MDv5, and there's no upside to this approach
#   compared to using either of the YOLOv5 PyPI packages, so... punting on this for now.

utils_imported = False
try_yolov5_import = True
try_yolov9_import = True

# See above; this should remain as "False" unless we update the MegaDetector .pt file
# to use more recent YOLOv5 namespace conventions.
try_ultralytics_import = False


# First try importing from the yolov5 package; this is how the pip
# package finds YOLOv5 utilities.
if try_yolov5_import and not utils_imported:
    
    try:
        from yolov5.utils.general import non_max_suppression, xyxy2xywh # noqa
        from yolov5.utils.augmentations import letterbox # noqa
        from yolov5.utils.general import scale_boxes as scale_coords # noqa
        utils_imported = True
        print('Imported YOLOv5 from YOLOv5 package')
    except Exception:
        # print('YOLOv5 module import failed, falling back to path-based import')
        pass

# Next try importing from the yolov9 package
if try_yolov9_import and not utils_imported:
    
    try:
        from yolov9.utils.general import non_max_suppression, xyxy2xywh # noqa
        from yolov9.utils.augmentations import letterbox # noqa
        from yolov9.utils.general import scale_boxes as scale_coords # noqa
        utils_imported = True
        print('Imported YOLOv5 from YOLOv9 package')
    except Exception:
        # print('YOLOv5 module import failed, falling back to path-based import')
        pass


# If we haven't succeeded yet, import from the ultralytics package        
if try_ultralytics_import and not utils_imported:
    
    try:
        from ultralytics.utils.ops import non_max_suppression # noqa
        from ultralytics.utils.ops import xyxy2xywh # noqa
        
        # In the ultralytics package, scale_boxes and scale_coords both exist;
        # we want scale_boxes.
        #        
        # from ultralytics.utils.ops import scale_coords # noqa
        from ultralytics.utils.ops import scale_boxes as scale_coords # noqa
        from ultralytics.data.augment import LetterBox
        
        # letterbox() became a LetterBox class in the ultralytics package
        def letterbox(img,new_shape,stride,auto=True): # noqa
            L = LetterBox(new_shape,stride=stride,auto=auto)
            letterbox_result = L(image=img)
            return [letterbox_result]
        utils_imported = True
        print('Imported YOLOv5 from ultralytics package')
    except Exception:
        # print('Ultralytics module import failed, falling back to yolov5 import')
        pass

# If we haven't succeeded yet, assume the YOLOv5 repo is on our PYTHONPATH.
if not utils_imported:
    
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
        print('Imported YOLOv5 as utils.* from {}'.format(imported_file))
                
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('Could not import YOLOv5 functions:\n{}'.format(str(e)))

assert utils_imported, 'YOLOv5 import error'

print(f'Using PyTorch version {torch.__version__}')

    
#%% Classes

default_compatibility_mode = 'classic'

# This is a useful hack when I want to verify that my test driver (md_tests.py) is 
# correctly forcing a specific compabitility mode (I use "classic-test" in that case)
require_non_default_compatibility_mode = False

class PTDetector:
    
    def __init__(self, model_path, detector_options=None):
        
        # Parse options specific to this detector family
        force_cpu = False
        use_model_native_classes = False        
        compatibility_mode = default_compatibility_mode
                
        if detector_options is not None:
            
            if 'force_cpu' in detector_options:
                force_cpu = detector_options['force_cpu']
            if 'use_model_native_classes' in detector_options:
                use_model_native_classes = detector_options['use_model_native_classes']
            if 'compatibility_mode' in detector_options:
                if detector_options['compatibility_mode'] is None:
                    compatibility_mode = default_compatibility_mode
                else:    
                    compatibility_mode = detector_options['compatibility_mode']        
            
        if require_non_default_compatibility_mode:
            
            print('### DEBUG: requiring non-default compatibility mode ###')
            assert compatibility_mode != 'classic'
            assert compatibility_mode != 'default'
        
        print('Loading PT detector with compatibility mode {}'.format(compatibility_mode))
        
        #: Image size passed to YOLOv5's letterbox() function; 1280 means "1280 on the long side, preserving 
        #: aspect ratio".
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
        #: way this class does inference.  Currently should either be "default" or should 
        self.compatibility_mode = compatibility_mode
        
        #: Stride size passed to YOLOv5's letterbox() function
        self.letterbox_stride = 32
        
        if 'classic' in self.compatibility_mode:
            self.letterbox_stride = 64
        
        #: Use half-precision inference... fixed by the model, generally don't mess with this
        self.half_precision = False
        
        if not force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            try:
                if torch.backends.mps.is_built and torch.backends.mps.is_available():
                    self.device = 'mps'
            except AttributeError:
                pass
        try:
            self.model = PTDetector._load_model(model_path, self.device)
            
        except Exception as e:
            # In a very esoteric scenario where an old version of YOLOv5 is used to run
            # newer models, we run into an issue because the "Model" class became
            # "DetectionModel".  New YOLOv5 code handles this case by just setting them
            # to be the same, so doing that via monkey-patch doesn't seem *that* rude.
            if "Can't get attribute 'DetectionModel'" in str(e):
                print('Forward-compatibility issue detected, patching')
                from models import yolo
                yolo.DetectionModel = yolo.Model
                self.model = PTDetector._load_model(model_path, self.device)                
            else:
                raise
        if (self.device != 'cpu'):
            print('Sending model to GPU')
            self.model.to(self.device)
                    

    @staticmethod
    def _load_model(model_pt_path, device):
        
        # There are two very slightly different ways to load the model, (1) using the
        # map_location=device parameter to torch.load and (2) calling .to(device) after
        # loading the model.  The former is what we did for a zillion years, but is not
        # supported on Apple silicon at of 2029.09.  Switching to the latter causes
        # very slight changes to the output, which always make me nervous, so I'm not
        # doing a wholesale swap just yet.  Instead, we'll just do this on M1 hardware.
        use_map_location = (device != 'mps')        
        
        if use_map_location:
            checkpoint = torch.load(model_pt_path, map_location=device)
        else:
            checkpoint = torch.load(model_pt_path)
        
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

    def generate_detections_one_image(self, 
                                      img_original, 
                                      image_id='unknown', 
                                      detection_threshold=0.00001, 
                                      image_size=None,
                                      skip_image_resizing=False,
                                      augment=False):
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
                external resizing)... you almost never want ot mess with this
            augment (bool, optional): enable (implementation-specific) image augmentation

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

        if detection_threshold is None:
            
            detection_threshold = 0
            
        try:
            
            if not isinstance(img_original,np.ndarray):                
                img_original = np.asarray(img_original)

            # PIL images are RGB already
            # img_original = img_original[:, :, ::-1]
            
            # Save the original shape for scaling boxes later
            scaling_shape = img_original.shape
            
            # If the caller is requesting a specific target size...
            #
            # Image size can be an int (which translates to a square target size) or (h,w)
            if image_size is not None:
                
                assert isinstance(image_size,int) or (len(image_size)==2)
                
                if not self.printed_image_size_warning:
                    print('Warning: using user-supplied image size {}'.format(image_size))
                    self.printed_image_size_warning = True                    
            
            # Otherwise resize to self.default_image_size
            else:
                
                image_size = self.default_image_size
                self.printed_image_size_warning = False
                
            # ...if the caller has specified an image size
            
            # If the caller wants us to skip all the resizing operations...
            if skip_image_resizing:
                
                img = img_original
                
            # Otherwise we have a bunch of resizing to do...
            else:
                            
                # In "classic mode", we only do the letterboxing resize, we don't do an
                # additional initial resizing operation
                if 'classic' in self.compatibility_mode:
                    
                    resize_ratio = 1.0
                    
                # Resize the image so the long side matches the target image size.  This is not 
                # letterboxing (i.e., padding) yet, just resizing.
                else:
                    
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
                        img_original = cv2.resize(
                            img_original, (math.ceil(w * resize_ratio), 
                                           math.ceil(h * resize_ratio)),
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
                    
                # Now we letterbox...
                img,letterbox_ratio,letterbox_pad = letterbox(img_original, 
                                                              new_shape=target_shape,
                                                              stride=self.letterbox_stride, 
                                                              auto=letterbox_auto,
                                                              scaleFill=False,
                                                              scaleup=letterbox_scaleup)
            
            # HWC to CHW; PIL Image is RGB already
            img = img.transpose((2, 0, 1)) # [::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)
            img = img.half() if self.half_precision else img.float()
            img /= 255

            # In practice this is always true 
            if len(img.shape) == 3:  
                img = torch.unsqueeze(img, 0)

            pred = self.model(img,augment=augment)[0]

            if 'classic' in self.compatibility_mode:
                nms_conf_thres = detection_threshold
                nms_iou_thres = 0.45
                nms_agnostic = False
                nms_multi_label = False
            else:
                nms_conf_thres = 0.01
                nms_iou_thres = 0.6
                nms_agnostic = False    
                # yolov5 sets this to True, but this has some infrastructural implications
                # that I don't want to deal with.  This is not really a numerical issue.
                nms_multi_label = False 
                
            # As of PyTorch 1.13.0.dev20220824, nms is not implemented for MPS.
            #
            # Send predictions back to the CPU for NMS.            
            if self.device == 'mps':
                pred_nms = pred.cpu()
            else:
                pred_nms = pred
                
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
                
                # letterbox_pad is a 2-tuple specifying the padding that was added on each axis
                # ratio is a 2-tuple specifying the scaling that was applied to each dimension
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
                        # MegaDetector output format's categories start at 1, but the MD 
                        # model's categories start at 0.
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
    
    import os
    from megadetector.visualization import visualization_utils as vis_utils
    
    model_file = os.environ['MDV5A']
    im_file = os.path.expanduser('~/git/MegaDetector/images/nacti.jpg')

    detector = PTDetector(model_file)
    image = vis_utils.load_image(im_file)

    res = detector.generate_detections_one_image(image, im_file, detection_threshold=0.00001)
    print(res)
