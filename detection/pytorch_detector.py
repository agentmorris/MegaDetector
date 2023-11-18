########
#
# pytorch_detector.py
#
# Module to run MegaDetector v5, a PyTorch YOLOv5 animal detection model.
#
########

#%% Imports

import torch
import numpy as np
import traceback

from detection.run_detector import CONF_DIGITS, COORD_DIGITS, FAILURE_INFER
from md_utils import ct_utils

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

# See above; this should remain as "False" unless we update the MegaDetector .pt file
# to use more recent YOLOv5 namespace conventions.
try_ultralytics_import = False

# First try importing from the yolov5 package
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

# If we haven't succeeded yet, import from the ultralytics package        
if try_ultralytics_import and not utils_imported:
    
    try:
        from ultralytics.utils.ops import non_max_suppression # noqa
        from ultralytics.utils.ops import xyxy2xywh # noqa
        from ultralytics.utils.ops import scale_coords # noqa
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

# If we haven't succeeded yet, import from the YOLOv5 repo
if not utils_imported:
    
    try:
        # import pre- and post-processing functions from the YOLOv5 repo
        from utils.general import non_max_suppression, xyxy2xywh # noqa
        from utils.augmentations import letterbox # noqa
        
        # scale_coords() became scale_boxes() in later YOLOv5 versions
        try:
            from utils.general import scale_coords # noqa
        except ImportError:
            from utils.general import scale_boxes as scale_coords
        utils_imported = True
        print('Imported YOLOv5 as utils.*')
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Could not import YOLOv5 functions.')

assert utils_imported, 'YOLOv5 import error'

print(f'Using PyTorch version {torch.__version__}')

    
#%% Classes

class PTDetector:

    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64

    def __init__(self, model_path: str, 
                 force_cpu: bool = False,
                 use_model_native_classes: bool = False):
        self.device = 'cpu'
        if not force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            try:
                if torch.backends.mps.is_built and torch.backends.mps.is_available():
                    self.device = 'mps'
            except AttributeError:
                pass
        self.model = PTDetector._load_model(model_path, self.device)
        if (self.device != 'cpu'):
            print('Sending model to GPU')
            self.model.to(self.device)
            
        self.printed_image_size_warning = False        
        self.use_model_native_classes = use_model_native_classes
        

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

    def generate_detections_one_image(self, img_original, image_id='unknown', 
                                      detection_threshold=0.00001, image_size=None,
                                      skip_image_resizing=False):
        """
        Apply the detector to an image.

        Args:
            img_original: the PIL Image object with EXIF rotation taken into account
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal
            skip_image_resizing: whether to skip internal image resizing and rely on external resizing

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/agentmorris/MegaDetector/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf' (removed from MegaDetector output by default, but generated here)
            - 'detections', which is a list of detection objects containing keys 'category', 
              'conf' and 'bbox'
            - 'failure'
        """

        result = {
            'file': image_id
        }
        detections = []
        max_conf = 0.0

        try:
            
            img_original = np.asarray(img_original)

            # padded resize
            target_size = PTDetector.IMAGE_SIZE
            
            # Image size can be an int (which translates to a square target size) or (h,w)
            if image_size is not None:
                
                assert isinstance(image_size,int) or (len(image_size)==2)
                
                if not self.printed_image_size_warning:
                    print('Warning: using user-supplied image size {}'.format(image_size))
                    self.printed_image_size_warning = True
            
                target_size = image_size
            
            else:
                
                self.printed_image_size_warning = False
                
            # ...if the caller has specified an image size
            
            if skip_image_resizing:
                img = img_original
            else:
                letterbox_result = letterbox(img_original, new_shape=target_size,
                                stride=PTDetector.STRIDE, auto=True)
                img = letterbox_result[0]                
            
            # HWC to CHW; PIL Image is RGB already
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)
            img = img.float()
            img /= 255

            # In practice this is always true 
            if len(img.shape) == 3:  
                img = torch.unsqueeze(img, 0)

            pred: list = self.model(img)[0]

            # NMS
            if self.device == 'mps':
                # As of v1.13.0.dev20220824, nms is not implemented for MPS.
                #
                # Send predication back to the CPU to fix.
                pred = non_max_suppression(prediction=pred.cpu(), conf_thres=detection_threshold)
            else: 
                pred = non_max_suppression(prediction=pred, conf_thres=detection_threshold)

            # format detections/bounding boxes
            #
            # normalization gain whwh
            gn = torch.tensor(img_original.shape)[[1, 0, 1, 0]]

            # This is a loop over detection batches, which will always be length 1 in our case,
            # since we're not doing batch inference.
            for det in pred:
                
                if len(det):
                    
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        
                        # normalized center-x, center-y, width and height
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        api_box = ct_utils.convert_yolo_to_xywh(xywh)

                        conf = ct_utils.truncate_float(conf.tolist(), precision=CONF_DIGITS)

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
                            'bbox': ct_utils.truncate_float_array(api_box, precision=COORD_DIGITS)
                        })
                        max_conf = max(max_conf, conf)
                        
                    # ...for each detection in this batch
                        
                # ...if this is a non-empty batch
                
            # ...for each detection batch

        # ...try
        
        except Exception as e:
            
            result['failure'] = FAILURE_INFER
            print('PTDetector: image {} failed during inference: {}\n'.format(image_id, str(e)))
            traceback.print_exc(e)

        result['max_detection_conf'] = max_conf
        result['detections'] = detections

        return result


#%% Command-line driver

if __name__ == '__main__':
    
    # For testing only... you don't really want to run this module directly

    #%%
    
    import md_visualization.visualization_utils as vis_utils
    import os
    
    model_file = 'MDV5A'
    im_file = os.path.expanduser('~/git/MegaDetector/images/nacti.jpg')

    detector = PTDetector(model_file)
    image = vis_utils.load_image(im_file)

    res = detector.generate_detections_one_image(image, im_file, detection_threshold=0.00001)
    print(res)
