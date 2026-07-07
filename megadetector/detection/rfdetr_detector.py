"""

rfdetr_detector.py

Module to run RF-DETR-based detectors within the MegaDetector Python package.

Supports only RF-DETR checkpoints produced by package version >= 1.8.3, which include
metadata about model architecture and training resolution that was not included in
earlier checkpoint formats.

The rfdetr package is not a dependency of the MegaDetector Python package, so it is
imported lazily (at the time a model is loaded), rather than at module import time.

"""

#%% Imports and constants

import torch
import numpy as np

from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS, FAILURE_INFER
from megadetector.utils.ct_utils import round_float, round_float_array
from megadetector.utils.ct_utils import parse_bool_string


#%% Model loading

def load_model(detector_file,
               image_size=None,
               optimize_for_inference=False,
               batch_size=1):
    """
    Load an RF-DETR model from an inference-ready .pth checkpoint via
    rfdetr.from_checkpoint(), which reads the architecture name ("Nano",
    "Medium", etc.), training resolution, and class names from metadata stored
    in the checkpoint.

    Args:
        detector_file (str): path to .pth checkpoint file.
        image_size (int, optional): image resolution for inference.  None uses the
            training resolution recorded in the checkpoint; a value overrides it.
        optimize_for_inference (bool, optional): whether to optimize the model for
            inference, which should be a free lunch, but as of 9/2025 there is some
            risk of accuracy regression.
        batch_size (int, optional): batch size to pass to optimize_for_inference()

    Returns:
        dict: dictionary with keys:
            - 'model': the loaded RF-DETR model
            - 'model_type' (str): resolved variant class name (e.g. 'RFDETRSmall')
            - 'image_size' (int): resolved inference resolution
            - 'detection_categories' (dict): mapping from string category IDs to class names
    """

    # The rfdetr package is not installed by default with the MegaDetector package,
    # so we import it here (rather than at module scope) and print a friendly warning
    # if it's not available.
    try:
        import rfdetr
    except Exception:
        print('\n\n*****\nIt looks like you are trying to run an RF-DETR model with the '
              'MegaDetector Python package.  This is supported, but the rfdetr package is not '
              'installed by default.  Run "pip install rfdetr" to install it, and try again.'
              '\n*****\n\n')
        raise

    assert detector_file.lower().endswith('.pth'), \
        '{} does not appear to be a compatible RF-DETR checkpoint'.format(detector_file)

    # This module uses rfdetr.from_checkpoint(), which relies on a 'model_config' field
    # that was not present in checkpoints produced by early RF-DETR library versions.
    print('Reading checkpoint metadata from: {}'.format(detector_file))
    checkpoint = torch.load(detector_file, weights_only=False, map_location='cpu')
    if 'model_config' not in checkpoint:
        raise ValueError(
            "Model file '{}' is in an older format that this inference ".format(detector_file) + \
            "code does not support (missing 'model_config' metadata).")
    del checkpoint

    # Load the model, letting from_checkpoint() resolve the model type and resolution.
    #
    # A caller-supplied image_size overrides the loaded resolution.
    from_checkpoint_kwargs = {}
    if image_size is not None:
        from_checkpoint_kwargs['resolution'] = image_size
    print('Loading model from {}...'.format(detector_file))
    model = rfdetr.from_checkpoint(detector_file, **from_checkpoint_kwargs)

    model_type = type(model).__name__
    image_size = model.model_config.resolution
    print('Loaded {} at resolution {}'.format(model_type, image_size))

    if optimize_for_inference:

        print('Optimizing loaded model for inference')
        model.optimize_for_inference(batch_size=batch_size)

        # optimize_for_inference is off by default because it reportedly created
        # inference errors in some environments.  This comment suggests that specifying
        # dtype=bfloat16 allows us to have our cake and eat it too, but this hasn't
        # been tested.
        #
        # https://github.com/roboflow/rf-detr/issues/326#issuecomment-3321838797
        # model.optimize_for_inference(batch_size=batch_size,dtype=torch.bfloat16)

    # Get class names from model
    #
    # model.class_names is a list of strings.  Note to self: in older rfdetr versions, it was
    # a dict mapping 1-indexed class IDs to names.
    class_names = model.class_names
    print('Class names: {}'.format(class_names))

    # Build detection_categories dict
    detection_categories = {}
    for i_class,class_name in enumerate(class_names):
        detection_categories[str(i_class)] = class_name

    return \
    {
        'model': model,
        'model_type': model_type,
        'image_size': image_size,
        'detection_categories': detection_categories
    }

# ...def load_model(...)


#%% Detection format conversion

def convert_detections_to_md_format(detections, image_width, image_height):
    """
    Convert RF-DETR/Supervision detections to MegaDetector format.

    Args:
        detections: supervision Detections object with xyxy, confidence, class_id
        image_width (int): image width in pixels
        image_height (int): image height in pixels

    Returns:
        list: list of detection dicts in MegaDetector format
    """

    md_detections = []

    if (detections is None) or (len(detections) == 0):
        return md_detections

    for i_detection in range(len(detections)):

        # Extract xyxy coordinates (absolute pixels)
        x1, y1, x2, y2 = detections.xyxy[i_detection]

        # Convert to normalized xywh format
        x_min_norm = float(x1) / image_width
        y_min_norm = float(y1) / image_height
        width_norm = float(x2 - x1) / image_width
        height_norm = float(y2 - y1) / image_height

        # Clamp values to [0, 1] range
        x_min_norm = max(0.0, min(1.0, x_min_norm))
        y_min_norm = max(0.0, min(1.0, y_min_norm))
        width_norm = max(0.0, min(1.0 - x_min_norm, width_norm))
        height_norm = max(0.0, min(1.0 - y_min_norm, height_norm))

        # Get confidence and class_id
        conf = float(detections.confidence[i_detection])

        # RF-DETR class_ids are 0-indexed when returned from the API
        class_id = int(detections.class_id[i_detection])

        category = str(class_id)

        bbox = round_float_array([x_min_norm, y_min_norm, width_norm, height_norm],
                                 precision=COORD_DIGITS)
        conf = round_float(conf, precision=CONF_DIGITS)

        md_detections.append({
            'category': category,
            'conf': conf,
            'bbox': bbox
        })

    # ...for each detection

    return md_detections

# ...def convert_detections_to_md_format(...)


#%% Inference class

class RFDETRDetector:
    """
    Class that runs an RF-DETR-based detector.  Also used as a preprocessor for images
    that will later be run through an instance of RFDETRDetector.
    """

    def __init__(self, model_path, detector_options=None, verbose=False):
        """
        RFDETRDetector constructor.  If detector_options['preprocess_only'] exists and is
        True, this instance is being used as a preprocessor, so we don't load model weights.

        Args:
            model_path (str): path to the .pth model file to load
            detector_options (dict, optional): dictionary of detector options that mean
                different things to different models
            verbose (bool, optional): enable additional debug output
        """

        if verbose:
            print('Initializing RFDETRDetector')

        # Parse options specific to this detector family
        image_size = None
        optimize_for_inference = False

        if detector_options is not None:
            if ('image_size' in detector_options) and \
                (detector_options['image_size'] is not None):
                image_size = int(detector_options['image_size'])
            if ('optimize_for_inference' in detector_options) and \
                (detector_options['optimize_for_inference'] is not None):
                optimize_for_inference = parse_bool_string(detector_options['optimize_for_inference'])

        #: Image resolution passed to from_checkpoint(); None means "use the resolution
        #: recorded in the checkpoint".  After the model is loaded, this is updated to the
        #: resolution actually used.
        self.image_size = image_size

        #: The loaded RF-DETR model; remains None for preprocess-only instances
        self.model = None

        #: The resolved variant class name (e.g. 'RFDETRNano'); None until the model is loaded
        self.model_type = None

        #: Mapping from string category IDs to class names; None until the model is loaded
        self.detection_categories = None

        preprocess_only = False
        if (detector_options is not None) and \
           ('preprocess_only' in detector_options) and \
           (detector_options['preprocess_only']):
            preprocess_only = True

        # If this instance is only going to be used for preprocessing, don't load weights
        if preprocess_only:
            if verbose:
                print('Created RFDETRDetector in preprocess-only mode')
            return

        # Load the model
        model_info = load_model(model_path,
                                image_size=self.image_size,
                                optimize_for_inference=optimize_for_inference)
        self.model = model_info['model']
        self.model_type = model_info['model_type']
        self.image_size = model_info['image_size']
        self.detection_categories = model_info['detection_categories']

    # ...def __init__(...)


    def preprocess_image(self,
                         img_original,
                         image_id='unknown',
                         image_size=None,
                         verbose=False):
        """
        Prepare an image for detection.  RF-DETR resizes and letterboxes internally, so
        this is almost a no-op.

        Args:
            img_original (Image or np.array): the image on which we should run the detector, with
                EXIF rotation already handled
            image_id (str, optional): a path to identify the image; will be in the "file" field
                of the output object
            image_size (int, optional): included for signature compatibility with
                PTDetector.preprocess_image(); ignored for RF-DETR models
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: dict with fields:
                - file (filename)
                - img_original (the input image as an np.array)
                - img_original_pil (the input image as a PIL Image, or None if a numpy array
                  was supplied)
        """

        result = {'file': image_id}

        # Store the PIL version of the original image; the caller may want to use it later
        # (e.g. for metadata extraction).  This mirrors PTDetector.preprocess_image(): it
        # remains None unless a PIL image (i.e., something that isn't already a numpy array)
        # was supplied.
        img_original_pil = None

        # If we were given a PIL image (or anything else that isn't already a numpy array),
        # convert it to a numpy array.
        if isinstance(img_original, np.ndarray):
            result['img_original'] = img_original
        else:
            img_original_pil = img_original
            result['img_original'] = np.asarray(img_original)

        result['img_original_pil'] = img_original_pil

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
        Run an RF-DETR detector on a batch of images.

        Args:
            img_original (list): list of images (Image, np.array, or dict) on which we should run
                the detector, with EXIF rotation already handled, or dicts representing preprocessed
                images (as produced by preprocess_image())
            image_id (list or None): list of paths to identify the images; will be in the "file" field
                of the output objects.  Ignored when img_original contains preprocessed dicts.
            detection_threshold (float, optional): only detections above this confidence threshold
                will be included in the return value
            image_size (int, optional): included for signature compatibility with PTDetector; must
                be None for RF-DETR models (set the resolution via the 'image_size' detector option
                at load time instead)
            augment (bool, optional): included for signature compatibility with PTDetector; must be
                False for RF-DETR models
            verbose (bool, optional): enable additional debug output

        Returns:
            list: a list of dictionaries, each with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, only present if inference failed)
        """

        # These parameters exist only for signature compatibility with other detectors; RF-DETR
        # handles resizing internally and does not support augmentation.
        assert image_size is None, \
            'image_size is not supported as an inference-time call for RF-DETR models; set the resolution ' + \
            'via the "image_size" detector option at load time instead'
        assert not augment, 'augmentation is not supported for RF-DETR models'

        # Validate inputs
        if not isinstance(img_original, list):
            raise ValueError('img_original must be a list for batch processing')

        if len(img_original) == 0:
            return []

        # Verify input consistency
        if isinstance(img_original[0], dict):
            # All items in img_original should be preprocessed dicts
            for i_img, img in enumerate(img_original):
                if not isinstance(img, dict):
                    raise ValueError(
                        'Mixed input types in batch: item {} is not a dict, but item 0 is a dict'.format(
                            i_img))
        else:
            # All items in img_original should be PIL/numpy images, and image_id should be a list
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

        # Assemble the list of images to run inference on, along with their IDs and sizes
        images_for_inference = []
        image_ids = []

        # (width,height) for each image, used to normalize box coordinates later
        image_shapes = []

        for i_img, img in enumerate(img_original):

            if isinstance(img, dict):
                current_image_id = img['file']
                image_np = img['img_original']
            else:
                current_image_id = image_id[i_img]
                image_np = img
                if not isinstance(image_np, np.ndarray):
                    image_np = np.asarray(image_np)

            images_for_inference.append(image_np)
            image_ids.append(current_image_id)

            # numpy images are stored as (height,width,channels)
            image_height = image_np.shape[0]
            image_width = image_np.shape[1]
            image_shapes.append((image_width, image_height))

        # ...for each image in this batch

        # Run inference.  model.predict() returns a single Detections object for a single
        # image, or a list of Detections objects for a list of images.
        try:
            if len(images_for_inference) == 1:
                detections_list = [self.model.predict(images_for_inference[0],
                                                      threshold=detection_threshold)]
            else:
                detections_list = self.model.predict(images_for_inference,
                                                     threshold=detection_threshold)
        except Exception as e:
            # If inference fails, mark all images in the batch as failed
            print('Warning: RF-DETR batch inference failed for {} images: {}'.format(
                len(images_for_inference),str(e)))
            return [{'file': image_ids[i_img], 'detections': None, 'failure': FAILURE_INFER}
                    for i_img in range(len(image_ids))]

        assert len(detections_list) == len(images_for_inference), \
            'Mismatch between prediction length {} and batch size {}'.format(
                len(detections_list),len(images_for_inference))

        # Format the outputs to follow MD package conventions
        results = []

        for i_img, detections in enumerate(detections_list):

            image_width, image_height = image_shapes[i_img]

            md_detections = convert_detections_to_md_format(detections, image_width, image_height)

            max_conf = 0.0
            for det in md_detections:
                max_conf = max(max_conf, det['conf'])

            results.append({
                'file': image_ids[i_img],
                'detections': md_detections,
                'max_detection_conf': max_conf
            })

        # ...for each image in this batch

        return results

    # ...def generate_detections_one_batch(...)


    def generate_detections_one_image(self,
                                      img_original,
                                      image_id='unknown',
                                      detection_threshold=0.00001,
                                      image_size=None,
                                      augment=False,
                                      verbose=False):
        """
        Run an RF-DETR detector on an image (wrapper around generate_detections_one_batch()).

        Args:
            img_original (Image, np.array, or dict): the image on which we should run the detector,
                with EXIF rotation already handled, or a dict representing a preprocessed image (as
                produced by preprocess_image())
            image_id (str, optional): a path to identify the image; will be in the "file" field
                of the output object
            detection_threshold (float, optional): only detections above this confidence threshold
                will be included in the return value
            image_size (int, optional): must be None for RF-DETR models (for which image size is specified
                at load time, not inference time)
            augment (bool, optional): must be False for RF-DETR models (which don't support augmentation)
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: a dictionary with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, only present if inference failed)
        """

        # These parameters exist only for signature compatibility with PTDetector
        assert image_size is None, \
            'image_size is not supported as an inference-time call for RF-DETR models; set the resolution ' + \
            'via the "image_size" detector option at load time instead'
        assert not augment, 'augmentation is not supported for RF-DETR models'

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

# ...class RFDETRDetector
