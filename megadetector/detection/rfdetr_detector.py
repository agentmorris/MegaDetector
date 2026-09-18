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

# Maps the dtype strings we accept in detector options to the corresponding torch dtypes
dtype_string_to_torch_dtype = {
    'float16': torch.float16,
    'float32': torch.float32
}

#: Whether to allow TF32 for RF-DETR model loading and inference by default.
#:
#: TF32 is a reduced-precision format that PyTorch uses automatically on recent NVIDIA GPUs.
#: It's around 10% faster, but it makes RF-DETR's results depend on the batch size.  In
#: practice, running the same image alone vs. in a batch of 4 shifts confidence values by up
#: to ~0.05.
#:
#: Note that this is not something we can leave to the defaults: PyTorch enables TF32 for
#: convolutions by default, and importing the rfdetr package additionally enables it for
#: matrix multiplications, for the whole process.
DEFAULT_USE_TF32 = False


class TF32ExecutionContext:
    """
    Context manager that applies PyTorch's TF32 settings for the duration of a block, then
    restores the values that were in effect when the block was entered.

    TF32 settings are global to a PyTorch process, so we apply them only around RF-DETR
    model loading and inference.
    """

    def __init__(self, use_tf32):
        """
        Initializes TF32ExecutionContext.

        Args:
            use_tf32 (bool): whether TF32 should be enabled within this context
        """

        #: Whether TF32 should be enabled within this context
        self.use_tf32 = use_tf32

        #: The value of torch.get_float32_matmul_precision() when this context was entered
        self.previous_matmul_precision = None

        #: The value of torch.backends.cudnn.allow_tf32 when this context was entered
        self.previous_cudnn_allow_tf32 = None

    def apply_settings(self):
        """
        Applies this context's TF32 settings.  Called automatically on entry; also called
        directly after importing rfdetr, which changes these settings as a side effect.
        """

        # "high" allows TF32 for matmuls, "highest" forces full fp32
        torch.set_float32_matmul_precision('high' if self.use_tf32 else 'highest')
        torch.backends.cudnn.allow_tf32 = self.use_tf32

    def __enter__(self):
        """
        Stores the current TF32 settings, then applies this context's settings.
        """

        self.previous_matmul_precision = torch.get_float32_matmul_precision()
        self.previous_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

        self.apply_settings()

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Restores the TF32 settings that were in effect when this context was entered.
        """

        torch.set_float32_matmul_precision(self.previous_matmul_precision)
        torch.backends.cudnn.allow_tf32 = self.previous_cudnn_allow_tf32

        # Don't suppress exceptions
        return False

# ...class TF32ExecutionContext


#%% Model loading

def load_model(detector_file,
               image_size=None,
               optimize_for_inference=False,
               batch_size=1,
               compile=None,
               dtype=None,
               use_tf32=DEFAULT_USE_TF32):
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
        batch_size (int, optional): batch size to pass to optimize_for_inference().  This
            only matters when [optimize_for_inference] is True *and* compilation is enabled;
            a compiled model can only be run at the batch size it was compiled for.  Ignored
            if [optimize_for_inference] is False.
        compile (bool, optional): whether optimize_for_inference() should compile the model
            (via torch.jit.trace).  None means "use the rfdetr default", which is currently
            True.  Compilation ties the model to a single batch size.  Ignored if
            [optimize_for_inference] is False.
        dtype (str, optional): floating-point dtype used for inference, either "float16" or
            "float32".  None means "use the rfdetr default", which is currently float32.  Ignored
            if [optimize_for_inference] is False.
        use_tf32 (bool, optional): whether to allow reduced-precision TF32 computations.
            Enabling TF32 is around 10% faster, but makes results depend on the batch size;
            see DEFAULT_USE_TF32.

    Returns:
        dict: dictionary with keys:
            - 'model': the loaded RF-DETR model
            - 'model_type' (str): model type name (e.g. 'RFDETRSmall')
            - 'image_size' (int): inference resolution
            - 'detection_categories' (dict): mapping from string category IDs to class names
    """

    if dtype is not None:
        assert dtype in dtype_string_to_torch_dtype, \
            'Illegal dtype {}, dtype should be one of: {}'.format(
                dtype,', '.join(dtype_string_to_torch_dtype.keys()))

    # Everything from the rfdetr import through model construction runs inside a
    # TF32ExecutionContext, for two reasons.  First, importing rfdetr enables TF32 matmuls
    # for the whole process, which would otherwise silently change the numerics of any other
    # model running in this process; entering the context around the import means we put that
    # setting back the way we found it on the way out.  Second, optimize_for_inference() may
    # trace/compile the model, which bakes in whatever precision is active at that time.
    with TF32ExecutionContext(use_tf32) as tf32_context:

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

        # Importing rfdetr changes the TF32 settings, so re-apply ours
        tf32_context.apply_settings()

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

            optimize_kwargs = {'batch_size':batch_size}

            # Leaving [compile] or [dtype] set to None means "use the rfdetr defaults", which
            # are currently True and float32, respectively.
            if compile is not None:
                optimize_kwargs['compile'] = compile
            if dtype is not None:
                optimize_kwargs['dtype'] = dtype_string_to_torch_dtype[dtype]

            print('Optimizing loaded model for inference (batch size {}, compile {}, dtype {})'.format(
                batch_size,str(compile),dtype))
            model.optimize_for_inference(**optimize_kwargs)

            # optimize_for_inference is off by default because it reportedly created
            # inference errors in some environments.  This comment suggests that specifying
            # dtype=bfloat16 allows us to have our cake and eat it too, but this hasn't
            # been tested.
            #
            # https://github.com/roboflow/rf-detr/issues/326#issuecomment-3321838797
            # model.optimize_for_inference(batch_size=batch_size,dtype=torch.bfloat16)

        elif (compile is not None) or (dtype is not None):

            print('Warning: the "compile" and/or "dtype" options were supplied, but ' + \
                  'optimize_for_inference is False, so they will have no effect.')

    # ...with TF32ExecutionContext(...)

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
            detector_options (dict, optional): dictionary of RFDETr-specific detector options,
                see load_model for documentation of available options.
            verbose (bool, optional): enable additional debug output
        """

        if verbose:
            print('Initializing RFDETRDetector')

        # Parse options specific to this detector family
        image_size = None
        optimize_for_inference = False
        batch_size = 1
        compile = None
        dtype = None
        use_tf32 = DEFAULT_USE_TF32

        if detector_options is not None:
            if ('image_size' in detector_options) and \
                (detector_options['image_size'] is not None):
                image_size = int(detector_options['image_size'])
            if ('optimize_for_inference' in detector_options) and \
                (detector_options['optimize_for_inference'] is not None):
                optimize_for_inference = parse_bool_string(detector_options['optimize_for_inference'])
            if ('batch_size' in detector_options) and \
                (detector_options['batch_size'] is not None):
                batch_size = int(detector_options['batch_size'])
            if ('compile' in detector_options) and \
                (detector_options['compile'] is not None):
                compile = parse_bool_string(detector_options['compile'])
            if ('dtype' in detector_options) and \
                (detector_options['dtype'] is not None):
                dtype = detector_options['dtype']
                assert dtype in dtype_string_to_torch_dtype, \
                    'Illegal dtype {}, dtype should be one of: {}'.format(
                        dtype,', '.join(dtype_string_to_torch_dtype.keys()))
            if ('use_tf32' in detector_options) and \
                (detector_options['use_tf32'] is not None):
                use_tf32 = parse_bool_string(detector_options['use_tf32'])

        # If the caller asked for inference optimization, but didn't say anything about
        # compilation, don't compile.  Compiling (torch.jit.trace) restricts the model to a
        # single batch size, and in practice buys very little compared to running at a
        # smaller dtype, so we don't opt into it implicitly.  Note that this differs from
        # the rfdetr default, which is to compile.
        if optimize_for_inference and (compile is None):
            compile = False

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

        #: The exact batch size this model requires at inference time, or None if any batch
        #: size is allowed.  This is only set when the model has been compiled, since
        #: compilation (torch.jit.trace) ties the model to a single batch size.
        self.required_batch_size = None

        #: Whether TF32 is allowed during inference for this model; see DEFAULT_USE_TF32
        self.use_tf32 = use_tf32

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
                                optimize_for_inference=optimize_for_inference,
                                batch_size=batch_size,
                                compile=compile,
                                dtype=dtype,
                                use_tf32=use_tf32)

        self.model = model_info['model']
        self.model_type = model_info['model_type']
        self.image_size = model_info['image_size']
        self.detection_categories = model_info['detection_categories']

        # A compiled model can only be run at the batch size it was compiled for, so record
        # that batch size; generate_detections_one_batch() pads short batches accordingly.
        # [compile] can no longer be None at this point; we resolved None to False above.
        if optimize_for_inference and compile:
            self.required_batch_size = batch_size

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
                PTDetector.preprocess_image().  RF-DETR resolution is fixed at load time, so
                this must be either None or the resolution this model was loaded at; any other
                value raises a ValueError.
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: dict with fields:
                - file (filename)
                - img_original (the input image as an np.array)
                - img_original_pil (the input image as a PIL Image, or None if a numpy array
                  was supplied)
        """

        if (image_size is not None) and (image_size != self.image_size):
            raise ValueError(
                'RF-DETR model initialized for image size {}, but image size {} requested'.format(
                    self.image_size,image_size))

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
            image_size (int, optional): included for signature compatibility with PTDetector.
                RF-DETR resolution is fixed at load time (set it via the 'image_size' detector
                option), so this must be either None or the resolution this model was loaded at;
                any other value raises a ValueError.
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

        if (image_size is not None) and (image_size != self.image_size):
            raise ValueError(
                'RF-DETR model initialized for image size {}, but image size {} requested'.format(
                    self.image_size,image_size))

        if augment:
            raise NotImplementedError('augmentation is not supported for RF-DETR models')

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

        # A compiled model can only be run at the batch size it was compiled for, so we pad
        # short batches with dummy images, and discard the dummy results after inference.
        # This happens after we've assembled [image_ids] and [image_shapes], so those lists
        # still refer only to the real images.
        n_images = len(images_for_inference)
        n_padding_images = 0

        if self.required_batch_size is not None:

            assert n_images <= self.required_batch_size, \
                'This model was compiled for a batch size of {}, but a batch of {} '.format(
                    self.required_batch_size,n_images) + \
                'images was supplied; batches larger than the compiled batch size are ' + \
                'not supported'

            n_padding_images = self.required_batch_size - n_images

            if (n_padding_images > 0) and verbose:
                print('Padding a batch of {} images out to the compiled batch size of {}'.format(
                    n_images,self.required_batch_size))

            for _ in range(n_padding_images):
                images_for_inference.append(
                    np.zeros((self.image_size,self.image_size,3),dtype=np.uint8))

        # ...if this model requires a specific batch size

        # Run inference.  model.predict() returns a single Detections object for a single
        # image, or a list of Detections objects for a list of images.
        #
        # We apply our TF32 settings for the duration of inference; leaving TF32 enabled
        # makes results depend on the batch size (see DEFAULT_USE_TF32).
        try:
            with TF32ExecutionContext(self.use_tf32):
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
            return [{'file': image_ids[i_img],
                     'detections': None,
                     'failure': FAILURE_INFER + ': ' + str(e)}
                    for i_img in range(len(image_ids))]

        assert len(detections_list) == len(images_for_inference), \
            'Mismatch between prediction length {} and batch size {}'.format(
                len(detections_list),len(images_for_inference))

        # Discard the results for any dummy images we added as padding
        if n_padding_images > 0:
            detections_list = detections_list[:n_images]

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
            image_size (int, optional): RF-DETR resolution is specified at load time (via the
                'image_size' detector option), not inference time, so this must be either None or
                the resolution this model was loaded at; any other value raises a ValueError.
            augment (bool, optional): must be False for RF-DETR models (which don't support augmentation)
            verbose (bool, optional): enable additional debug output

        Returns:
            dict: a dictionary with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, only present if inference failed)
        """

        if (image_size is not None) and (image_size != self.image_size):
            raise ValueError(
                'RF-DETR model initialized for image size {}, but image size {} requested'.format(
                    self.image_size,image_size))

        if augment:
            raise NotImplementedError('augmentation is not supported for RF-DETR models')

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
