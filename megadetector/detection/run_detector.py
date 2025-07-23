"""

run_detector.py

Module to run an animal detection model on images.  The main function in this script also renders
the predicted bounding boxes on images and saves the resulting images (with bounding boxes).

**This script is not a good way to process lots of images**.  It does not produce a useful
output format, and it does not facilitate checkpointing the results so if it crashes you
would have to start from scratch. **If you want to run a detector on lots of images, you should
check out run_detector_batch.py**.

That said, this script (run_detector.py) is a good way to test our detector on a handful of images
and get super-satisfying, graphical results.

If you would like to *not* use the GPU on the machine, set the environment
variable CUDA_VISIBLE_DEVICES to "-1".

This script will only consider detections with > 0.005 confidence at all times.
The threshold you provide is only for rendering the results. If you need to
see lower-confidence detections, you can change DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD.

"""

#%% Constants, imports, environment

import argparse
import os
import statistics
import sys
import time
import json
import warnings
import tempfile
import zipfile

import humanfriendly
from tqdm import tqdm

from megadetector.utils import path_utils as path_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.utils.url_utils import download_url
from megadetector.utils.ct_utils import parse_kvp_list
from megadetector.utils.path_utils import compute_file_hash

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# String constants used for consistent reporting of processing errors
FAILURE_INFER = 'inference failure'
FAILURE_IMAGE_OPEN = 'image access failure'

# Number of decimal places to round to for confidence and bbox coordinates
CONF_DIGITS = 3
COORD_DIGITS = 4

# Label mapping for MegaDetector
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'  # available in megadetector v4+
}

# Should we allow classes that don't look anything like the MegaDetector classes?
#
# This flag needs to get set if you want to, for example, run an off-the-shelf
# YOLO model with this package.
#
# By default, we error if we see unfamiliar classes.
#
# TODO: the use of a global variable to manage this was fine when this was really
# experimental, but this is really sloppy now that we actually use this code for
# models other than MegaDetector.
USE_MODEL_NATIVE_CLASSES = False

# Maps a variety of strings that might occur in filenames to canonical version numbers.
#
# Order matters here.
model_string_to_model_version = {

    # Specific model versions that might be expressed in a variety of ways
    'mdv2':'v2.0.0',
    'mdv3':'v3.0.0',
    'mdv4':'v4.1.0',
    'mdv5a':'v5a.0.1',
    'mdv5b':'v5b.0.1',

    'v2':'v2.0.0',
    'v3':'v3.0.0',
    'v4':'v4.1.0',
    'v4.1':'v4.1.0',
    'v5a.0.0':'v5a.0.1',
    'v5b.0.0':'v5b.0.1',
    'v5a.0.1':'v5a.0.1',
    'v5b.0.1':'v5b.0.1',

    'md1000-redwood':'v1000.0.0-redwood',
    'md1000-cedar':'v1000.0.0-cedar',
    'md1000-larch':'v1000.0.0-larch',
    'md1000-sorrel':'v1000.0.0-sorrel',
    'md1000-spruce':'v1000.0.0-spruce',

    'mdv1000-redwood':'v1000.0.0-redwood',
    'mdv1000-cedar':'v1000.0.0-cedar',
    'mdv1000-larch':'v1000.0.0-larch',
    'mdv1000-sorrel':'v1000.0.0-sorrel',
    'mdv1000-spruce':'v1000.0.0-spruce',

    'v1000-redwood':'v1000.0.0-redwood',
    'v1000-cedar':'v1000.0.0-cedar',
    'v1000-larch':'v1000.0.0-larch',
    'v1000-sorrel':'v1000.0.0-sorrel',
    'v1000-spruce':'v1000.0.0-spruce',

    # Arguably less specific model versions
    'redwood':'v1000.0.0-redwood',
    'spruce':'v1000.0.0-spruce',
    'cedar':'v1000.0.0-cedar',
    'larch':'v1000.0.0-larch',

    # Opinionated defaults
    'mdv5':'v5a.0.1',
    'md5':'v5a.0.1',
    'mdv1000':'v1000.0.0-redwood',
    'md1000':'v1000.0.0-redwood',
    'default':'v5a.0.1',
    'megadetector':'v5a.0.1',
}

# python -m http.server 8181
model_url_base = 'https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/'
assert model_url_base.endswith('/')

if os.environ.get('MD_MODEL_URL_BASE') is not None:
    model_url_base = os.environ['MD_MODEL_URL_BASE']
    print('Model URL base provided via environment variable: {}'.format(
        model_url_base
    ))
    if not model_url_base.endswith('/'):
        model_url_base += '/'

# Maps canonical model version numbers to metadata
known_models = {
    'v2.0.0':
    {
        'url':'https://lila.science/public/models/megadetector/megadetector_v2.pb',
        'typical_detection_threshold':0.8,
        'conservative_detection_threshold':0.3,
        'model_type':'tf',
        'normalized_typical_inference_speed':1.0/3.5
    },
    'v3.0.0':
    {
        'url':'https://lila.science/public/models/megadetector/megadetector_v3.pb',
        'typical_detection_threshold':0.8,
        'conservative_detection_threshold':0.3,
        'model_type':'tf',
        'normalized_typical_inference_speed':1.0/3.5
    },
    'v4.1.0':
    {
        'url':'https://github.com/agentmorris/MegaDetector/releases/download/v4.1/md_v4.1.0.pb',
        'typical_detection_threshold':0.8,
        'conservative_detection_threshold':0.3,
        'model_type':'tf',
        'normalized_typical_inference_speed':1.0/3.5
    },
    'v5a.0.0':
    {
        'url':'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt',
        'typical_detection_threshold':0.2,
        'conservative_detection_threshold':0.05,
        'image_size':1280,
        'model_type':'yolov5',
        'normalized_typical_inference_speed':1.0,
        'md5':'ec1d7603ec8cf642d6e0cd008ba2be8c'
    },
    'v5b.0.0':
    {
        'url':'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt',
        'typical_detection_threshold':0.2,
        'conservative_detection_threshold':0.05,
        'image_size':1280,
        'model_type':'yolov5',
        'normalized_typical_inference_speed':1.0,
        'md5':'bc235e73f53c5c95e66ea0d1b2cbf542'
    },
    'v5a.0.1':
    {
        'url':'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.1.pt',
        'typical_detection_threshold':0.2,
        'conservative_detection_threshold':0.05,
        'image_size':1280,
        'model_type':'yolov5',
        'normalized_typical_inference_speed':1.0,
        'md5':'60f8e7ec1308554df258ed1f4040bc4f'
    },
    'v5b.0.1':
    {
        'url':'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.1.pt',
        'typical_detection_threshold':0.2,
        'conservative_detection_threshold':0.05,
        'image_size':1280,
        'model_type':'yolov5',
        'normalized_typical_inference_speed':1.0,
        'md5':'f17ed6fedfac2e403606a08c89984905'
    },
    'v1000.0.0-redwood':
    {
        'url':model_url_base + 'md_v1000.0.0-redwood.pt',
        'normalized_typical_inference_speed':1.0,
        'md5':'74474b3aec9cf1a990da38b37ddf9197'
    },
    'v1000.0.0-spruce':
    {
        'url':model_url_base + 'md_v1000.0.0-spruce.pt',
        'normalized_typical_inference_speed':12.7,
        'md5':'1c9d1d2b3ba54931881471fdd508e6f2'
    },
    'v1000.0.0-larch':
    {
        'url':model_url_base + 'md_v1000.0.0-larch.pt',
        'normalized_typical_inference_speed':2.4,
        'md5':'cab94ebd190c2278e12fb70ffd548b6d'
    },
    'v1000.0.0-cedar':
    {
        'url':model_url_base + 'md_v1000.0.0-cedar.pt',
        'normalized_typical_inference_speed':2.0,
        'md5':'3d6472c9b95ba687b59ebe255f7c576b'
    },
    'v1000.0.0-sorrel':
    {
        'url':model_url_base + 'md_v1000.0.0-sorrel.pt',
        'normalized_typical_inference_speed':7.0,
        'md5':'4339a2c8af7a381f18ded7ac2a4df03e'
    }
}

DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = known_models['v5a.0.0']['typical_detection_threshold']
DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.005

DEFAULT_BOX_THICKNESS = 4
DEFAULT_BOX_EXPANSION = 0
DEFAULT_LABEL_FONT_SIZE = 16
DETECTION_FILENAME_INSERT = '_detections'

# Approximate inference speeds (in images per second) for MDv5 based on
# benchmarks, only used for reporting very coarse expectations about inference time.
device_token_to_mdv5_inference_speed = {
    '4090':17.6,
    '3090':11.4,
    '3080':9.5,
    '3050':4.2,
    'P2000':2.1,
    # These are written this way because they're MDv4 benchmarks, and MDv5
    # is around 3.5x faster than MDv4.
    'V100':2.79*3.5,
    '2080':2.3*3.5,
    '2060':1.6*3.5
}


#%% Utility functions

def get_detector_metadata_from_version_string(detector_version):
    """
    Given a MegaDetector version string (e.g. "v4.1.0"), returns the metadata for
    the model.  Used for writing standard defaults to batch output files.

    Args:
        detector_version (str): a detection version string, e.g. "v4.1.0", which you
            can extract from a filename using get_detector_version_from_filename()

    Returns:
        dict: metadata for this model, suitable for writing to a MD output file
    """

    if detector_version not in known_models:
        print('Warning: no metadata for unknown detector version {}'.format(detector_version))
        default_detector_metadata = {
            'megadetector_version':'unknown',
            'typical_detection_threshold':0.2,
            'conservative_detection_threshold':0.1
        }
        return default_detector_metadata
    else:
        to_return = known_models[detector_version]
        to_return['megadetector_version'] = detector_version
        return to_return


def get_detector_version_from_filename(detector_filename,
                                       accept_first_match=True,
                                       verbose=False):
    r"""
    Gets the canonical version number string of a detector from the model filename.

    [detector_filename] will almost always end with one of the following:

    * megadetector_v2.pb
    * megadetector_v3.pb
    * megadetector_v4.1 (not produced by run_detector_batch.py, only found in output files from
      the deprecated Azure Batch API)
    * md_v4.1.0.pb
    * md_v5a.0.0.pt
    * md_v5b.0.0.pt

    This function identifies the version number as "v2.0.0", "v3.0.0", "v4.1.0",
    "v4.1.0", "v5a.0.0", and "v5b.0.0", respectively.  See known_models for the list
    of valid version numbers.

    Args:
        detector_filename (str): model filename, e.g. c:/x/z/md_v5a.0.0.pt
        accept_first_match (bool, optional): if multiple candidates match the filename, choose the
            first one, otherwise returns the string "multiple"
        verbose (bool, optional): enable additional debug output

    Returns:
        str: a detector version string, e.g. "v5a.0.0", or "multiple" if I'm confused
    """

    fn = os.path.basename(detector_filename).lower()
    matches = []
    for s in model_string_to_model_version.keys():
        if s in fn:
            matches.append(s)
    if len(matches) == 0:
        return 'unknown'
    elif len(matches) > 1:
        if accept_first_match:
            return model_string_to_model_version[matches[0]]
        else:
            if verbose:
                print('Warning: multiple MegaDetector versions for model file {}:'.format(detector_filename))
                for s in matches:
                    print(s)
            return 'multiple'
    else:
        return model_string_to_model_version[matches[0]]


def get_detector_version_from_model_file(detector_filename,verbose=False):
    """
    Gets the canonical detection version from a model file, preferably by reading it
    from the file itself, otherwise based on the filename.

    Args:
        detector_filename (str): model filename, e.g. c:/x/z/md_v5a.0.0.pt
        verbose (bool, optional): enable additional debug output

    Returns:
        str: a canonical detector version string, e.g. "v5a.0.0", or "unknown"
    """

    # Try to extract a version string from the filename
    version_string_based_on_filename = get_detector_version_from_filename(
        detector_filename, verbose=verbose)
    if version_string_based_on_filename == 'unknown':
        version_string_based_on_filename = None

    # Try to extract a version string from the file itself; currently this is only
    # a thing for PyTorch models

    version_string_based_on_model_file = None

    if detector_filename.endswith('.pt') or detector_filename.endswith('.zip'):

        from megadetector.detection.pytorch_detector import \
            read_metadata_from_megadetector_model_file
        metadata = read_metadata_from_megadetector_model_file(detector_filename,verbose=verbose)

        if metadata is not None and isinstance(metadata,dict):

            if 'metadata_format_version' not in metadata or \
                not isinstance(metadata['metadata_format_version'],float):

                print(f'Warning: I found a metadata file in detector file {detector_filename}, '+\
                      'but it doesn\'t have a valid format version number')

            elif 'model_version_string' not in metadata or \
                not isinstance(metadata['model_version_string'],str):

                print(f'Warning: I found a metadata file in detector file {detector_filename}, '+\
                      'but it doesn\'t have a format model version string')

            else:

                version_string_based_on_model_file = metadata['model_version_string']

                if version_string_based_on_model_file not in known_models:
                    print('Warning: unknown model version:\n\n{}\n\n...specified in file:\n\n{}'.format(
                        version_string_based_on_model_file,os.path.basename(detector_filename)))

        # ...if there's metadata in this file

    # ...if this looks like a PyTorch file

    # If we got versions strings from the filename *and* the model file...
    if (version_string_based_on_filename is not None) and \
       (version_string_based_on_model_file is not None):

        if version_string_based_on_filename != version_string_based_on_model_file:
            # This is a one-off special case where models were re-released with different filenames
            if (version_string_based_on_filename in ('v5a.0.1','v5b.0.1')) and \
                (version_string_based_on_model_file in ('v5a.0.0','v5b.0.0')):
                pass
            else:
                print(
                    'Warning: model version string in file:' + \
                    '\n\n{}\n\n...is:\n\n{}\n\n...but the filename implies:\n\n{}'.format(
                    os.path.basename(detector_filename),
                    version_string_based_on_model_file,
                    version_string_based_on_filename))

        return version_string_based_on_model_file

    # If we got version string from neither the filename nor the model file...
    if (version_string_based_on_filename is None) and \
       (version_string_based_on_model_file is None):

        print('Warning: could not determine model version string for model file {}'.format(
            detector_filename))
        return None

    elif version_string_based_on_filename is not None:

        return version_string_based_on_filename

    else:

        assert version_string_based_on_model_file is not None
        return version_string_based_on_model_file

# ...def get_detector_version_from_model_file(...)


def estimate_md_images_per_second(model_file, device_name=None):
    r"""
    Estimates how fast MegaDetector will run on a particular device, based on benchmarks.
    Defaults to querying the current device.  Returns None if no data is available for the current
    card/model.  Estimates only available for a small handful of GPUs.  Uses an absurdly simple
    lookup approach, e.g. if the string "4090" appears in the device name, congratulations,
    you have an RTX 4090.

    Args:
        model_file (str): model filename, e.g. c:/x/z/md_v5a.0.0.pt
        device_name (str, optional): device name, e.g. blah-blah-4090-blah-blah

    Returns:
        float: the approximate number of images this model version can process on this
        device per second
    """

    if device_name is None:
        try:
            import torch
            device_name = torch.cuda.get_device_name()
        except Exception as e:
            print('Error querying device name: {}'.format(e))
            return None

    # About how fast is this model compared to MDv5?
    model_version = get_detector_version_from_model_file(model_file)

    if model_version not in known_models.keys():
        print('Could not estimate inference speed: error determining model version for model file {}'.format(
            model_file))
        return None

    model_info = known_models[model_version]

    if 'normalized_typical_inference_speed' not in model_info or \
        model_info['normalized_typical_inference_speed'] is None:
        print('No speed ratio available for model type {}'.format(model_version))
        return None

    normalized_inference_speed = model_info['normalized_typical_inference_speed']

    # About how fast would MDv5 run on this device?
    mdv5_inference_speed = None
    for device_token in device_token_to_mdv5_inference_speed.keys():
        if device_token in device_name:
            mdv5_inference_speed = device_token_to_mdv5_inference_speed[device_token]
            break

    if mdv5_inference_speed is None:
        print('No baseline speed estimate available for device {}'.format(device_name))
        return None

    return normalized_inference_speed * mdv5_inference_speed


def get_typical_confidence_threshold_from_results(results):
    """
    Given the .json data loaded from a MD results file, returns a typical confidence
    threshold based on the detector version.

    Args:
        results (dict or str): a dict of MD results, as it would be loaded from a MD results .json
            file, or a .json filename

    Returns:
        float: a sensible default threshold for this model
    """

    # Load results if necessary
    if isinstance(results,str):
        with open(results,'r') as f:
            results = json.load(f)

    if 'detector_metadata' in results['info'] and \
        'typical_detection_threshold' in results['info']['detector_metadata']:
        default_threshold = results['info']['detector_metadata']['typical_detection_threshold']
    elif ('detector' not in results['info']) or (results['info']['detector'] is None):
        print('Warning: detector version not available in results file, using MDv5 defaults')
        detector_metadata = get_detector_metadata_from_version_string('v5a.0.0')
        default_threshold = detector_metadata['typical_detection_threshold']
    else:
        print('Warning: detector metadata not available in results file, inferring from MD version')
        detector_filename = results['info']['detector']
        detector_version = get_detector_version_from_filename(detector_filename)
        detector_metadata = get_detector_metadata_from_version_string(detector_version)
        default_threshold = detector_metadata['typical_detection_threshold']

    return default_threshold


def is_gpu_available(model_file):
    r"""
    Determines whether a GPU is available, importing PyTorch or TF depending on the extension
    of model_file.  Does not actually load model_file, just uses that to determine how to check
    for GPU availability (PT vs. TF).

    Args:
        model_file (str): model filename, e.g. c:/x/z/md_v5a.0.0.pt

    Returns:
        bool: whether a GPU is available
    """

    if model_file.endswith('.pb'):
        import tensorflow.compat.v1 as tf
        gpu_available = tf.test.is_gpu_available()
        print('TensorFlow version:', tf.__version__)
        print('tf.test.is_gpu_available:', gpu_available)
        return gpu_available
    if not model_file.endswith('.pt'):
        print('Warning: could not determine environment from model file name, assuming PyTorch')

    import torch
    gpu_available = torch.cuda.is_available()
    print('PyTorch reports {} available CUDA devices'.format(torch.cuda.device_count()))
    if not gpu_available:
        try:
            # mps backend only available in torch >= 1.12.0
            if torch.backends.mps.is_built and torch.backends.mps.is_available():
                gpu_available = True
                print('PyTorch reports Metal Performance Shaders are available')
        except AttributeError:
            pass
    return gpu_available


def load_detector(model_file,
                  force_cpu=False,
                  force_model_download=False,
                  detector_options=None,
                  verbose=False):
    r"""
    Loads a TF or PT detector, depending on the extension of model_file.

    Args:
        model_file (str): model filename (e.g. c:/x/z/md_v5a.0.0.pt) or known model
            name (e.g. "MDV5A")
        force_cpu (bool, optional): force the model to run on the CPU even if a GPU
            is available
        force_model_download (bool, optional): force downloading the model file if
            a named model (e.g. "MDV5A") is supplied, even if the local file already
            exists
        detector_options (dict, optional): key/value pairs that are interpreted differently
            by different detectors
        verbose (bool, optional): enable additional debug output

    Returns:
        object: loaded detector object
    """

    # Possibly automatically download the model
    model_file = try_download_known_detector(model_file,
                                             force_download=force_model_download)

    if verbose:
        print('GPU available: {}'.format(is_gpu_available(model_file)))

    start_time = time.time()

    if model_file.endswith('.pb'):

        from megadetector.detection.tf_detector import TFDetector
        if force_cpu:
            raise ValueError('force_cpu is not currently supported for TF detectors, ' + \
                             'use CUDA_VISIBLE_DEVICES=-1 instead')
        detector = TFDetector(model_file, detector_options)

    elif model_file.endswith('.pt'):

        from megadetector.detection.pytorch_detector import PTDetector

        # Prepare options specific to the PTDetector class
        if detector_options is None:
            detector_options = {}
        if 'force_cpu' in detector_options:
            if force_cpu != detector_options['force_cpu']:
                print('Warning: over-riding force_cpu parameter ({}) based on detector_options ({})'.format(
                    force_cpu,detector_options['force_cpu']))
        else:
            detector_options['force_cpu'] = force_cpu
        detector_options['use_model_native_classes'] = USE_MODEL_NATIVE_CLASSES
        detector = PTDetector(model_file, detector_options, verbose=verbose)

    else:

        raise ValueError('Unrecognized model format: {}'.format(model_file))

    elapsed = time.time() - start_time

    if verbose:
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    return detector

# ...def load_detector(...)


#%% Main function

def load_and_run_detector(model_file,
                          image_file_names,
                          output_dir,
                          render_confidence_threshold=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
                          crop_images=False,
                          box_thickness=DEFAULT_BOX_THICKNESS,
                          box_expansion=DEFAULT_BOX_EXPANSION,
                          image_size=None,
                          label_font_size=DEFAULT_LABEL_FONT_SIZE,
                          augment=False,
                          force_model_download=False,
                          detector_options=None,
                          verbose=False):
    r"""
    Loads and runs a detector on target images, and visualizes the results.

    Args:
        model_file (str): model filename, e.g. c:/x/z/md_v5a.0.0.pt, or a known model
            string, e.g. "MDV5A"
        image_file_names (list): list of absolute paths to process
        output_dir (str): folder to write visualized images to
        render_confidence_threshold (float, optional): only render boxes for detections
            above this threshold
        crop_images (bool, optional): whether to crop detected objects to individual images
            (default is to render images with boxes, rather than cropping)
        box_thickness (float, optional): thickness in pixels for box rendering
        box_expansion (float, optional): box expansion in pixels
        image_size (tuple, optional): image size to use for inference, only mess with this
            if (a) you're using a model other than MegaDetector or (b) you know what you're
            doing
        label_font_size (float, optional): font size to use for displaying class names
            and confidence values in the rendered images
        augment (bool, optional): enable (implementation-specific) image augmentation
        force_model_download (bool, optional): force downloading the model file if
            a named model (e.g. "MDV5A") is supplied, even if the local file already
            exists
        detector_options (dict, optional): key/value pairs that are interpreted differently
            by different detectors
        verbose (bool, optional): enable additional debug output
    """

    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    # Possibly automatically download the model
    model_file = try_download_known_detector(model_file,
                                             force_download=force_model_download,
                                             verbose=verbose)

    detector = load_detector(model_file,
                             detector_options=detector_options,
                             verbose=verbose)

    detection_results = []
    time_load = []
    time_infer = []

    # Dictionary mapping output file names to a collision-avoidance count.
    #
    # Since we'll be writing a bunch of files to the same folder, we rename
    # as necessary to avoid collisions.
    output_filename_collision_counts = {}

    def input_file_to_detection_file(fn, crop_index=-1):
        """
        Creates unique file names for output files.

        This function does 3 things:
        1) If the --crop flag is used, then each input image may produce several output
            crops. For example, if foo.jpg has 3 detections, then this function should
            get called 3 times, with crop_index taking on 0, 1, then 2. Each time, this
            function appends crop_index to the filename, resulting in
                foo_crop00_detections.jpg
                foo_crop01_detections.jpg
                foo_crop02_detections.jpg

        2) If the --recursive flag is used, then the same file (base)name may appear
            multiple times. However, we output into a single flat folder. To avoid
            filename collisions, we prepend an integer prefix to duplicate filenames:
                foo_crop00_detections.jpg
                0000_foo_crop00_detections.jpg
                0001_foo_crop00_detections.jpg

        3) Prepends the output directory:
                out_dir/foo_crop00_detections.jpg

        Args:
            fn: str, filename
            crop_index: int, crop number

        Returns: output file path
        """

        fn = os.path.basename(fn).lower()
        name, ext = os.path.splitext(fn)
        if crop_index >= 0:
            name += '_crop{:0>2d}'.format(crop_index)
        fn = '{}{}{}'.format(name, DETECTION_FILENAME_INSERT, '.jpg')
        if fn in output_filename_collision_counts:
            n_collisions = output_filename_collision_counts[fn]
            fn = '{:0>4d}'.format(n_collisions) + '_' + fn
            output_filename_collision_counts[fn] += 1
        else:
            output_filename_collision_counts[fn] = 0
        fn = os.path.join(output_dir, fn)
        return fn

    # ...def input_file_to_detection_file()

    for im_file in tqdm(image_file_names):

        try:
            start_time = time.time()

            image = vis_utils.load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('Image {} cannot be loaded, error: {}'.format(im_file, str(e)))
            result = {
                'file': im_file,
                'failure': FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            result = detector.generate_detections_one_image(
                image,
                im_file,
                detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                image_size=image_size,
                augment=augment)
            detection_results.append(result)

            elapsed = time.time() - start_time
            time_infer.append(elapsed)

        except Exception as e:
            print('An error occurred while running the detector on image {}: {}'.format(
                im_file, str(e)))
            continue

        try:
            if crop_images:

                images_cropped = vis_utils.crop_image(result['detections'], image,
                                   confidence_threshold=render_confidence_threshold,
                                   expansion=box_expansion)

                for i_crop, cropped_image in enumerate(images_cropped):
                    output_full_path = input_file_to_detection_file(im_file, i_crop)
                    cropped_image.save(output_full_path)

            else:

                # Image is modified in place
                vis_utils.render_detection_bounding_boxes(result['detections'], image,
                            label_map=DEFAULT_DETECTOR_LABEL_MAP,
                            confidence_threshold=render_confidence_threshold,
                            thickness=box_thickness, expansion=box_expansion,
                            label_font_size=label_font_size,
                            box_sort_order='confidence')
                output_full_path = input_file_to_detection_file(im_file)
                image.save(output_full_path)

        except Exception as e:
            print('Visualizing results on the image {} failed. Exception: {}'.format(im_file, e))
            continue

    # ...for each image

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available (<=1 image processed)'
        std_dev_time_infer = 'not available (<=1 image processed)'
    print('On average, for each image,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))

# ...def load_and_run_detector()


def _validate_zip_file(file_path, file_description='file'):
    """
    Validates that a .pt file is a valid zip file.

    Args:
        file_path (str): path to the file to validate
        file_description (str): descriptive string for error messages

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zipf:
            zipf.testzip()
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        print('{} {} appears to be corrupted (bad zip): {}'.format(
            file_description.capitalize(), file_path, str(e)))
        return False
    except Exception as e:
        print('Error validating {}: {}'.format(file_description, str(e)))
        return False


def _validate_md5_hash(file_path, expected_hash, file_description='file'):
    """
    Validates that a file has the expected MD5 hash.

    Args:
        file_path (str): path to the file to validate
        expected_hash (str): expected MD5 hash
        file_description (str): descriptive string for error messages

    Returns:
        bool: True if hash matches, False otherwise
    """
    try:
        actual_hash = compute_file_hash(file_path, algorithm='md5').lower()
        expected_hash = expected_hash.lower()
        if actual_hash != expected_hash:
            print('{} {} has incorrect hash. Expected: {}, Actual: {}'.format(
                file_description.capitalize(), file_path, expected_hash, actual_hash))
            return False
        return True
    except Exception as e:
        print('Error computing hash for {}: {}'.format(file_description, str(e)))
        return False


def _download_model(model_name,force_download=False):
    """
    Downloads one of the known models to local temp space if it hasn't already been downloaded.

    Args:
        model_name (str): a known model string, e.g. "MDV5A".  Returns None if this string is not
            a known model name.
        force_download (bool, optional): whether to download the model even if the local target
            file already exists
    """

    model_tempdir = os.path.join(tempfile.gettempdir(), 'megadetector_models')
    os.makedirs(model_tempdir,exist_ok=True)

    # This is a lazy fix to an issue... if multiple users run this script, the
    # "megadetector_models" folder is owned by the first person who creates it, and others
    # can't write to it.  I could create uniquely-named folders, but I philosophically prefer
    # to put all the individual UUID-named folders within a larger folder, so as to be a
    # good tempdir citizen.  So, the lazy fix is to make this world-writable.
    try:
        os.chmod(model_tempdir,0o777)
    except Exception:
        pass
    if model_name.lower() not in known_models:
        print('Unrecognized downloadable model {}'.format(model_name))
        return None

    model_info = known_models[model_name.lower()]
    url = model_info['url']
    destination_filename = os.path.join(model_tempdir,url.split('/')[-1])

    # Check whether the file already exists, in which case we want to validate it
    if os.path.exists(destination_filename) and not force_download:

        # Only validate .pt files, not .pb files
        if destination_filename.endswith('.pt'):

            is_valid = True

            # Check whether the file is a valid zip file (.pt files are zip files in disguise)
            if not _validate_zip_file(destination_filename,
                                      'existing model file'):
                is_valid = False

            # Check MD5 hash if available
            if is_valid and \
                ('md5' in model_info) and \
                (model_info['md5'] is not None) and \
                (len(model_info['md5'].strip()) > 0):

                if not _validate_md5_hash(destination_filename, model_info['md5'],
                                          'existing model file'):
                    is_valid = False

            # If validation failed, delete the corrupted file and re-download
            if not is_valid:
                print('Deleting corrupted model file and re-downloading: {}'.format(
                    destination_filename))
                try:
                    os.remove(destination_filename)
                    # This should be a no-op at this point, but it can't hurt
                    force_download = True
                except Exception as e:
                    print('Warning: failed to delete corrupted file {}: {}'.format(
                        destination_filename, str(e)))
                    # Continue with download attempt anyway, setting force_download to True
                    force_download = True
            else:
                print('Model {} already exists and is valid at {}'.format(
                    model_name, destination_filename))
                return destination_filename

    # Download the model
    try:
        local_file = download_url(url,
                                destination_filename=destination_filename,
                                progress_updater=None,
                                force_download=force_download,
                                verbose=True)
    except Exception as e:
        print('Error downloading model {} from {}: {}'.format(model_name, url, str(e)))
        raise

    # Validate the downloaded file if it's a .pt file
    if local_file and local_file.endswith('.pt'):

        # Check if the downloaded file is a valid zip file
        if not _validate_zip_file(local_file, "downloaded model file"):
            # Clean up the corrupted download
            try:
                os.remove(local_file)
            except Exception:
                pass
            return None

        # Check MD5 hash if available
        if ('md5' in model_info) and \
            (model_info['md5'] is not None) and \
            (len(model_info['md5'].strip()) > 0):

            if not _validate_md5_hash(local_file, model_info['md5'], "downloaded model file"):
                # Clean up the corrupted download
                try:
                    os.remove(local_file)
                except Exception:
                    pass
                return None

    print('Model {} available at {}'.format(model_name,local_file))
    return local_file

# ...def _download_model(...)

def try_download_known_detector(detector_file,force_download=False,verbose=False):
    """
    Checks whether detector_file is really the name of a known model, in which case we will
    either read the actual filename from the corresponding environment variable or download
    (if necessary) to local temp space.  Otherwise just returns the input string.

    Args:
        detector_file (str): a known model string (e.g. "MDV5A"), or any other string (in which
            case this function is a no-op)
        force_download (bool, optional): whether to download the model even if the local target
            file already exists
        verbose (bool, optional): enable additional debug output

    Returns:
        str: the local filename to which the model was downloaded, or the same string that
        was passed in, if it's not recognized as a well-known model name
    """

    model_string = detector_file.lower()

    # If this is a short model string (e.g. "MDV5A"), convert to a canonical version
    # string (e.g. "v5a.0.0")
    if model_string in model_string_to_model_version:

        if verbose:
            print('Converting short string {} to canonical version string {}'.format(
                model_string,
                model_string_to_model_version[model_string]))
        model_string = model_string_to_model_version[model_string]

    if model_string in known_models:

        if detector_file in os.environ:
            fn = os.environ[detector_file]
            print('Reading MD location from environment variable {}: {}'.format(
                detector_file,fn))
            detector_file = fn
        else:
            detector_file = _download_model(model_string,force_download=force_download)

    return detector_file




#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser(
        description='Module to run an animal detection model on images')

    parser.add_argument(
        'detector_file',
        help='Path detector model file (.pb or .pt).  Can also be MDV4, MDV5A, or MDV5B to request automatic download.')

    # Must specify either an image file or a directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image_file',
        type=str,
        default=None,
        help='Single file to process, mutually exclusive with --image_dir')
    group.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory to search for images, with optional recursion by adding --recursive')

    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir')

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for output images (defaults to same as input)')

    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help=('Force image resizing to a (square) integer size (not recommended to change this)'))

    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
        help=('Confidence threshold between 0 and 1.0; only render' +
              ' boxes above this confidence (defaults to {})'.format(
              DEFAULT_RENDERING_CONFIDENCE_THRESHOLD)))

    parser.add_argument(
        '--crop',
        default=False,
        action='store_true',
        help=('If set, produces separate output images for each crop, '
              'rather than adding bounding boxes to the original image'))

    parser.add_argument(
        '--augment',
        default=False,
        action='store_true',
        help=('Enable image augmentation'))

    parser.add_argument(
        '--box_thickness',
        type=int,
        default=DEFAULT_BOX_THICKNESS,
        help=('Line width (in pixels) for box rendering (defaults to {})'.format(
              DEFAULT_BOX_THICKNESS)))

    parser.add_argument(
        '--box_expansion',
        type=int,
        default=DEFAULT_BOX_EXPANSION,
        help=('Number of pixels to expand boxes by (defaults to {})'.format(
              DEFAULT_BOX_EXPANSION)))

    parser.add_argument(
        '--label_font_size',
        type=int,
        default=DEFAULT_LABEL_FONT_SIZE,
        help=('Label font size (defaults to {})'.format(
              DEFAULT_LABEL_FONT_SIZE)))

    parser.add_argument(
        '--process_likely_output_images',
        action='store_true',
        help=('By default, we skip images that end in {}, because they probably came from this script. '\
              .format(DETECTION_FILENAME_INSERT) + \
              'This option disables that behavior.'))

    parser.add_argument(
        '--force_model_download',
        action='store_true',
        help=('If a named model (e.g. "MDV5A") is supplied, force a download of that model even if the ' +\
              'local file already exists.'))

    parser.add_argument(
        '--verbose',
        action='store_true',
        help=('Enable additional debug output'))

    parser.add_argument(
        '--detector_options',
        nargs='*',
        metavar='KEY=VALUE',
        default='',
        help='Detector-specific options, as a space-separated list of key-value pairs')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    detector_options = parse_kvp_list(args.detector_options)

    # If the specified detector file is really the name of a known model, find
    # (and possibly download) that model
    args.detector_file = try_download_known_detector(args.detector_file,
                                                     force_download=args.force_model_download)

    assert os.path.exists(args.detector_file), 'detector file {} does not exist'.format(
        args.detector_file)
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        image_file_names = path_utils.find_images(args.image_dir, args.recursive)

    # Optionally skip images that were probably generated by this script
    if not args.process_likely_output_images:
        image_file_names_valid = []
        for fn in image_file_names:
            if os.path.splitext(fn)[0].endswith(DETECTION_FILENAME_INSERT):
                print('Skipping likely output image {}'.format(fn))
            else:
                image_file_names_valid.append(fn)
        image_file_names = image_file_names_valid

    print('Running detector on {} images...'.format(len(image_file_names)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            # but for a single image, args.image_dir is also None
            args.output_dir = os.path.dirname(args.image_file)

    load_and_run_detector(model_file=args.detector_file,
                          image_file_names=image_file_names,
                          output_dir=args.output_dir,
                          render_confidence_threshold=args.threshold,
                          box_thickness=args.box_thickness,
                          box_expansion=args.box_expansion,
                          crop_images=args.crop,
                          image_size=args.image_size,
                          label_font_size=args.label_font_size,
                          augment=args.augment,
                          # If --force_model_download was specified, we already handled it
                          force_model_download=False,
                          detector_options=detector_options,
                          verbose=args.verbose)

if __name__ == '__main__':
    main()


#%% Interactive driver(s)

if False:

    pass

    #%% Test model download

    r"""
    cd i:\models\all_models_in_the_wild
    i:
    python -m http.server 8181
    """

    model_name = 'redwood'
    try_download_known_detector(model_name,force_download=True,verbose=True)


    #%% Load and run detector

    model_file = r'c:\temp\models\md_v4.1.0.pb'
    image_file_names = path_utils.find_images(r'c:\temp\demo_images\ssverymini')
    output_dir = r'c:\temp\demo_images\ssverymini'
    render_confidence_threshold = 0.8
    crop_images = True

    load_and_run_detector(model_file=model_file,
                          image_file_names=image_file_names,
                          output_dir=output_dir,
                          render_confidence_threshold=render_confidence_threshold,
                          crop_images=crop_images)
