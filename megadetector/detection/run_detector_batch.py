"""

run_detector_batch.py

Module to run MegaDetector on lots of images, writing the results
to a file in the MegaDetector results format.

https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing#megadetector-batch-output-format

This enables the results to be used in our post-processing pipeline; see postprocess_batch_results.py.

This script can save results to checkpoints intermittently, in case disaster
strikes. To enable this, set --checkpoint_frequency to n > 0, and results 
will be saved as a checkpoint every n images. Checkpoints will be written 
to a file in the same directory as the output_file, and after all images
are processed and final results file written to output_file, the temporary
checkpoint file will be deleted. If you want to resume from a checkpoint, set
the checkpoint file's path using --resume_from_checkpoint.

Has multiprocessing support for CPUs only; if a GPU is available, it will
use the GPU instead of CPUs, and the --ncores option will be ignored.  Checkpointing
is not supported when using a GPU.

The lack of GPU multiprocessing support might sound annoying, but in practice we
run a gazillion MegaDetector images on multiple GPUs using this script, we just only use
one GPU *per invocation of this script*.  Dividing a big batch of images into one chunk
per GPU happens outside of this script.

Does not have a command-line option to bind the process to a particular GPU, but you can 
prepend with "CUDA_VISIBLE_DEVICES=0 ", for example, to bind to GPU 0, e.g.:

CUDA_VISIBLE_DEVICES=0 python detection/run_detector_batch.py md_v4.1.0.pb ~/data ~/mdv4test.json 

You can disable GPU processing entirely by setting CUDA_VISIBLE_DEVICES=''.

"""

#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import shutil
import warnings
import itertools
import humanfriendly

from datetime import datetime
from functools import partial
from copy import deepcopy
from tqdm import tqdm

import multiprocessing
from threading import Thread
from multiprocessing import Process, Manager

# This pool is used for multi-CPU parallelization, not for data loading workers
# from multiprocessing.pool import ThreadPool as workerpool
from multiprocessing.pool import Pool as workerpool

from megadetector.detection import run_detector
from megadetector.detection.run_detector import \
    is_gpu_available,\
    load_detector,\
    try_download_known_detector,\
    get_detector_version_from_filename,\
    get_detector_metadata_from_version_string

from megadetector.utils import path_utils
from megadetector.utils.ct_utils import parse_kvp_list
from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.data_management import read_exif
from megadetector.data_management.yolo_output_to_md_output import read_classes_from_yolo_dataset_file

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Default number of loaders to use when --image_queue is set
default_loaders = 4

# Should we do preprocessing on the image queue?
default_preprocess_on_image_queue = False

# Number of images to pre-fetch per worker
max_queue_size = 10

# How often should we print progress when using the image queue?
n_queue_print = 1000

# TODO: it's a little sloppy that these are module-level globals, but in practice it 
# doesn't really matter, so I'm not in a big rush to move these to options until I do
# a larger cleanup of all the long argument lists in this module.
#
# Should the consumer loop run on its own process, or here in the main process?
run_separate_consumer_process = False
use_threads_for_queue = False
verbose = False

exif_options = read_exif.ReadExifOptions()
exif_options.processing_library = 'pil'
exif_options.byte_handling = 'convert_to_string'


#%% Support functions for multiprocessing

def _producer_func(q,
                   image_files,
                   producer_id=-1,
                   preprocessor=None,
                   detector_options=None,
                   verbose=False,
                   image_size=None,
                   augment=None):
    """ 
    Producer function; only used when using the (optional) image queue.
    
    Reads up to images from disk and puts them on the blocking queue for 
    processing.  Each image is queued as a tuple of [filename,Image].  Sends 
    "None" to the queue when finished.
    
    The "detector" argument is only used for preprocessing.
    """
    
    if verbose:
        print('Producer starting: ID {}, preprocessor {}'.format(producer_id,preprocessor))
        sys.stdout.flush()
        
    if preprocessor is not None:
        assert isinstance(preprocessor,str)
        detector_options = deepcopy(detector_options)
        detector_options['preprocess_only'] = True
        preprocessor = load_detector(preprocessor,detector_options=detector_options,verbose=verbose)
        
    for im_file in image_files:
    
        try:
            if verbose:
                print('Loading image {} on producer {}'.format(im_file,producer_id))
                sys.stdout.flush()
            image = vis_utils.load_image(im_file)
                        
            if preprocessor is not None:
                
                image_info = preprocessor.generate_detections_one_image(
                                                  image, 
                                                  im_file, 
                                                  detection_threshold=None, 
                                                  image_size=image_size,
                                                  skip_image_resizing=False,
                                                  augment=augment,
                                                  preprocess_only=True,
                                                  verbose=verbose)
                if 'failure' in image_info:
                    assert image_info['failure'] == run_detector.FAILURE_INFER
                    raise
                    
                image = image_info
                
        except Exception:
            print('Producer process: image {} cannot be loaded'.format(im_file))
            image = run_detector.FAILURE_IMAGE_OPEN            
        
        if verbose:
            print('Queueing image {} from producer {}'.format(im_file,producer_id))
            sys.stdout.flush()
        
        q.put([im_file,image,producer_id])
    
    # This is a signal to the consumer function that a worker has finished
    q.put(None)
        
    if verbose:
        print('Loader worker {} finished'.format(producer_id))
    sys.stdout.flush()

# ...def _producer_func(...)
    
    
def _consumer_func(q,
                   return_queue,
                   model_file,
                   confidence_threshold,
                   loader_workers,
                   image_size=None,
                   include_image_size=False,
                   include_image_timestamp=False, 
                   include_exif_data=False,
                   augment=False,
                   detector_options=None,
                   preprocess_on_image_queue=default_preprocess_on_image_queue,
                   n_total_images=None
                   ):
    """ 
    Consumer function; only used when using the (optional) image queue.
    
    Pulls images from a blocking queue and processes them.  Returns when "None" has
    been read from each loader's queue.
    """
    
    if verbose:
        print('Consumer starting'); sys.stdout.flush()

    start_time = time.time()
    
    if isinstance(model_file,str):
        detector = load_detector(model_file,detector_options=detector_options,verbose=verbose)
        elapsed = time.time() - start_time
        print('Loaded model (before queueing) in {}, printing updates every {} images'.format(
            humanfriendly.format_timespan(elapsed),n_queue_print))
        sys.stdout.flush()
    else:
        detector = model_file
        print('Detector of type {} passed to consumer function'.format(type(detector)))
        
    results = []
    
    n_images_processed = 0
    n_queues_finished = 0
    
    pbar = None
    if n_total_images is not None:
        # TODO: in principle I should close this pbar
        pbar = tqdm(total=n_total_images)
        
    while True:
        
        r = q.get()
        
        # Is this the last image in one of the producer queues?
        if r is None:
            n_queues_finished += 1
            q.task_done()
            if verbose:
                print('Consumer thread: {} of {} queues finished'.format(
                    n_queues_finished,loader_workers))
            if n_queues_finished == loader_workers:
                return_queue.put(results)
                return
            else:
                continue
        n_images_processed += 1
        im_file = r[0]
        image = r[1]
        
        """
        result['img_processed'] = img
        result['img_original'] = img_original
        result['target_shape'] = target_shape
        result['scaling_shape'] = scaling_shape
        result['letterbox_ratio'] = letterbox_ratio
        result['letterbox_pad'] = letterbox_pad
        """
        
        if pbar is not None:
            pbar.update(1)
            
        if False:
            if verbose or ((n_images_processed % n_queue_print) == 1):
                elapsed = time.time() - start_time
                images_per_second = n_images_processed / elapsed
                print('De-queued image {} ({:.2f}/s) ({})'.format(n_images_processed,
                                                              images_per_second,
                                                              im_file));
                sys.stdout.flush()
          
        if isinstance(image,str):
            # This is how the producer function communicates read errors
            results.append({'file': im_file,
                            'failure': image})
        elif preprocess_on_image_queue and (not isinstance(image,dict)):
                print('Expected a dict, received an image of type {}'.format(type(image)))
                results.append({'file': im_file,
                                'failure': 'illegal image type'})
            
        else:
            results.append(process_image(im_file=im_file,
                                         detector=detector,
                                         confidence_threshold=confidence_threshold,
                                         image=image,
                                         quiet=True,
                                         image_size=image_size,
                                         include_image_size=include_image_size,
                                         include_image_timestamp=include_image_timestamp, 
                                         include_exif_data=include_exif_data,
                                         augment=augment,
                                         skip_image_resizing=preprocess_on_image_queue))
        if verbose:
            print('Processed image {}'.format(im_file)); sys.stdout.flush()
        q.task_done()
            
    # ...while True (consumer loop)

# ...def _consumer_func(...)


def run_detector_with_image_queue(image_files,
                                  model_file,
                                  confidence_threshold,
                                  quiet=False,
                                  image_size=None,
                                  include_image_size=False, 
                                  include_image_timestamp=False,
                                  include_exif_data=False,
                                  augment=False,
                                  detector_options=None,
                                  loader_workers=default_loaders,
                                  preprocess_on_image_queue=default_preprocess_on_image_queue):
    """
    Driver function for the (optional) multiprocessing-based image queue; only used 
    when --use_image_queue is specified.  Starts a reader process to read images from disk, but 
    processes images in the  process from which this function is called (i.e., does not currently
    spawn a separate consumer process).
    
    Args:
        image_files (str): list of absolute paths to images
        model_file (str): filename or model identifier (e.g. "MDV5A")
        confidence_threshold (float): minimum confidence detection to include in
            output
        quiet (bool, optional): suppress per-image console printouts
        image_size (tuple, optional): image size to use for inference, only mess with this
            if (a) you're using a model other than MegaDetector or (b) you know what you're
            doing
        include_image_size (bool, optional): should we include image size in the output for each image?
        include_image_timestamp (bool, optional): should we include image timestamps in the output for each image?
        include_exif_data (bool, optional): should we include EXIF data in the output for each image?
        augment (bool, optional): enable image augmentation
        detector_options (dict, optional): key/value pairs that are interpreted differently 
            by different detectors
        loader_workers (int, optional): number of loaders to use
            
    Returns:
        list: list of dicts in the format returned by process_image()
    """
    
    # Validate inputs
    assert isinstance(model_file,str)
    
    if loader_workers <= 0:
        loader_workers = 1
        
    q = multiprocessing.JoinableQueue(max_queue_size)
    return_queue = multiprocessing.Queue(1)
    
    producers = []
    
    worker_string = 'thread' if use_threads_for_queue else 'process'
    print('Starting a {} pool with {} workers'.format(worker_string,loader_workers))
    
    preprocessor = None
    
    if preprocess_on_image_queue:
        preprocessor = model_file
    
    n_total_images = len(image_files)
    
    chunks = split_list_into_n_chunks(image_files, loader_workers, chunk_strategy='greedy')
    for i_chunk,chunk in enumerate(chunks):
        if use_threads_for_queue:
            producer = Thread(target=_producer_func,args=(q,
                                                          chunk,
                                                          i_chunk,preprocessor,
                                                          detector_options,
                                                          verbose,
                                                          image_size,
                                                          augment))
        else:
            producer = Process(target=_producer_func,args=(q,
                                                           chunk,
                                                           i_chunk,
                                                           preprocessor,
                                                           detector_options,
                                                           verbose,
                                                           image_size,
                                                           augment))
        producers.append(producer)
        
    for producer in producers:
        producer.daemon = False
        producer.start()
    
    if run_separate_consumer_process:
        if use_threads_for_queue:
            consumer = Thread(target=_consumer_func,args=(q,
                                                          return_queue,
                                                          model_file,
                                                          confidence_threshold,
                                                          loader_workers,
                                                          image_size,
                                                          include_image_size,
                                                          include_image_timestamp, 
                                                          include_exif_data,
                                                          augment,
                                                          detector_options,
                                                          preprocess_on_image_queue,
                                                          n_total_images))
        else:
            consumer = Process(target=_consumer_func,args=(q,
                                                           return_queue,
                                                           model_file,
                                                           confidence_threshold,
                                                           loader_workers,
                                                           image_size,
                                                           include_image_size,
                                                           include_image_timestamp, 
                                                           include_exif_data,
                                                           augment,
                                                           detector_options,
                                                           preprocess_on_image_queue,
                                                           n_total_images))
        consumer.daemon = True
        consumer.start()
    else:
        _consumer_func(q,
                       return_queue,
                       model_file,
                       confidence_threshold,
                       loader_workers,
                       image_size,
                       include_image_size,
                       include_image_timestamp, 
                       include_exif_data,
                       augment,
                       detector_options,
                       preprocess_on_image_queue,
                       n_total_images)

    for i_producer,producer in enumerate(producers):
        producer.join()
        if verbose:
            print('Producer {} finished'.format(i_producer))
    
    if verbose:
        print('All producers finished')
   
    if run_separate_consumer_process:
        consumer.join()
    if verbose:
        print('Consumer loop finished')
    
    q.join()
    if verbose:
        print('Queue joined')

    results = return_queue.get()
    
    return results

# ...def run_detector_with_image_queue(...)


#%% Other support functions

def _chunks_by_number_of_chunks(ls, n):
    """
    Splits a list into n even chunks.
    
    External callers should use ct_utils.split_list_into_n_chunks().

    Args:
        ls (list): list to break up into chunks
        n (int): number of chunks
    """
    
    for i in range(0, n):
        yield ls[i::n]


#%% Image processing functions

def process_images(im_files, 
                   detector, 
                   confidence_threshold, 
                   use_image_queue=False, 
                   quiet=False, 
                   image_size=None, 
                   checkpoint_queue=None, 
                   include_image_size=False, 
                   include_image_timestamp=False, 
                   include_exif_data=False,
                   augment=False,
                   detector_options=None,
                   loader_workers=default_loaders,
                   preprocess_on_image_queue=default_preprocess_on_image_queue):
    """
    Runs a detector (typically MegaDetector) over a list of image files on a single thread.
    
    Args:
        im_files (list: paths to image files                                   
        detector (str or detector object): loaded model or str; if this is a string, it can be a
            path to a .pb/.pt model file or a known model identifier (e.g. "MDV5A")
        confidence_threshold (float): only detections above this threshold are returned
        use_image_queue (bool, optional): separate image loading onto a dedicated worker process
        quiet (bool, optional): suppress per-image printouts
        image_size (tuple, optional): image size to use for inference, only mess with this
            if (a) you're using a model other than MegaDetector or (b) you know what you're
            doing
        checkpoint_queue (Queue, optional): internal parameter used to pass image queues around
        include_image_size (bool, optional): should we include image size in the output for each image?
        include_image_timestamp (bool, optional): should we include image timestamps in the output for each image?
        include_exif_data (bool, optional): should we include EXIF data in the output for each image?
        augment (bool, optional): enable image augmentation
        detector_options (dict, optional): key/value pairs that are interpreted differently 
            by different detectors
        loader_workers (int, optional): number of loaders to use (only relevant when using image queue)

    Returns:
        list: list of dicts, in which each dict represents detections on one image,
        see the 'images' key in https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing#batch-processing-api-output-format
    """
    
    if isinstance(detector, str):
        
        start_time = time.time()
        detector = load_detector(detector,detector_options=detector_options,verbose=verbose)
        elapsed = time.time() - start_time
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    if use_image_queue:
        
        run_detector_with_image_queue(im_files, 
                                      detector, 
                                      confidence_threshold, 
                                      quiet=quiet, 
                                      image_size=image_size,
                                      include_image_size=include_image_size, 
                                      include_image_timestamp=include_image_timestamp,
                                      include_exif_data=include_exif_data,
                                      augment=augment,
                                      detector_options=detector_options,
                                      loader_workers=loader_workers,
                                      preprocess_on_image_queue=preprocess_on_image_queue)
        
    else:            
        
        results = []
        for im_file in im_files:
            result = process_image(im_file, 
                                   detector, 
                                   confidence_threshold,
                                   quiet=quiet, 
                                   image_size=image_size, 
                                   include_image_size=include_image_size, 
                                   include_image_timestamp=include_image_timestamp,
                                   include_exif_data=include_exif_data,
                                   augment=augment)

            if checkpoint_queue is not None:
                checkpoint_queue.put(result)
            results.append(result)                                    
            
        return results

# ...def process_images(...)


def process_image(im_file, 
                  detector, 
                  confidence_threshold, 
                  image=None, 
                  quiet=False, 
                  image_size=None, 
                  include_image_size=False,
                  include_image_timestamp=False, 
                  include_exif_data=False,
                  skip_image_resizing=False,
                  augment=False):
    """
    Runs a detector (typically MegaDetector) on a single image file.

    Args:
        im_file (str): path to image file
        detector (detector object): loaded model, this can no longer be a string by the time 
            you get this far down the pipeline
        confidence_threshold (float): only detections above this threshold are returned
        image (Image, optional): previously-loaded image, if available, used when a worker
            thread is handling image loads
        quiet (bool, optional): suppress per-image printouts
        image_size (tuple, optional): image size to use for inference, only mess with this
            if (a) you're using a model other than MegaDetector or (b) you know what you're
            doing        
        include_image_size (bool, optional): should we include image size in the output for each image?
        include_image_timestamp (bool, optional): should we include image timestamps in the output for each image?
        include_exif_data (bool, optional): should we include EXIF data in the output for each image?                
        skip_image_resizing (bool, optional): whether to skip internal image resizing and rely on external resizing
        augment (bool, optional): enable image augmentation

    Returns:
        dict: dict representing detections on one image,
        see the 'images' key in 
        https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing#batch-processing-api-output-format
    """
    
    if not quiet:
        print('Processing image {}'.format(im_file))
    
    if image is None:
        try:
            image = vis_utils.load_image(im_file)
        except Exception as e:
            if not quiet:
                print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': run_detector.FAILURE_IMAGE_OPEN
            }
            return result

    try:
        
        result = detector.generate_detections_one_image(
                    image, 
                    im_file, 
                    detection_threshold=confidence_threshold, 
                    image_size=image_size,
                    skip_image_resizing=skip_image_resizing,
                    augment=augment)
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': run_detector.FAILURE_INFER
        }
        return result

    if isinstance(image,dict):
        image = image['img_original_pil']

    if include_image_size:        
        result['width'] = image.width
        result['height'] = image.height

    if include_image_timestamp:
        result['datetime'] = get_image_datetime(image)

    if include_exif_data:
        result['exif_metadata'] = read_exif.read_pil_exif(image,exif_options)

    return result

# ...def process_image(...)


def _load_custom_class_mapping(class_mapping_filename):
    """
    This is an experimental hack to allow the use of non-MD YOLOv5 models through
    the same infrastructure; it disables the code that enforces MDv5-like class lists.
    
    Should be a .json file that maps int-strings to strings, or a YOLOv5 dataset.yaml file.
    """
    
    if class_mapping_filename is None:
        return
    
    run_detector.USE_MODEL_NATIVE_CLASSES = True
    if class_mapping_filename.endswith('.json'):
        with open(class_mapping_filename,'r') as f:
            class_mapping = json.load(f)
    elif (class_mapping_filename.endswith('.yml') or class_mapping_filename.endswith('.yaml')):
        class_mapping = read_classes_from_yolo_dataset_file(class_mapping_filename)
        # convert from ints to int-strings
        class_mapping = {str(k):v for k,v in class_mapping.items()}
    else:
        raise ValueError('Unrecognized class mapping file {}'.format(class_mapping_filename))
        
    print('Loaded custom class mapping:')
    print(class_mapping)
    run_detector.DEFAULT_DETECTOR_LABEL_MAP = class_mapping
    return class_mapping
    
    
#%% Main function

def load_and_run_detector_batch(model_file, 
                                image_file_names, 
                                checkpoint_path=None,
                                confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                                checkpoint_frequency=-1, 
                                results=None, 
                                n_cores=1,
                                use_image_queue=False, 
                                quiet=False, 
                                image_size=None, 
                                class_mapping_filename=None, 
                                include_image_size=False, 
                                include_image_timestamp=False, 
                                include_exif_data=False,
                                augment=False,
                                force_model_download=False,
                                detector_options=None,
                                loader_workers=default_loaders,
                                preprocess_on_image_queue=default_preprocess_on_image_queue):
    """
    Load a model file and run it on a list of images.
    
    Args:
        
        model_file (str): path to model file, or supported model string (e.g. "MDV5A")
        image_file_names (list or str): list of strings (image filenames), a single image filename, 
            a folder to recursively search for images in, or a .json or .txt file containing a list 
            of images.
        checkpoint_path (str, optional), path to use for checkpoints (if None, checkpointing
            is disabled)
        confidence_threshold (float, optional): only detections above this threshold are returned
        checkpoint_frequency (int, optional): int, write results to JSON checkpoint file every N 
            images, -1 disabled checkpointing
        results (list, optional): list of dicts, existing results loaded from checkpoint; generally 
            not useful if you're using this function outside of the CLI
        n_cores (int, optional): number of parallel worker to use, ignored if we're running on a GPU
        use_image_queue (bool, optional): use a dedicated worker for image loading
        quiet (bool, optional): disable per-image console output
        image_size (tuple, optional): image size to use for inference, only mess with this
            if (a) you're using a model other than MegaDetector or (b) you know what you're
            doing
        class_mapping_filename (str, optional), use a non-default class mapping supplied in a .json 
            file or YOLOv5 dataset.yaml file
        include_image_size (bool, optional): should we include image size in the output for each image?
        include_image_timestamp (bool, optional): should we include image timestamps in the output for each image?
        include_exif_data (bool, optional): should we include EXIF data in the output for each image?
        augment (bool, optional): enable image augmentation
        force_model_download (bool, optional): force downloading the model file if
            a named model (e.g. "MDV5A") is supplied, even if the local file already
            exists
        detector_options (dict, optional): key/value pairs that are interpreted differently 
            by different detectors
        loader_workers (int, optional): number of loaders to use, only relevant when use_image_queue is True
        
    Returns:
        results: list of dicts; each dict represents detections on one image
    """
    
    # Validate input arguments
    if n_cores is None:
        n_cores = 1
    
    if confidence_threshold is None:
        confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
    
    # Disable checkpointing if checkpoint_path is None
    if checkpoint_frequency is None or checkpoint_path is None:
        checkpoint_frequency = -1

    if class_mapping_filename is not None:
        _load_custom_class_mapping(class_mapping_filename)
        
    # Handle the case where image_file_names is not yet actually a list
    if isinstance(image_file_names,str):
        
        # Find the images to score; images can be a directory, may need to recurse
        if os.path.isdir(image_file_names):
            image_dir = image_file_names
            image_file_names = path_utils.find_images(image_dir, True)
            print('{} image files found in folder {}'.format(len(image_file_names),image_dir))
            
        # A single file, or a list of image paths
        elif os.path.isfile(image_file_names):
            list_file = image_file_names
            if image_file_names.endswith('.json'):
                with open(list_file,'r') as f:
                    image_file_names = json.load(f)
                print('Loaded {} image filenames from .json list file {}'.format(
                    len(image_file_names),list_file))
            elif image_file_names.endswith('.txt'):
                with open(list_file,'r') as f:
                    image_file_names = f.readlines()
                    image_file_names = [s.strip() for s in image_file_names if len(s.strip()) > 0]
                print('Loaded {} image filenames from .txt list file {}'.format(
                    len(image_file_names),list_file))
            elif path_utils.is_image_file(image_file_names):
                image_file_names = [image_file_names]
                print('Processing image {}'.format(image_file_names[0]))
            else:
                raise ValueError(
                    'File {} supplied as [image_file_names] argument, but extension is neither .json nor .txt'\
                        .format(
                        list_file))
        else:            
            raise ValueError(
                '{} supplied as [image_file_names] argument, but it does not appear to be a file or folder'.format(
                    image_file_names))
            
    if results is None:
        results = []

    already_processed = set([i['file'] for i in results])

    model_file = try_download_known_detector(model_file, force_download=force_model_download)
        
    print('GPU available: {}'.format(is_gpu_available(model_file)))
    
    if n_cores > 1 and is_gpu_available(model_file):
        
        print('Warning: multiple cores requested, but a GPU is available; parallelization across ' + \
              'GPUs is not currently supported, defaulting to one GPU')
        n_cores = 1

    if n_cores > 1 and use_image_queue:
        
        print('Warning: multiple cores requested, but the image queue is enabled; parallelization ' + \
              'with the image queue is not currently supported, defaulting to one worker')
        n_cores = 1
        
    if use_image_queue:
        
        assert checkpoint_frequency < 0, \
            'Using an image queue is not currently supported when checkpointing is enabled'
        assert len(results) == 0, \
            'Using an image queue with results loaded from a checkpoint is not currently supported'
        assert n_cores <= 1
        results = run_detector_with_image_queue(image_file_names, 
                                                model_file, 
                                                confidence_threshold, 
                                                quiet, 
                                                image_size=image_size,
                                                include_image_size=include_image_size,
                                                include_image_timestamp=include_image_timestamp,
                                                include_exif_data=include_exif_data,
                                                augment=augment,
                                                detector_options=detector_options,
                                                loader_workers=loader_workers,
                                                preprocess_on_image_queue=preprocess_on_image_queue)
        
    elif n_cores <= 1:

        # Load the detector
        start_time = time.time()
        detector = load_detector(model_file,detector_options=detector_options,verbose=verbose)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

        # This is only used for console reporting, so it's OK that it doesn't
        # include images we might have loaded from a previous checkpoint
        count = 0

        for im_file in tqdm(image_file_names):

            # Will not add additional entries not in the starter checkpoint
            if im_file in already_processed:
                if not quiet:
                    print('Bypassing image {}'.format(im_file))
                continue

            count += 1

            result = process_image(im_file, 
                                   detector, 
                                   confidence_threshold, 
                                   quiet=quiet, 
                                   image_size=image_size, 
                                   include_image_size=include_image_size,
                                   include_image_timestamp=include_image_timestamp,
                                   include_exif_data=include_exif_data,
                                   augment=augment)
            results.append(result)

            # Write a checkpoint if necessary
            if (checkpoint_frequency != -1) and ((count % checkpoint_frequency) == 0):
                
                print('Writing a new checkpoint after having processed {} images since '
                      'last restart'.format(count))
                
                _write_checkpoint(checkpoint_path, results)
            
    else:
        
        # Multiprocessing is enabled at this point
        
        # When using multiprocessing, tell the workers to load the model on each
        # process, by passing the model_file string as the "model" argument to
        # process_images.
        detector = model_file

        print('Creating pool with {} cores'.format(n_cores))

        if len(already_processed) > 0:
            n_images_all = len(image_file_names)
            image_file_names = [fn for fn in image_file_names if fn not in already_processed]
            print('Loaded {} of {} images from checkpoint'.format(
                len(already_processed),n_images_all))
        
        # Divide images into chunks; we'll send one chunk to each worker process   
        image_batches = list(_chunks_by_number_of_chunks(image_file_names, n_cores))
                
        pool = workerpool(n_cores)

        if checkpoint_path is not None:
            
            # Multiprocessing and checkpointing are both enabled at this point
            
            checkpoint_queue = Manager().Queue()
            
            # Pass the "results" array (which may already contain images loaded from an existing
            # checkpoint) to the checkpoint queue handler function, which will append results to 
            # the list as they become available.
            checkpoint_thread = Thread(target=_checkpoint_queue_handler, 
                                       args=(checkpoint_path, checkpoint_frequency,
                                             checkpoint_queue, results), daemon=True)
            checkpoint_thread.start()

            pool.map(partial(process_images, 
                             detector=detector,
                             confidence_threshold=confidence_threshold,
                             use_image_queue=False,
                             quiet=quiet,
                             image_size=image_size, 
                             checkpoint_queue=checkpoint_queue,
                             include_image_size=include_image_size,
                             include_image_timestamp=include_image_timestamp,
                             include_exif_data=include_exif_data,
                             augment=augment,
                             detector_options=detector_options), 
                             image_batches)

            checkpoint_queue.put(None)

        else:
            
            # Multprocessing is enabled, but checkpointing is not
            
            new_results = pool.map(partial(process_images, 
                                           detector=detector,
                                           confidence_threshold=confidence_threshold,
                                           use_image_queue=False,
                                           quiet=quiet,
                                           checkpoint_queue=None,
                                           image_size=image_size,
                                           include_image_size=include_image_size,
                                           include_image_timestamp=include_image_timestamp,
                                           include_exif_data=include_exif_data,
                                           augment=augment,
                                           detector_options=detector_options), 
                                           image_batches)

            new_results = list(itertools.chain.from_iterable(new_results))
            
            # Append the results we just computed to "results", which is *usually* empty, but will
            # be non-empty if we resumed from a checkpoint
            results += new_results

        # ...if checkpointing is/isn't enabled
    
        try:
            pool.close()
        except Exception as e:
            print('Warning: error closing multiprocessing pool:\n{}'.format(str(e)))
                
    # ...if we're running (1) with image queue, (2) on one core, or (3) on multiple cores
    
    # 'results' may have been modified in place, but we also return it for
    # backwards-compatibility.
    return results

# ...def load_and_run_detector_batch(...)


def _checkpoint_queue_handler(checkpoint_path, checkpoint_frequency, checkpoint_queue, results):
    """
    Thread function to accumulate results and write checkpoints when checkpointing and
    multiprocessing are both enabled.
    """
    
    result_count = 0
    while True:
        result = checkpoint_queue.get()        
        if result is None:            
            break  
        
        result_count +=1
        results.append(result)

        if (checkpoint_frequency != -1) and (result_count % checkpoint_frequency == 0):
                
            print('Writing a new checkpoint after having processed {} images since '
                    'last restart'.format(result_count))
            
            _write_checkpoint(checkpoint_path, results)


def _write_checkpoint(checkpoint_path, results):
    """
    Writes the 'images' field in the dict 'results' to a json checkpoint file.
    """
    
    assert checkpoint_path is not None             
            
    # Back up any previous checkpoints, to protect against crashes while we're writing
    # the checkpoint file.
    checkpoint_tmp_path = None
    if os.path.isfile(checkpoint_path):
        checkpoint_tmp_path = checkpoint_path + '_tmp'
        shutil.copyfile(checkpoint_path,checkpoint_tmp_path)
        
    # Write the new checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump({'images': results}, f, indent=1, default=str)
        
    # Remove the backup checkpoint if it exists
    if checkpoint_tmp_path is not None:
        os.remove(checkpoint_tmp_path)


def get_image_datetime(image):
    """
    Reads EXIF datetime from a PIL Image object.
    
    Args:
        image (Image): the PIL Image object from which we should read datetime information
        
    Returns:
        str: the EXIF datetime from [image] (a PIL Image object), if available, as a string;
        returns None if EXIF datetime is not available.
    """
    
    exif_tags = read_exif.read_pil_exif(image,exif_options)
    
    try:
        datetime_str = exif_tags['DateTimeOriginal']
        _ = time.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        return datetime_str

    except Exception:
        return None        


def write_results_to_file(results, 
                          output_file, 
                          relative_path_base=None, 
                          detector_file=None, 
                          info=None, 
                          include_max_conf=False,
                          custom_metadata=None, 
                          force_forward_slashes=True):
    """
    Writes list of detection results to JSON output file. Format matches:

    https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing#batch-processing-api-output-format

    Args:
        results (list): list of dict, each dict represents detections on one image
        output_file (str): path to JSON output file, should end in '.json'
        relative_path_base (str, optional): path to a directory as the base for relative paths, can
            be None if the paths in [results] are absolute
        detector_file (str, optional): filename of the detector used to generate these results, only
            used to pull out a version number for the "info" field
        info (dict, optional): dictionary to put in the results file instead of the default "info" field
        include_max_conf (bool, optional): old files (version 1.2 and earlier) included a "max_conf" field
            in each image; this was removed in version 1.3.  Set this flag to force the inclusion
            of this field.
        custom_metadata (object, optional): additional data to include as info['custom_metadata']; typically
            a dictionary, but no type/format checks are performed
        force_forward_slashes (bool, optional): convert all slashes in filenames within [results] to
            forward slashes
                    
    Returns:
        dict: the MD-formatted dictionary that was written to [output_file]
    """
    
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    if force_forward_slashes:
        results_converted = []
        for r in results:
            r_converted = copy.copy(r)
            r_converted['file'] = r_converted['file'].replace('\\','/')
            results_converted.append(r_converted)
        results = results_converted
            
    # The typical case: we need to build the 'info' struct
    if info is None:
        
        info = { 
            'detection_completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.4' 
        }
        
        if detector_file is not None:
            detector_filename = os.path.basename(detector_file)
            detector_version = get_detector_version_from_filename(detector_filename)
            detector_metadata = get_detector_metadata_from_version_string(detector_version)
            info['detector'] = detector_filename  
            info['detector_metadata'] = detector_metadata
        else:
            info['detector'] = 'unknown'
            info['detector_metadata'] = get_detector_metadata_from_version_string('unknown')
        
    # If the caller supplied the entire "info" struct
    else:
        
        if detector_file is not None:            
            print('Warning (write_results_to_file): info struct and detector file ' + \
                  'supplied, ignoring detector file')

    if custom_metadata is not None:
        info['custom_metadata'] = custom_metadata
        
    # The 'max_detection_conf' field used to be included by default, and it caused all kinds
    # of headaches, so it's no longer included unless the user explicitly requests it.
    if not include_max_conf:
        for im in results:
            if 'max_detection_conf' in im:
                del im['max_detection_conf']
        
    # Sort results by filename; not required by the format, but convenient for consistency
    results = sort_list_of_dicts_by_key(results,'file')
    
    # Sort detections in descending order by confidence; not required by the format, but
    # convenient for consistency
    for r in results:
        if ('detections' in r) and (r['detections'] is not None):
            r['detections'] = sort_list_of_dicts_by_key(r['detections'], 'conf', reverse=True)
            
    final_output = {
        'images': results,
        'detection_categories': run_detector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': info
    }
    
    # Create the folder where the output file belongs; this will fail if
    # this is a relative path with no folder component
    try:
        os.makedirs(os.path.dirname(output_file),exist_ok=True)
    except Exception:
        pass
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1, default=str)
    print('Output file saved at {}'.format(output_file))
    
    return final_output

# ...def write_results_to_file(...)


#%% Interactive driver

if False:
    
    pass

    #%%
    
    model_file = 'MDV5A'
    image_dir = r'g:\camera_traps\camera_trap_images'
    output_file = r'g:\temp\md-test.json'
    
    recursive = True
    output_relative_filenames = True
    include_max_conf = False
    quiet = True
    image_size = None
    use_image_queue = False
    confidence_threshold = 0.0001
    checkpoint_frequency = 5   
    checkpoint_path = None
    resume_from_checkpoint = 'auto'
    allow_checkpoint_overwrite = False
    ncores = 1
    class_mapping_filename = None
    include_image_size = True
    include_image_timestamp = True
    include_exif_data = True
    overwrite_handling = None
        
    # Generate a command line
    cmd = 'python run_detector_batch.py "{}" "{}" "{}"'.format(
        model_file,image_dir,output_file)
    
    if recursive:
        cmd += ' --recursive'
    if output_relative_filenames:
        cmd += ' --output_relative_filenames'
    if include_max_conf:
        cmd += ' --include_max_conf'
    if quiet:
        cmd += ' --quiet'
    if image_size is not None:
        cmd += ' --image_size {}'.format(image_size)
    if use_image_queue:
        cmd += ' --use_image_queue'
    if confidence_threshold is not None:
        cmd += ' --threshold {}'.format(confidence_threshold)
    if checkpoint_frequency is not None:
        cmd += ' --checkpoint_frequency {}'.format(checkpoint_frequency)
    if checkpoint_path is not None:
        cmd += ' --checkpoint_path "{}"'.format(checkpoint_path)
    if resume_from_checkpoint is not None:
        cmd += ' --resume_from_checkpoint "{}"'.format(resume_from_checkpoint)
    if allow_checkpoint_overwrite:
        cmd += ' --allow_checkpoint_overwrite'
    if ncores is not None:
        cmd += ' --ncores {}'.format(ncores)
    if class_mapping_filename is not None:
        cmd += ' --class_mapping_filename "{}"'.format(class_mapping_filename)
    if include_image_size:
        cmd += ' --include_image_size'
    if include_image_timestamp:
        cmd += ' --include_image_timestamp'
    if include_exif_data:
        cmd += ' --include_exif_data'
    if overwrite_handling is not None:
        cmd += ' --overwrite_handling {}'.format(overwrite_handling)
    
    print(cmd)
    import clipboard; clipboard.copy(cmd)
    
    
    #%% Run inference interactively
    
    image_file_names = path_utils.find_images(image_dir, recursive=False)    
    results = None
    
    start_time = time.time()
    
    results = load_and_run_detector_batch(model_file=model_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=confidence_threshold,
                                          checkpoint_frequency=checkpoint_frequency,
                                          results=results,
                                          n_cores=ncores,
                                          use_image_queue=use_image_queue,
                                          quiet=quiet,
                                          image_size=image_size)
    
    elapsed = time.time() - start_time
    
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

    
#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Module to run a TF/PT animal detection model on lots of images')
    parser.add_argument(
        'detector_file',
        help='Path to detector model file (.pb or .pt).  Can also be the strings "MDV4", "MDV5A", or "MDV5B" to request automatic download.')
    parser.add_argument(
        'image_file',
        help=\
        'Path to a single image file, a .json or .txt file containing a list of paths to images, or a directory')
    parser.add_argument(
        'output_file',
        help='Path to output JSON results file, should end with a .json extension')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--output_relative_filenames',
        action='store_true',
        help='Output relative file names, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--include_max_conf',
        action='store_true',
        help='Include the "max_detection_conf" field in the output')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-image console output')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable additional debug output')
    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help=('Force image resizing to a specific integer size on the long axis (not recommended to change this)'))    
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable image augmentation'
    )
    parser.add_argument(
        '--use_image_queue',
        action='store_true',
        help='Pre-load images, may help keep your GPU busy; does not currently support ' + \
             'checkpointing.  Useful if you have a very fast GPU and a very slow disk.')
    parser.add_argument(
        '--preprocess_on_image_queue',
        action='store_true',
        help='Whether to do image resizing on the image queue (PyTorch detectors only)')
    parser.add_argument(
        '--use_threads_for_queue',
        action='store_true',
        help='Use threads (rather than processes) for the image queue; only relevant if --use_image_queue is set')
    parser.add_argument(
        '--threshold',
        type=float,
        default=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.0, don't include boxes below this " + \
            "confidence in the output file. Default is {}".format(
                run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD))
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=-1,
        help='Write results to a temporary file every N images; default is -1, which ' + \
             'disables this feature')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='File name to which checkpoints will be written if checkpoint_frequency is > 0, ' + \
             'defaults to md_checkpoint_[date].json in the same folder as the output file')    
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to a JSON checkpoint file to resume from, or "auto" to ' + \
             'find the most recent checkpoint in the same folder as the output file.  "auto" uses' + \
             'checkpoint_path (rather than searching the output folder) if checkpoint_path is specified.')
    parser.add_argument(
        '--allow_checkpoint_overwrite',
        action='store_true',
        help='By default, this script will bail if the specified checkpoint file ' + \
              'already exists; this option allows it to overwrite existing checkpoints')
    parser.add_argument(
        '--ncores',
        type=int,
        default=0,
        help='Number of cores to use for inference; only applies to CPU-based inference')
    parser.add_argument(
        '--loader_workers',
        type=int,
        default=default_loaders,
        help='Number of image loader workers to use; only relevant when --use_image_queue is set')
    parser.add_argument(
        '--class_mapping_filename',
        type=str,
        default=None,
        help='Use a non-default class mapping, supplied in a .json file with a dictionary mapping' + \
            'int-strings to strings.  This will also disable the addition of "1" to all category ' + \
            'IDs, so your class mapping should start at zero.  Can also be a YOLOv5 dataset.yaml file.')
    parser.add_argument(
        '--include_image_size',
        action='store_true',
        help='Include image dimensions in output file'
    )
    parser.add_argument(
        '--include_image_timestamp',
        action='store_true',
        help='Include image datetime (if available) in output file'
    )
    parser.add_argument(
        '--include_exif_data',
        action='store_true',
        help='Include available EXIF data in output file'
    )
    parser.add_argument(
        '--overwrite_handling',
        type=str,
        default='overwrite',
        help='What should we do if the output file exists?  overwrite/skip/error (default overwrite)'
    )    
    parser.add_argument(
        '--force_model_download',
        action='store_true',
        help=('If a named model (e.g. "MDV5A") is supplied, force a download of that model even if the ' +\
              'local file already exists.'))
    parser.add_argument(
        '--previous_results_file',
        type=str,
        default=None,
        help=('If supplied, this should point to a previous .json results file; any results in that ' +\
              'file will be transferred to the output file without reprocessing those images.  Useful ' +\
              'for "updating" a set of results when you may have added new images to a folder you\'ve ' +\
              'already processed.  Only supported when using relative paths.'))
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
    
    global verbose
    global use_threads_for_queue
    
    if args.verbose:
        verbose = True
    if args.use_threads_for_queue:
        use_threads_for_queue = True
        
    detector_options = parse_kvp_list(args.detector_options)
    
    # If the specified detector file is really the name of a known model, find 
    # (and possibly download) that model
    args.detector_file = try_download_known_detector(args.detector_file, 
                                                     force_download=args.force_model_download)
    
    assert os.path.exists(args.detector_file), \
        'detector file {} does not exist'.format(args.detector_file)
    assert 0.0 <= args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'
    assert args.output_file.endswith('.json'), 'output_file specified needs to end with .json'
    if args.checkpoint_frequency != -1:
        assert args.checkpoint_frequency > 0, 'Checkpoint_frequency needs to be > 0 or == -1'
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), \
            f'Could not find folder {args.image_file}, must supply a folder when ' + \
                '--output_relative_filenames is set'
    if args.previous_results_file is not None:
        assert os.path.isdir(args.image_file) and args.output_relative_filenames, \
            "Can only process previous results when using relative paths"
    if os.path.exists(args.output_file):
        if args.overwrite_handling == 'overwrite':
            print('Warning: output file {} already exists and will be overwritten'.format(
                args.output_file))
        elif args.overwrite_handling == 'skip':
            print('Output file {} exists, returning'.format(
                args.output_file))
            return
        elif args.overwrite_handling == 'error':
            raise Exception('Output file {} exists'.format(args.output_file))
        else:
            raise ValueError('Illegal overwrite handling string {}'.format(args.overwrite_handling))

    output_dir = os.path.dirname(args.output_file)

    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
        
    assert not os.path.isdir(args.output_file), 'Specified output file is a directory'
    
    if args.class_mapping_filename is not None:
        _load_custom_class_mapping(args.class_mapping_filename)
    
    # Load the checkpoint if available
    #
    # File paths in the checkpoint are always absolute paths; conversion to relative paths
    # happens below (if necessary).
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == 'auto':
            checkpoint_files = os.listdir(output_dir)
            checkpoint_files = [fn for fn in checkpoint_files if \
                                (fn.startswith('md_checkpoint') and fn.endswith('.json'))]
            if len(checkpoint_files) == 0:
                raise ValueError('resume_from_checkpoint set to "auto", but no checkpoints found in {}'.format(
                    output_dir))
            else:
                if len(checkpoint_files) > 1:
                    print('Warning: found {} checkpoints in {}, using the latest'.format(
                        len(checkpoint_files),output_dir))
                    checkpoint_files = sorted(checkpoint_files)
                checkpoint_file_relative = checkpoint_files[-1]
                checkpoint_file = os.path.join(output_dir,checkpoint_file_relative)                                
        else:
            checkpoint_file = args.resume_from_checkpoint
        assert os.path.exists(checkpoint_file), \
            'File at resume_from_checkpoint specified does not exist'
        with open(checkpoint_file) as f:
            print('Loading previous results from checkpoint file {}'.format(
                checkpoint_file))
            saved = json.load(f)
        assert 'images' in saved, \
            'The checkpoint file does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []

    # Find the images to process; images can be a directory, may need to recurse
    if os.path.isdir(args.image_file):
        image_file_names = path_utils.find_images(args.image_file, args.recursive)
        if len(image_file_names) > 0:
            print('{} image files found in the input directory'.format(len(image_file_names)))                        
        else:
            if (args.recursive):
                print('No image files found in directory {}, exiting'.format(args.image_file))
            else:
                print('No image files found in directory {}, did you mean to specify '
                      '--recursive?'.format(
                    args.image_file))
            return
        
    # A json list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.json'):        
        with open(args.image_file) as f:
            image_file_names = json.load(f)
        print('Loaded {} image filenames from .json list file {}'.format(
            len(image_file_names),args.image_file))
    
    # A text list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.txt'):
        with open(args.image_file) as f:
            image_file_names = f.readlines()
            image_file_names = [fn.strip() for fn in image_file_names if len(fn.strip()) > 0]
        print('Loaded {} image filenames from .txt list file {}'.format(
            len(image_file_names),args.image_file))
        
    # A single image file
    elif os.path.isfile(args.image_file) and path_utils.is_image_file(args.image_file):
        image_file_names = [args.image_file]
        print('Processing image {}'.format(args.image_file))
        
    else:        
        raise ValueError('image_file specified is not a directory, a json list, or an image file, '
                         '(or does not have recognizable extensions).')

    # At this point, regardless of how they were specified, [image_file_names] is a list of 
    # absolute image paths.
    assert len(image_file_names) > 0, 'Specified image_file does not point to valid image files'
    
    # Convert to forward slashes to facilitate comparison with previous results
    image_file_names = [fn.replace('\\','/') for fn in image_file_names]
    
    # We can head off many problems related to incorrect command line formulation if we confirm
    # that one image exists before proceeding.  The use of the first image for this test is 
    # arbitrary.
    assert os.path.exists(image_file_names[0]), \
        'The first image to be processed does not exist at {}'.format(image_file_names[0])

    # Possibly load results from a previous pass
    previous_results = None
    
    if args.previous_results_file is not None:
        
        assert os.path.isfile(args.previous_results_file), \
            'Could not find previous results file {}'.format(args.previous_results_file)
        with open(args.previous_results_file,'r') as f:
            previous_results = json.load(f)
                
        assert previous_results['detection_categories'] == run_detector.DEFAULT_DETECTOR_LABEL_MAP, \
            "Can't merge previous results when those results use a different set of detection categories"
        
        print('Loaded previous results for {} images from {}'.format(
            len(previous_results['images']), args.previous_results_file))
        
        # Convert previous result filenames to absolute paths if necessary 
        #
        # We asserted above to make sure that we are using relative paths and processing a 
        # folder, but just to be super-clear...
        assert os.path.isdir(args.image_file)
        
        previous_image_files_set = set()
        for im in previous_results['images']:
            assert not os.path.isabs(im['file']), \
                "When processing previous results, relative paths are required"
            fn_abs = os.path.join(args.image_file,im['file']).replace('\\','/')
            # Absolute paths are expected at the final output stage below
            im['file'] = fn_abs
            previous_image_files_set.add(fn_abs)            
        
        image_file_names_to_keep = []
        for fn_abs in image_file_names:
            if fn_abs not in previous_image_files_set:
                image_file_names_to_keep.append(fn_abs)
                
        print('Based on previous results file, processing {} of {} images'.format(
            len(image_file_names_to_keep), len(image_file_names)))
        
        image_file_names = image_file_names_to_keep
        
    # ...if we're handling previous results
        
    # Test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        
        if args.checkpoint_path is not None:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = os.path.join(output_dir,
                                           'md_checkpoint_{}.json'.format(
                                               datetime.now().strftime("%Y%m%d%H%M%S")))
        
        # Don't overwrite existing checkpoint files, this is a sure-fire way to eventually
        # erase someone's checkpoint.
        if (checkpoint_path is not None) and (not args.allow_checkpoint_overwrite) \
            and (args.resume_from_checkpoint is None):
            
            assert not os.path.isfile(checkpoint_path), \
                f'Checkpoint path {checkpoint_path} already exists, delete or move it before ' + \
                're-using the same checkpoint path, or specify --allow_checkpoint_overwrite'

        
        # Confirm that we can write to the checkpoint path; this avoids issues where
        # we crash after several thousand images.
        #
        # But actually, commenting this out for now... the scenario where we are resuming from a 
        # checkpoint, then immediately overwrite that checkpoint with empty data is higher-risk
        # than the annoyance of crashing a few minutes after starting a job.
        if False:
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': []}, f)
                
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
        
    else:
        
        if args.checkpoint_path is not None:
            print('Warning: checkpointing disabled because checkpoint_frequency is -1, ' + \
                  'but a checkpoint path was specified')
        checkpoint_path = None

    start_time = time.time()

    results = load_and_run_detector_batch(model_file=args.detector_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=args.threshold,
                                          checkpoint_frequency=args.checkpoint_frequency,
                                          results=results,
                                          n_cores=args.ncores,
                                          use_image_queue=args.use_image_queue,
                                          quiet=args.quiet,
                                          image_size=args.image_size,
                                          class_mapping_filename=args.class_mapping_filename,
                                          include_image_size=args.include_image_size,
                                          include_image_timestamp=args.include_image_timestamp,
                                          include_exif_data=args.include_exif_data,
                                          augment=args.augment,
                                          # Don't download the model *again*
                                          force_model_download=False,
                                          detector_options=detector_options,
                                          loader_workers=args.loader_workers,
                                          preprocess_on_image_queue=args.preprocess_on_image_queue)

    elapsed = time.time() - start_time
    images_per_second = len(results) / elapsed
    print('Finished inference for {} images in {} ({:.2f} images per second)'.format(
        len(results),humanfriendly.format_timespan(elapsed),images_per_second))

    relative_path_base = None
    
    # We asserted above to make sure that if output_relative_filenames is set, 
    # args.image_file is a folder, but we'll double-check for clarity.
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file)
        relative_path_base = args.image_file
    
    # Merge results from a previous file if necessary
    if previous_results is not None:
        previous_filenames_set = set([im['file'] for im in previous_results['images']])
        new_filenames_set = set([im['file'] for im in results])
        assert len(previous_filenames_set.intersection(new_filenames_set)) == 0, \
            'Previous results handling error: redundant image filenames'
        results.extend(previous_results['images'])        
        
    write_results_to_file(results, 
                          args.output_file, 
                          relative_path_base=relative_path_base,
                          detector_file=args.detector_file,
                          include_max_conf=args.include_max_conf)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
        print('Deleted checkpoint file {}'.format(checkpoint_path))

    print('Done, thanks for MegaDetect\'ing!')

if __name__ == '__main__':
    main()
