"""

run_md_and_speciesnet.py

Module to run MegaDetector followed by SpeciesNet classifier on images and videos.

This script simplifies the SpeciesNet pipeline by:
- Running MegaDetector and SpeciesNet classification in separate, sequential steps
- Supporting multiple detections per image with individual classification
- Using the standard MegaDetector output format
- Supporting both images and videos
- Using a simplified multiprocessing architecture

"""

#%% Constants, imports, environment

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from copy import deepcopy
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import humanfriendly
from PIL import Image
from tqdm import tqdm

from megadetector.detection import run_detector_batch
from megadetector.detection.run_detector_batch import load_and_run_detector_batch
from megadetector.detection.run_detector_batch import write_results_to_file
from megadetector.utils import path_utils
from megadetector.visualization import visualization_utils as vis_utils

import speciesnet
from speciesnet import SpeciesNetClassifier
from speciesnet.ensemble import SpeciesNetEnsemble
from speciesnet.geofence_utils import roll_up_labels_to_first_matching_level
from speciesnet.geofence_utils import geofence_animal_classification
from speciesnet.utils import BBox


#%% Constants

DEFAULT_DETECTOR_MODEL = 'MDV5A'
DEFAULT_CLASSIFIER_MODEL = 'kaggle:google/speciesnet/pyTorch/v4.0.1a'
DEFAULT_DETECTION_CONFIDENCE_THRESHOLD = 0.1
DEFAULT_DETECTOR_BATCH_SIZE = 1
DEFAULT_CLASSIFIER_BATCH_SIZE = 8
DEFAULT_CLASSIFIER_WORKER_THREADS = 4

verbose = False


#%% Support classes

class CropMetadata:
    """
    Metadata for a crop extracted from an image detection.
    """

    def __init__(self,
                 image_file: str,
                 detection_index: int,
                 bbox: List[float],
                 original_width: int,
                 original_height: int):
        """
        Args:
            image_file (str): path to the original image file
            detection_index (int): index of this detection in the image
            bbox (List[float]): normalized bounding box [x_min, y_min, width, height]
            original_width (int): width of the original image
            original_height (int): height of the original image
        """

        self.image_file = image_file
        self.detection_index = detection_index
        self.bbox = bbox
        self.original_width = original_width
        self.original_height = original_height


class CropBatch:
    """
    A batch of crops with their metadata for classification.
    """

    def __init__(self):
        self.crops = []  # List of preprocessed images
        self.metadata = []  # List of CropMetadata objects

    def add_crop(self, crop_data, metadata: CropMetadata):
        """
        Args:
            crop_data: preprocessed image data from SpeciesNetClassifier.preprocess()
            metadata (CropMetadata): metadata for this crop
        """

        self.crops.append(crop_data)
        self.metadata.append(metadata)

    def __len__(self):
        return len(self.crops)


#%% Support functions for argument parsing

def _str_to_bool(v):
    """
    Convert string to boolean.
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


#%% Support functions for classification

def _crop_producer_func(image_queue: JoinableQueue,
                        batch_queue: Queue,
                        classifier_model: str,
                        detection_confidence_threshold: float,
                        source_folder: str,
                        producer_id: int = -1):
    """
    Producer function for classification workers.

    Reads images from the image_queue, crops detections above threshold,
    preprocesses them, and immediately sends individual crops to batch_queue.
    See the documentation of _crop_consumer_func to for the format of the
    tuples placed on batch_queue.

    Args:
        image_queue (JoinableQueue): queue containing detection_results dicts
        batch_queue (Queue): queue to put individual crops into
        classifier_model (str): classifier model identifier to load in this process
        detection_confidence_threshold (float): minimum confidence for detections to process
        source_folder (str): source folder to resolve relative paths
        producer_id (int, optional): identifier for this producer worker
    """

    if verbose:
        print(f'Classification producer starting: ID {producer_id}')

    # Load classifier in this process
    try:
        classifier = SpeciesNetClassifier(classifier_model)
        if verbose:
            print(f'Classification producer {producer_id}: loaded classifier')
    except Exception as e:
        print(f'Classification producer {producer_id}: failed to load classifier: {str(e)}')
        # Send sentinel to indicate this producer failed to start
        batch_queue.put(None)
        return

    while True:

        try:
            detection_results = image_queue.get()

            # Pulling None from the queue indicates that this producer is done
            if detection_results is None:
                image_queue.task_done()
                break

            image_file = detection_results['file']

            if verbose:
                print(f'Processing {image_file} on producer {producer_id}')

            # Skip images with failures
            if 'failure' in detection_results:
                image_queue.task_done()
                continue

            # Skip images with no detections
            detections = detection_results.get('detections', [])
            if not detections:
                image_queue.task_done()
                continue

            # Load the image
            try:
                # The image_file from detection results is relative to source folder
                absolute_image_path = os.path.join(source_folder, image_file)
                image = vis_utils.load_image(absolute_image_path)
                original_width, original_height = image.size
            except Exception as e:
                if verbose:
                    print(f'Failed to load image {image_file}: {str(e)}')
                # Send failure information to consumer
                failure_metadata = CropMetadata(
                    image_file=image_file,
                    detection_index=-1,  # -1 indicates whole-image failure
                    bbox=[],
                    original_width=0,
                    original_height=0
                )
                batch_queue.put(('failure', f'Failed to load image: {str(e)}', failure_metadata))
                image_queue.task_done()
                continue

            # Process each detection above threshold
            for detection_index, detection in enumerate(detections):

                conf = detection.get('conf', 0.0)
                if conf < detection_confidence_threshold:
                    continue

                bbox = detection['bbox']
                assert len(bbox) == 4

                # Convert normalized bbox to BBox object for SpeciesNet
                speciesnet_bbox = BBox(
                    xmin=bbox[0],
                    ymin=bbox[1],
                    width=bbox[2],
                    height=bbox[3]
                )

                # Preprocess the crop
                try:
                    preprocessed_crop = classifier.preprocess(
                        image,
                        bboxes=[speciesnet_bbox],
                        resize=True
                    )

                    if preprocessed_crop is not None:
                        metadata = CropMetadata(
                            image_file=image_file,
                            detection_index=detection_index,
                            bbox=bbox,
                            original_width=original_width,
                            original_height=original_height
                        )

                        # Send individual crop immediately to consumer
                        batch_queue.put(('crop', preprocessed_crop, metadata))

                except Exception as e:
                    if verbose:
                        print(f'Failed to preprocess crop from {image_file}, '
                              f'detection {detection_index}: {str(e)}')

                    # Send failure information to consumer
                    failure_metadata = CropMetadata(
                        image_file=image_file,
                        detection_index=detection_index,
                        bbox=bbox,
                        original_width=original_width,
                        original_height=original_height
                    )
                    batch_queue.put(('failure', f'Failed to preprocess crop: {str(e)}', failure_metadata))
                    continue

            # ...for each detection in this image

            image_queue.task_done()

        except Exception as e:

            print(f'**** Producer {producer_id} error: {str(e)} ****')
            # Try to send failure information if we can determine the image file
            try:
                if 'detection_results' in locals() and detection_results is not None:
                    image_file = detection_results.get('file', 'unknown')
                    failure_metadata = CropMetadata(
                        image_file=image_file,
                        detection_index=-1,
                        bbox=[],
                        original_width=0,
                        original_height=0
                    )
                    batch_queue.put(('failure', f'Producer error: {str(e)}', failure_metadata))
            except:
                pass  # If we can't send failure info, just continue

            image_queue.task_done()
            continue

        # ...try/except

    # ...while(we still have images to process)

    # Send sentinel to indicate this producer is done
    batch_queue.put(None)

    if verbose:
        print(f'Classification producer {producer_id} finished')

# ...def _crop_producer_func(...)


def _crop_consumer_func(batch_queue: Queue,
                        results_queue: Queue,
                        classifier_model: str,
                        batch_size: int,
                        num_producers: int,
                        enable_rollup: bool,
                        country: Optional[str]=None,
                        admin1_region: Optional[str]=None):
    """
    Consumer function for classification inference.

    Pulls individual crops from batch_queue, assembles them into batches,
    runs inference, and puts results into results_queue.

    Args:
        batch_queue (Queue): queue containing individual crop tuples or failures.
            Items on this queue are either None (to indicate that a producer finished)
            or tuples formatted as (type,image,metadata).  [type] is a string (either
            "crop" or "failure"), [image] is a PreprocessedImage, and [metadata] is
            a CropMetadata object.
        results_queue (Queue): queue to put classification results into
        classifier_model (str): classifier model identifier to load in this process
        batch_size (int): batch size for inference
        num_producers (int): number of producer workers
        enable_rollup (bool): whether to apply taxonomic rollup
        country (str, optional): country code for geofencing
        admin1_region (str, optional): admin1 region for geofencing
    """

    if verbose:
        print('Classification consumer starting')

    # Load classifier in this process
    try:
        classifier = SpeciesNetClassifier(classifier_model)
        if verbose:
            print('Classification consumer: loaded classifier')
    except Exception as e:
        print(f'Classification consumer: failed to load classifier: {str(e)}')
        results_queue.put({})
        return

    all_results = {}  # image_file -> {detection_index -> classification_result}
    current_batch = CropBatch()
    producers_finished = 0

    # Load ensemble metadata if rollup/geofencing is enabled
    taxonomy_map = {}
    geofence_map = {}
    if (enable_rollup is not None) or (country is not None):
        try:
            # Use the model name string directly instead of model_info.name
            model_name = classifier_model if isinstance(
                classifier_model, str) else str(classifier_model)
            ensemble = SpeciesNetEnsemble(
                model_name, geofence=(country is not None))
            taxonomy_map = ensemble.taxonomy_map
            geofence_map = ensemble.geofence_map
        except Exception as e:
            print(
                f'Warning: failed to load ensemble metadata for rollup/geofencing: {str(e)}')
            enable_rollup = False
            country = None

    while True:

        try:
            item = batch_queue.get()

            # Sentinel signal - a producer finished
            if item is None:
                producers_finished += 1
                if producers_finished == num_producers:
                    # Process any remaining batch
                    if len(current_batch) > 0:
                        _process_classification_batch(
                            current_batch, classifier, all_results,
                            enable_rollup, taxonomy_map, geofence_map,
                            country, admin1_region
                        )
                    break
                continue

            # Handle different item types
            if isinstance(item, tuple) and len(item) == 3:
                item_type, data, metadata = item

                if item_type == 'failure':
                    # Handle failure - record it in results
                    if metadata.image_file not in all_results:
                        all_results[metadata.image_file] = {}

                    all_results[metadata.image_file][metadata.detection_index] = {
                        'failure': f'Failure classification: {data}'
                    }

                elif item_type == 'crop':
                    # Handle successful crop - add to current batch
                    current_batch.add_crop(data, metadata)

                    # Process batch if it's full
                    if len(current_batch) >= batch_size:
                        _process_classification_batch(
                            current_batch, classifier, all_results,
                            enable_rollup, taxonomy_map, geofence_map,
                            country, admin1_region
                        )
                        current_batch = CropBatch()

            else:
                print(f'Warning: unexpected item format in batch_queue: {type(item)}')
                continue

        except Exception as e:
            print(f'Classification consumer error: {str(e)}')
            continue

    results_queue.put(all_results)

    if verbose:
        print('Classification consumer finished')

# ...def _crop_consumer_func(...)


def _process_classification_batch(batch: CropBatch,
                                  classifier: 'SpeciesNetClassifier',
                                  all_results: Dict,
                                  enable_rollup: bool,
                                  taxonomy_map: Dict,
                                  geofence_map: Dict,
                                  country: Optional[str],
                                  admin1_region: Optional[str]):
    """
    Process a batch of crops through classification.

    Args:
        batch (CropBatch): batch of crops to process
        classifier (SpeciesNetClassifier): classifier instance
        all_results (Dict): dictionary to store results in
        enable_rollup (bool): whether to apply rollup
        taxonomy_map (Dict): taxonomy mapping for rollup
        geofence_map (Dict): geofence mapping
        country (Optional[str]): country code for geofencing
        admin1_region (Optional[str]): admin1 region for geofencing
    """

    if len(batch) == 0:
        return

    # Prepare batch for inference
    filepaths = [f"{metadata.image_file}_{metadata.detection_index}"
                 for metadata in batch.metadata]

    # Run batch inference
    try:
        batch_results = classifier.batch_predict(filepaths, batch.crops)
    except Exception as e:
        print(f'Batch classification failed: {str(e)}')
        # Mark all crops in this batch as failed
        for metadata in batch.metadata:
            if metadata.image_file not in all_results:
                all_results[metadata.image_file] = {}
            all_results[metadata.image_file][metadata.detection_index] = {
                'failure': f'Failure classification: {str(e)}'
            }
        return

    # Process results
    for result, metadata in zip(batch_results, batch.metadata):

        if metadata.image_file not in all_results:
            all_results[metadata.image_file] = {}

        detection_index = metadata.detection_index

        # Handle classification failure
        if 'failures' in result:
            all_results[metadata.image_file][detection_index] = {
                'failure': 'Failure classification: SpeciesNet classifier failed'
            }
            continue

        # Extract classification results
        classifications = result.get('classifications', {})
        classes = classifications.get('classes', [])
        scores = classifications.get('scores', [])

        if not classes or not scores:
            all_results[metadata.image_file][detection_index] = {
                'failure': 'Failure classification: No valid classifications returned'
            }
            continue

        # Apply rollup and/or geofencing if enabled
        final_classes = classes
        final_scores = scores

        if enable_rollup or country:
            try:
                # Apply rollup
                if enable_rollup:
                    rollup_result = roll_up_labels_to_first_matching_level(
                        labels=classes,
                        scores=scores,
                        country=country,
                        admin1_region=admin1_region,
                        target_taxonomy_levels=[
                            'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'],
                        non_blank_threshold=0.1,  # Hard-coded as mentioned
                        taxonomy_map=taxonomy_map,
                        geofence_map=geofence_map,
                        enable_geofence=(country is not None)
                    )

                    if rollup_result is not None:
                        rolled_up_class, rolled_up_score, _ = rollup_result
                        # Replace the top prediction with rolled-up result
                        final_classes = [rolled_up_class] + classes[1:]
                        final_scores = [rolled_up_score] + scores[1:]

                # Apply geofencing
                if country:
                    geofence_result = geofence_animal_classification(
                        labels=final_classes,
                        scores=final_scores,
                        country=country,
                        admin1_region=admin1_region,
                        taxonomy_map=taxonomy_map,
                        geofence_map=geofence_map,
                        enable_geofence=True
                    )

                    if geofence_result is not None:
                        geofenced_class, geofenced_score, _ = geofence_result
                        # Replace the top prediction with geofenced result
                        final_classes = [geofenced_class] + final_classes[1:]
                        final_scores = [geofenced_score] + final_scores[1:]

            except Exception as e:
                if verbose:
                    print(f'Warning: rollup/geofencing failed for {metadata.image_file}, '
                          f'detection {detection_index}: {str(e)}')

        # Store result in MegaDetector format
        # Classifications are stored as [category_id, confidence] pairs
        # For now, we'll create a simple mapping of class names to string IDs
        classification_pairs = []
        for class_name, score in zip(final_classes, final_scores):
            # Use class name as category ID (this could be improved with a proper mapping)
            classification_pairs.append([str(class_name), float(score)])

        all_results[metadata.image_file][detection_index] = {
            'classifications': classification_pairs
        }

# ...def _process_classification_batch(...)


#%% Inference functions

def run_detection_step(source_folder: str,
                      detector_model: str,
                      detector_batch_size: int,
                      detection_confidence_threshold: float,
                      temp_folder: str) -> str:
    """
    Run MegaDetector on all images/videos in source_folder.

    Args:
        source_folder (str): folder containing images/videos
        detector_model (str): detector model identifier
        detector_batch_size (int): batch size for detection
        detection_confidence_threshold (float): confidence threshold for detections
        temp_folder (str): folder for temporary files

    Returns:
        str: path to temporary file containing detection results
    """

    if verbose:
        print('Starting MegaDetector detection step...')

    # Create temporary file for detection results
    detector_output_file = os.path.join(temp_folder, 'detector_results.json')

    # Find all image and video files
    image_extensions = path_utils.IMG_EXTENSIONS
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.flv', '.wmv')
    all_extensions = image_extensions + video_extensions

    image_files = path_utils.find_images(source_folder, recursive=True)

    if len(image_files) == 0:
        raise ValueError(f'No images or videos found in {source_folder}')

    if verbose:
        print(f'Found {len(image_files)} files to process')

    # For now, only handle images - video support can be added later
    video_files = [f for f in image_files if any(
        f.lower().endswith(ext) for ext in video_extensions)]
    if len(video_files) > 0:
        print(
            f'Warning: found {len(video_files)} video files, but video processing is not yet implemented')
        image_files = [f for f in image_files if f not in video_files]

    if len(image_files) == 0:
        raise ValueError('No supported image files found')

    # Run MegaDetector
    print(f'Running MegaDetector on {len(image_files)} images...')

    results = load_and_run_detector_batch(
        model_file=detector_model,
        image_file_names=image_files,
        checkpoint_path=None,
        confidence_threshold=detection_confidence_threshold,
        checkpoint_frequency=-1,
        results=None,
        n_cores=0,  # Use default
        use_image_queue=True,
        quiet=not verbose,
        image_size=None,
        batch_size=detector_batch_size,
        include_image_size=True,
        include_image_timestamp=False,
        include_exif_data=False
    )

    # Write results to temporary file
    write_results_to_file(results, detector_output_file,
                          relative_path_base=source_folder)

    if verbose:
        print(f'MegaDetector results written to {detector_output_file}')

    return detector_output_file

# ...def run_detection_step(...)

def run_classification_step(detector_results_file: str,
                            classifier_model: str,
                            classifier_batch_size: int,
                            classifier_worker_threads: int,
                            detection_confidence_threshold: float,
                            enable_rollup: bool,
                            country: Optional[str],
                            admin1_region: Optional[str],
                            source_folder: str,
                            temp_folder: str) -> str:
    """
    Run SpeciesNet classification on detections from MegaDetector results.

    Args:
        detector_results_file (str): path to MegaDetector output JSON file
        classifier_model (str): classifier model identifier
        classifier_batch_size (int): batch size for classification
        classifier_worker_threads (int): number of worker threads
        detection_confidence_threshold (float): minimum confidence for detections to classify
        enable_rollup (bool): whether to apply taxonomic rollup
        country (Optional[str]): country code for geofencing
        admin1_region (Optional[str]): admin1 region for geofencing
        source_folder (str): source folder for resolving relative paths
        temp_folder (str): folder for temporary files

    Returns:
        str: path to temporary file containing classification results
    """

    if verbose:
        print('Starting SpeciesNet classification step...')

    # Load MegaDetector results
    with open(detector_results_file, 'r') as f:
        detector_results = json.load(f)

    images = detector_results.get('images', [])
    if len(images) == 0:
        raise ValueError('No images found in detector results')

    # Classifier will be loaded in each subprocess
    print(f'Using SpeciesNet classifier: {classifier_model}')

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    original_start_method = multiprocessing.get_start_method()
    if original_start_method != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        if verbose:
            print(f'Set multiprocessing start method to spawn (was {original_start_method})')

    # Set up multiprocessing queues
    max_queue_size = classifier_worker_threads * 10
    image_queue = JoinableQueue(max_queue_size)
    batch_queue = Queue()
    results_queue = Queue()

    # Start producer workers
    producers = []
    for i in range(classifier_worker_threads):
        p = Process(target=_crop_producer_func,
                    args=(image_queue, batch_queue, classifier_model,
                          detection_confidence_threshold, source_folder, i))
        p.start()
        producers.append(p)

    # Start consumer worker
    consumer = Process(target=_crop_consumer_func,
                       args=(batch_queue, results_queue, classifier_model,
                             classifier_batch_size, classifier_worker_threads,
                             enable_rollup, country, admin1_region))
    consumer.start()

    # Populate image queue
    for image_data in images:
        image_queue.put(image_data)

    # Send sentinel signals to producers
    for _ in range(classifier_worker_threads):
        image_queue.put(None)

    # Wait for all work to complete
    image_queue.join()

    # Wait for results
    classification_results = results_queue.get()

    # Clean up processes
    for p in producers:
        p.join()
    consumer.join()

    # Merge classification results back into detector results
    for image_data in images:
        image_file = image_data['file']
        detections = image_data.get('detections', [])

        if image_file in classification_results:
            image_classifications = classification_results[image_file]

            for detection_index, detection in enumerate(detections):
                if detection_index in image_classifications:
                    result = image_classifications[detection_index]

                    if 'failure' in result:
                        # Add failure to the image, not the detection
                        if 'failure' not in image_data:
                            image_data['failure'] = result['failure']
                        else:
                            image_data['failure'] += f"; {result['failure']}"
                    else:
                        # Add classifications to the detection
                        detection['classifications'] = result['classifications']

    # Create output file
    classification_output_file = os.path.join(
        temp_folder, 'classification_results.json')

    # Update metadata in the results
    if 'info' not in detector_results:
        detector_results['info'] = {}

    detector_results['info']['classifier'] = classifier_model
    detector_results['info']['classification_completion_time'] = time.strftime(
        '%Y-%m-%d %H:%M:%S')

    # Add classification categories - this would need to be populated from the classifier
    # For now, we'll leave it empty as it would require mapping SpeciesNet labels
    if 'classification_categories' not in detector_results:
        detector_results['classification_categories'] = {}

    # Write results
    with open(classification_output_file, 'w') as f:
        json.dump(detector_results, f, indent=2)

    if verbose:
        print(
            f'Classification results written to {classification_output_file}')

    return classification_output_file

# ...def def run_classification_step(...)


#%% Command-line driver
def main():
    """
    Command-line driver for run_md_and_speciesnet.py
    """

    parser = argparse.ArgumentParser(
        description='Run MegaDetector and SpeciesNet on a folder of images/videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('source',
                        help='Folder containing images and/or videos to process')
    parser.add_argument('output_file',
                        help='Output file for results (JSON format)')

    # Optional arguments
    parser.add_argument('--detector_model', default=DEFAULT_DETECTOR_MODEL,
                        help='MegaDetector model identifier')
    parser.add_argument('--classification_model', default=DEFAULT_CLASSIFIER_MODEL,
                        help='SpeciesNet classifier model identifier')
    parser.add_argument('--detector_batch_size', type=int, default=DEFAULT_DETECTOR_BATCH_SIZE,
                        help='Batch size for MegaDetector inference')
    parser.add_argument('--classifier_batch_size', type=int, default=DEFAULT_CLASSIFIER_BATCH_SIZE,
                        help='Batch size for SpeciesNet classification')
    parser.add_argument('--classifier_worker_threads', type=int, default=DEFAULT_CLASSIFIER_WORKER_THREADS,
                        help='Number of worker threads for classification preprocessing')
    parser.add_argument('--detection_confidence_threshold', type=float, default=DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for detections to classify')
    parser.add_argument('--intermediate_file_folder', default=None,
                        help='Folder for intermediate files (default: system temp)')
    parser.add_argument('--keep_intermediate_files', action='store_true',
                        help='Keep intermediate files for debugging')
    parser.add_argument('--norollup', action='store_true',
                        help='Disable taxonomic rollup')
    parser.add_argument('--country', default=None,
                        help='Country code (ISO 3166-1 alpha-3) for geofencing')
    parser.add_argument('--admin1_region', '--state', default=None,
                        help='Admin1 region/state code for geofencing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # Set global verbose flag
    global verbose
    verbose = args.verbose

    # Also set the run_detector_batch verbose flag
    run_detector_batch.verbose = verbose

    # Validate arguments
    if not os.path.isdir(args.source):
        raise ValueError(f'Source folder does not exist: {args.source}')

    if args.admin1_region and not args.country:
        raise ValueError('--admin1_region requires --country to be specified')

    # Set up intermediate file folder
    if args.intermediate_file_folder:
        temp_folder = args.intermediate_file_folder
        os.makedirs(temp_folder, exist_ok=True)
    else:
        temp_folder = tempfile.mkdtemp(prefix='md_speciesnet_')

    try:
        start_time = time.time()

        print(f'Processing folder: {args.source}')
        print(f'Output file: {args.output_file}')
        print(f'Intermediate files: {temp_folder}')

        # Step 1: Run MegaDetector
        detector_output_file = run_detection_step(
            source_folder=args.source,
            detector_model=args.detector_model,
            detector_batch_size=args.detector_batch_size,
            detection_confidence_threshold=args.detection_confidence_threshold,
            temp_folder=temp_folder
        )

        # Step 2: Run SpeciesNet classification
        final_output_file = run_classification_step(
            detector_results_file=detector_output_file,
            classifier_model=args.classification_model,
            classifier_batch_size=args.classifier_batch_size,
            classifier_worker_threads=args.classifier_worker_threads,
            detection_confidence_threshold=args.detection_confidence_threshold,
            enable_rollup=(not args.norollup),
            country=args.country,
            admin1_region=args.admin1_region,
            source_folder=args.source,
            temp_folder=temp_folder
        )

        # Copy final results to output location
        import shutil
        shutil.copy2(final_output_file, args.output_file)

        elapsed_time = time.time() - start_time
        print(
            f'Processing complete in {humanfriendly.format_timespan(elapsed_time)}')
        print(f'Results written to: {args.output_file}')

    finally:
        # Clean up intermediate files if requested
        if not args.keep_intermediate_files and not args.intermediate_file_folder:
            try:
                import shutil
                shutil.rmtree(temp_folder)
                if verbose:
                    print(f'Cleaned up temporary folder: {temp_folder}')
            except Exception as e:
                print(
                    f'Warning: failed to clean up temporary folder {temp_folder}: {str(e)}')

# ...def main(...)


if __name__ == '__main__':
    main()
