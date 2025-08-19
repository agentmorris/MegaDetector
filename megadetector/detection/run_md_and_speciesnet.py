"""

run_md_and_speciesnet.py

Script to run MegaDetector and SpeciesNet on a folder of images and/or videos.
Runs MD first, then runs SpeciesNet on every above-threshold crop.

"""

#%% Constants, imports, environment

import argparse
import json
import multiprocessing
import os
import sys
import time

from tqdm import tqdm
from multiprocessing import JoinableQueue, Process, Queue

import humanfriendly

from megadetector.detection import run_detector_batch
from megadetector.detection.video_utils import find_videos, run_callback_on_frames, is_video_file
from megadetector.detection.run_detector_batch import load_and_run_detector_batch
from megadetector.detection.run_detector import DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
from megadetector.detection.run_detector import CONF_DIGITS
from megadetector.detection.run_detector_batch import write_results_to_file
from megadetector.utils.ct_utils import round_float
from megadetector.utils.ct_utils import write_json
from megadetector.utils.ct_utils import make_temp_folder
from megadetector.utils.ct_utils import is_list_sorted
from megadetector.utils.ct_utils import is_sphinx_build
from megadetector.utils import path_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.postprocessing.validate_batch_results import \
    validate_batch_results, ValidateBatchResultsOptions
from megadetector.detection.process_video import \
    process_videos, ProcessVideoOptions
from megadetector.postprocessing.combine_batch_outputs import combine_batch_output_files

# We aren't taking an explicit dependency on the speciesnet package yet,
# so we wrap this in a try/except so sphinx can still document this module.
try:
    from speciesnet import SpeciesNetClassifier
    from speciesnet.utils import BBox
    from speciesnet.ensemble import SpeciesNetEnsemble
    from speciesnet.geofence_utils import roll_up_labels_to_first_matching_level
    from speciesnet.geofence_utils import geofence_animal_classification
except Exception:
    pass


#%% Constants

DEFAULT_DETECTOR_MODEL = 'MDV5A'
DEFAULT_CLASSIFIER_MODEL = 'kaggle:google/speciesnet/pyTorch/v4.0.1a'
DEFAULT_DETECTION_CONFIDENCE_THRESHOLD_FOR_CLASSIFICATION = 0.1
DEFAULT_DETECTION_CONFIDENCE_THRESHOLD_FOR_OUTPUT = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
DEFAULT_DETECTOR_BATCH_SIZE = 1
DEFAULT_CLASSIFIER_BATCH_SIZE = 8
DEFAULT_LOADER_WORKERS = 4
MAX_QUEUE_SIZE_IMAGES_PER_WORKER = 10
DEAFULT_SECONDS_PER_VIDEO_FRAME = 1.0

# Max number of classification scores to include per detection
DEFAULT_TOP_N_SCORES = 2

# Unless --norollup is specified, roll up taxonomic levels until the
# cumulative confidence is above this value
ROLLUP_TARGET_CONFIDENCE = 0.5

verbose = False


#%% Support classes

class CropMetadata:
    """
    Metadata for a crop extracted from an image detection.
    """

    def __init__(self,
                 image_file: str,
                 detection_index: int,
                 bbox: list[float],
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
        # List of preprocessed images
        self.crops = []

        # List of CropMetadata objects
        self.metadata = []

    def add_crop(self, crop_data, metadata):
        """
        Args:
            crop_data (PreprocessedImage): preprocessed image data from
                SpeciesNetClassifier.preprocess()
            metadata (CropMetadata): metadata for this crop
        """

        self.crops.append(crop_data)
        self.metadata.append(metadata)

    def __len__(self):
        return len(self.crops)


#%% Support functions for classification

def _process_image_detections(file_path: str,
                              absolute_file_path: str,
                              detection_results: dict,
                              classifier: 'SpeciesNetClassifier',
                              detection_confidence_threshold: float,
                              batch_queue: Queue):
    """
    Process detections from a single image.

    Args:
        file_path (str): relative path to the image file
        absolute_file_path (str): absolute path to the image file
        detection_results (dict): detection results for this image
        classifier (SpeciesNetClassifier): classifier instance for preprocessing
        detection_confidence_threshold (float): classify detections above this threshold
        batch_queue (Queue): queue to send crops to
    """

    detections = detection_results['detections']

    # Load the image
    try:
        image = vis_utils.load_image(absolute_file_path)
        original_width, original_height = image.size
    except Exception as e:
        print('Warning: failed to load image {}: {}'.format(file_path, str(e)))

        # Send failure information to consumer
        failure_metadata = CropMetadata(
            image_file=file_path,
            detection_index=-1,  # -1 indicates whole-image failure
            bbox=[],
            original_width=0,
            original_height=0
        )
        batch_queue.put(('failure',
                         'Failed to load image: {}'.format(str(e)),
                         failure_metadata))
        return

    # Process each detection above threshold
    for detection_index, detection in enumerate(detections):

        conf = detection['conf']
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
                    image_file=file_path,
                    detection_index=detection_index,
                    bbox=bbox,
                    original_width=original_width,
                    original_height=original_height
                )

                # Send individual crop immediately to consumer
                batch_queue.put(('crop', preprocessed_crop, metadata))

        except Exception as e:
            print('Warning: failed to preprocess crop from {}, detection {}: {}'.format(
                file_path, detection_index, str(e)))

            # Send failure information to consumer
            failure_metadata = CropMetadata(
                image_file=file_path,
                detection_index=detection_index,
                bbox=bbox,
                original_width=original_width,
                original_height=original_height
            )
            batch_queue.put(('failure',
                             'Failed to preprocess crop: {}'.format(str(e)),
                             failure_metadata))

    # ...for each detection in this image

# ...def _process_image_detections(...)


def _process_video_detections(file_path: str,
                              absolute_file_path: str,
                              detection_results: dict,
                              classifier: 'SpeciesNetClassifier',
                              detection_confidence_threshold: float,
                              batch_queue: Queue):
    """
    Process detections from a single video.

    Args:
        file_path (str): relative path to the video file
        absolute_file_path (str): absolute path to the video file
        detection_results (dict): detection results for this video
        classifier (SpeciesNetClassifier): classifier instance for preprocessing
        detection_confidence_threshold (float): classify detections above this threshold
        batch_queue (Queue): queue to send crops to
    """

    detections = detection_results['detections']

    # Find frames with above-threshold detections
    frames_with_detections = set()
    frame_to_detections = {}

    for detection_index, detection in enumerate(detections):
        conf = detection['conf']
        if conf < detection_confidence_threshold:
            continue

        frame_number = detection['frame_number']
        frames_with_detections.add(frame_number)

        if frame_number not in frame_to_detections:
            frame_to_detections[frame_number] = []
        frame_to_detections[frame_number].append((detection_index, detection))

    if len(frames_with_detections) == 0:
        return

    frames_to_process = sorted(list(frames_with_detections))

    # Define callback for processing each frame
    def frame_callback(frame_array, frame_id):
        """
        Callback to process a single frame.

        Args:
            frame_array (numpy.ndarray): frame data in PIL format
            frame_id (str): frame identifier like "frame0006.jpg"
        """

        # Extract frame number from frame_id (e.g., "frame0006.jpg" -> 6)
        import re
        match = re.match(r'frame(\d+)\.jpg', frame_id)
        if not match:
            print('Warning: could not parse frame number from {}'.format(frame_id))
            return
        frame_number = int(match.group(1))

        if frame_number not in frame_to_detections:
            return

        # Convert numpy array to PIL Image
        from PIL import Image
        if frame_array.dtype != 'uint8':
            frame_array = (frame_array * 255).astype('uint8')
        frame_image = Image.fromarray(frame_array)
        original_width, original_height = frame_image.size

        # Process each detection in this frame
        for detection_index, detection in frame_to_detections[frame_number]:

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
                    frame_image,
                    bboxes=[speciesnet_bbox],
                    resize=True
                )

                if preprocessed_crop is not None:
                    metadata = CropMetadata(
                        image_file=file_path,
                        detection_index=detection_index,
                        bbox=bbox,
                        original_width=original_width,
                        original_height=original_height
                    )

                    # Send individual crop immediately to consumer
                    batch_queue.put(('crop', preprocessed_crop, metadata))

            except Exception as e:

                print('Warning: failed to preprocess crop from {}, detection {}: {}'.format(
                    file_path, detection_index, str(e)))

                # Send failure information to consumer
                failure_metadata = CropMetadata(
                    image_file=file_path,
                    detection_index=detection_index,
                    bbox=bbox,
                    original_width=original_width,
                    original_height=original_height
                )
                batch_queue.put(('failure',
                                 'Failed to preprocess crop: {}'.format(str(e)),
                                 failure_metadata))

            # ...try/except

        # ...for each detection

    # ...def frame_callback(...)

    # Process the video frames
    try:
        run_callback_on_frames(
            input_video_file=absolute_file_path,
            frame_callback=frame_callback,
            frames_to_process=frames_to_process,
            verbose=verbose
        )
    except Exception as e:
        print('Warning: failed to process video {}: {}'.format(file_path, str(e)))

        # Send failure information to consumer for the whole video
        failure_metadata = CropMetadata(
            image_file=file_path,
            detection_index=-1,  # -1 indicates whole-file failure
            bbox=[],
            original_width=0,
            original_height=0
        )
        batch_queue.put(('failure',
                         'Failed to process video: {}'.format(str(e)),
                         failure_metadata))
    # ...try/except

# ...def _process_video_detections(...)


def _crop_producer_func(image_queue: JoinableQueue,
                        batch_queue: Queue,
                        classifier_model: str,
                        detection_confidence_threshold: float,
                        source_folder: str,
                        producer_id: int = -1):
    """
    Producer function for classification workers.

    Reads images and videos from [image_queue], crops detections above a threshold,
    preprocesses them, and sends individual crops to [batch_queue].
    See the documentation of _crop_consumer_func to for the format of the
    tuples placed on batch_queue.

    Args:
        image_queue (JoinableQueue): queue containing detection_results dicts (for both images and videos)
        batch_queue (Queue): queue to put individual crops into
        classifier_model (str): classifier model identifier to load in this process
        detection_confidence_threshold (float): classify detections above this threshold
        source_folder (str): source folder to resolve relative paths
        producer_id (int, optional): identifier for this producer worker
    """

    if verbose:
        print('Classification producer starting: ID {}'.format(producer_id))

    # Load classifier; this is just being used as a preprocessor, so we force device=cpu.
    #
    # There are a number of reasons loading the model might fail; note to self: *don't*
    # catch Exceptions here.  This should be a catastrophic failure that stops the whole
    # process.
    classifier = SpeciesNetClassifier(classifier_model, device='cpu')
    if verbose:
        print('Classification producer {}: loaded classifier'.format(producer_id))

    while True:

        # Pull an image of detection results from the queue
        detection_results = image_queue.get()

        # Pulling None from the queue indicates that this producer is done
        if detection_results is None:
            image_queue.task_done()
            break

        file_path = detection_results['file']

        # Skip files that failed at the detection stage
        if 'failure' in detection_results:
            image_queue.task_done()
            continue

        # Skip files with no detections
        detections = detection_results['detections']
        if len(detections) == 0:
            image_queue.task_done()
            continue

        # Determine if this is an image or video
        absolute_file_path = os.path.join(source_folder, file_path)
        is_video = is_video_file(file_path)

        if is_video:
            # Process video
            _process_video_detections(
                file_path=file_path,
                absolute_file_path=absolute_file_path,
                detection_results=detection_results,
                classifier=classifier,
                detection_confidence_threshold=detection_confidence_threshold,
                batch_queue=batch_queue
            )
        else:
            # Process image
            _process_image_detections(
                file_path=file_path,
                absolute_file_path=absolute_file_path,
                detection_results=detection_results,
                classifier=classifier,
                detection_confidence_threshold=detection_confidence_threshold,
                batch_queue=batch_queue
            )

        image_queue.task_done()

    # ...while(we still have items to process)

    # Send sentinel to indicate this producer is done
    batch_queue.put(None)

    if verbose:
        print('Classification producer {} finished'.format(producer_id))

# ...def _crop_producer_func(...)


def _crop_consumer_func(batch_queue: Queue,
                        results_queue: Queue,
                        classifier_model: str,
                        batch_size: int,
                        num_producers: int,
                        enable_rollup: bool,
                        country: str = None,
                        admin1_region: str = None):
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
        classifier_model (str): classifier model identifier to load
        batch_size (int): batch size for inference
        num_producers (int): number of producer workers
        enable_rollup (bool): whether to apply taxonomic rollup
        country (str, optional): country code for geofencing
        admin1_region (str, optional): admin1 region for geofencing
    """

    if verbose:
        print('Classification consumer starting')

    # Load classifier
    try:
        classifier = SpeciesNetClassifier(classifier_model)
        if verbose:
            print('Classification consumer: loaded classifier')
    except Exception as e:
        print('Classification consumer: failed to load classifier: {}'.format(str(e)))
        results_queue.put({})
        return

    all_results = {}  # image_file -> {detection_index -> classification_result}
    current_batch = CropBatch()
    producers_finished = 0

    # Load ensemble metadata if rollup/geofencing is enabled
    taxonomy_map = {}
    geofence_map = {}

    if (enable_rollup is not None) or (country is not None):

        # Note to self: there are a number of reasons loading the ensemble
        # could fail here; don't catch this exception, this should be a
        # catatstrophic failure.
        ensemble = SpeciesNetEnsemble(
            classifier_model, geofence=(country is not None))
        taxonomy_map = ensemble.taxonomy_map
        geofence_map = ensemble.geofence_map

    # ...if we need to load ensemble components

    while True:

        # Pull an item from the queue
        item = batch_queue.get()

        # This indicates that a producer worker finished
        if item is None:

            producers_finished += 1
            if producers_finished == num_producers:
                # Process any remaining images
                if len(current_batch) > 0:
                    _process_classification_batch(
                        current_batch, classifier, all_results,
                        enable_rollup, taxonomy_map, geofence_map,
                        country, admin1_region
                    )
                break
            continue

        # ...if a producer finished

        # If we got here, we know we have a crop to process, or
        # a failure to ignore.
        assert isinstance(item, tuple) and len(item) == 3
        item_type, data, metadata = item

        if metadata.image_file not in all_results:
                all_results[metadata.image_file] = {}

        # We should never be processing the same detetion twice
        assert metadata.detection_index not in all_results[metadata.image_file]

        if item_type == 'failure':

            all_results[metadata.image_file][metadata.detection_index] = {
                'failure': 'Failure classification: {}'.format(data)
            }

        else:

            assert item_type == 'crop'
            current_batch.add_crop(data, metadata)
            assert len(current_batch) <= batch_size

            # Process batch if necessary
            if len(current_batch) == batch_size:
                _process_classification_batch(
                    current_batch, classifier, all_results,
                    enable_rollup, taxonomy_map, geofence_map,
                    country, admin1_region
                )
                current_batch = CropBatch()

        # ...was this item a failure or a crop?

    # ...while (we have items to process)

    results_queue.put(all_results)

    if verbose:
        print('Classification consumer finished')

# ...def _crop_consumer_func(...)


def _process_classification_batch(batch: CropBatch,
                                  classifier: 'SpeciesNetClassifier',
                                  all_results: dict,
                                  enable_rollup: bool,
                                  taxonomy_map: dict,
                                  geofence_map: dict,
                                  country: str = None,
                                  admin1_region: str = None):
    """
    Run a batch of crops through the classifier.

    Args:
        batch (CropBatch): batch of crops to process
        classifier (SpeciesNetClassifier): classifier instance
        all_results (dict): dictionary to store results in, modified in-place with format:
            {image_file: {detection_index: {'predictions': [[class_name, score], ...]}
            or {image_file: {detection_index: {'failure': error_message}}}.
        enable_rollup (bool): whether to apply rollup
        taxonomy_map (dict): taxonomy mapping for rollup
        geofence_map (dict): geofence mapping
        country (str, optional): country code for geofencing
        admin1_region (str, optional): admin1 region for geofencing
    """

    if len(batch) == 0:
        print('Warning: _process_classification_batch received empty batch')
        return

    # Prepare batch for inference
    filepaths = [f"{metadata.image_file}_{metadata.detection_index}"
                 for metadata in batch.metadata]

    # Run batch inference
    try:
        batch_results = classifier.batch_predict(filepaths, batch.crops)
    except Exception as e:
        print('*** Batch classification failed: {} ***'.format(str(e)))
        # Mark all crops in this batch as failed
        for metadata in batch.metadata:
            if metadata.image_file not in all_results:
                all_results[metadata.image_file] = {}
            all_results[metadata.image_file][metadata.detection_index] = {
                'failure': 'Failure classification: {}'.format(str(e))
            }
        return

    # Process results
    assert len(batch_results) == len(batch.metadata)
    assert len(batch_results) == len(filepaths)

    for i_result in range(0, len(batch_results)):

        result = batch_results[i_result]
        metadata = batch.metadata[i_result]

        assert metadata.image_file in all_results, \
            'File {} not in results dict'.format(metadata.image_file)

        detection_index = metadata.detection_index

        # Handle classification failure
        if 'failures' in result:
            print('*** Classification failure for image: {} ***'.format(
                filepaths[i_result]))
            all_results[metadata.image_file][detection_index] = {
                'failure': 'Failure classification: SpeciesNet classifier failed'
            }
            continue

        # Extract classification results; this is a dict with keys "classes"
        # and "scores", each of which points to a list.
        classifications = result['classifications']
        classes = classifications['classes']
        scores = classifications['scores']

        classification_was_geofenced = False

        predicted_class = classes[0]
        predicted_score = scores[0]

        # Possibly apply geofencing
        if country:

            geofence_result = geofence_animal_classification(
                labels=classes,
                scores=scores,
                country=country,
                admin1_region=admin1_region,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=True
            )

            geofenced_class, geofenced_score, prediction_source = geofence_result

            if prediction_source != 'classifier':
                classification_was_geofenced = True
                predicted_class = geofenced_class
                predicted_score = geofenced_score

        # ...if we might need to apply geofencing

        # Possibly apply rollup; this was already done if geofencing was applied
        if enable_rollup and (not classification_was_geofenced):

            rollup_result = roll_up_labels_to_first_matching_level(
                labels=classes,
                scores=scores,
                country=country,
                admin1_region=admin1_region,
                target_taxonomy_levels=['species','genus','family', 'order','class', 'kingdom'],
                non_blank_threshold=ROLLUP_TARGET_CONFIDENCE,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=(country is not None)
            )

            if rollup_result is not None:
                rolled_up_class, rolled_up_score, prediction_source = rollup_result
                if rolled_up_class != predicted_class:
                    predicted_class = rolled_up_class
                    predicted_score = rolled_up_score

        # ...if we might need to apply taxonomic rollup

        # For now, we'll store category names as strings; these will be assigned to integer
        # IDs before writing results to file later.
        classification = [predicted_class,predicted_score]

        # Also report raw model classifications
        raw_classifications = []
        for i_class in range(0,len(classes)):
            raw_classifications.append([classes[i_class],scores[i_class]])

        all_results[metadata.image_file][detection_index] = {
            'classifications': [classification],
            'raw_classifications': raw_classifications
        }

    # ...for each result in this batch

# ...def _process_classification_batch(...)


#%% Inference functions

def _run_detection_step(source_folder: str,
                        detector_output_file: str,
                        detector_model: str = DEFAULT_DETECTOR_MODEL,
                        detector_batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE,
                        detection_confidence_threshold: float = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                        detector_worker_threads: int = DEFAULT_LOADER_WORKERS,
                        skip_images: bool = False,
                        skip_video: bool = False,
                        frame_sample: int = None,
                        time_sample: float = None) -> str:
    """
    Run MegaDetector on all images/videos in [source_folder].

    Args:
        source_folder (str): folder containing images/videos
        detector_output_file (str): output .json file
        detector_model (str, optional): detector model identifier
        detector_batch_size (int, optional): batch size for detection
        detection_confidence_threshold (float, optional): confidence threshold for detections
            (to include in the output file)
        detector_worker_threads (int, optional): number of workers to use for preprocessing
        skip_images (bool, optional): ignore images, only process videos
        skip_video (bool, optional): ignore videos, only process images
        frame_sample (int, optional): sample every Nth frame from videos
        time_sample (float, optional): sample frames every N seconds from videos
    """

    print('Starting detection step...')

    # Validate arguments
    assert not (frame_sample is None and time_sample is None), \
        'Must specify either frame_sample or time_sample'

    # Find image and video files
    if not skip_images:
        image_files = path_utils.find_images(source_folder, recursive=True,
                                             return_relative_paths=False)
    else:
        image_files = []

    if not skip_video:
        video_files = find_videos(source_folder, recursive=True,
                                  return_relative_paths=False)
    else:
        video_files = []

    if len(image_files) == 0 and len(video_files) == 0:
        raise ValueError(
            'No images or videos found in {}'.format(source_folder))

    print('Found {} images and {} videos'.format(len(image_files), len(video_files)))

    files_to_merge = []

    # Process images if necessary
    if len(image_files) > 0:

        print('Running MegaDetector on {} images...'.format(len(image_files)))

        image_results = load_and_run_detector_batch(
            model_file=detector_model,
            image_file_names=image_files,
            checkpoint_path=None,
            confidence_threshold=detection_confidence_threshold,
            checkpoint_frequency=-1,
            results=None,
            n_cores=0,
            use_image_queue=True,
            quiet=True,
            image_size=None,
            batch_size=detector_batch_size,
            include_image_size=False,
            include_image_timestamp=False,
            include_exif_data=False,
            loader_workers=detector_worker_threads,
            preprocess_on_image_queue=True
        )

        # Write image results to temporary file
        image_output_file = detector_output_file.replace('.json', '_images.json')
        write_results_to_file(image_results,
                              image_output_file,
                              relative_path_base=source_folder,
                              detector_file=detector_model)

        print('Image detection results written to {}'.format(image_output_file))
        files_to_merge.append(image_output_file)

    # ...if we had images to process

    # Process videos if necessary
    if len(video_files) > 0:

        print('Running MegaDetector on {} videos...'.format(len(video_files)))

        # Set up video processing options
        video_options = ProcessVideoOptions()
        video_options.model_file = detector_model
        video_options.input_video_file = source_folder
        video_options.output_json_file = detector_output_file.replace('.json', '_videos.json')
        video_options.json_confidence_threshold = detection_confidence_threshold
        video_options.frame_sample = frame_sample
        video_options.time_sample = time_sample
        video_options.recursive = True

        # Process videos
        process_videos(video_options)

        print('Video detection results written to {}'.format(video_options.output_json_file))
        files_to_merge.append(video_options.output_json_file)

    # ...if we had videos to process

    # Merge results if we have both images and videos
    if len(files_to_merge) > 1:
        print('Merging image and video detection results...')
        combine_batch_output_files(files_to_merge, detector_output_file)
        print('Merged detection results written to {}'.format(detector_output_file))
    elif len(files_to_merge) == 1:
        # Just rename the single file
        if files_to_merge[0] != detector_output_file:
            if os.path.isfile(detector_output_file):
                print('Detector file {} exists, over-writing'.format(detector_output_file))
                os.remove(detector_output_file)
            os.rename(files_to_merge[0], detector_output_file)
        print('Detection results written to {}'.format(detector_output_file))

# ...def _run_detection_step(...)


def _run_classification_step(detector_results_file: str,
                             merged_results_file: str,
                             source_folder: str,
                             classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
                             classifier_batch_size: int = DEFAULT_CLASSIFIER_BATCH_SIZE,
                             classifier_worker_threads: int = DEFAULT_LOADER_WORKERS,
                             detection_confidence_threshold: float = \
                                DEFAULT_DETECTION_CONFIDENCE_THRESHOLD_FOR_CLASSIFICATION,
                             enable_rollup: bool = True,
                             country: str = None,
                             admin1_region: str = None,
                             top_n_scores: int = DEFAULT_TOP_N_SCORES):
    """
    Run SpeciesNet classification on detections from MegaDetector results.

    Args:
        detector_results_file (str): path to MegaDetector output .json file
        merged_results_file (str): path to which we should write the merged results
        source_folder (str): source folder for resolving relative paths
        classifier_model (str, optional): classifier model identifier
        classifier_batch_size (int, optional): batch size for classification
        classifier_worker_threads (int, optional): number of worker threads
        detection_confidence_threshold (float, optional): classify detections above this threshold
        enable_rollup (bool, optional): whether to apply taxonomic rollup
        country (str, optional): country code for geofencing
        admin1_region (str, optional): admin1 region (typically a state code) for geofencing
        top_n_scores (int, optional): maximum number of scores to include for each detection
    """

    print('Starting SpeciesNet classification step...')

    # Load MegaDetector results
    with open(detector_results_file, 'r') as f:
        detector_results = json.load(f)

    print('Classification step loaded detection results for {} images'.format(
        len(detector_results['images'])))

    images = detector_results['images']
    if len(images) == 0:
        raise ValueError('No images found in detector results')

    print('Using SpeciesNet classifier: {}'.format(classifier_model))

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    original_start_method = multiprocessing.get_start_method()
    if original_start_method != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        print('Set multiprocessing start method to spawn (was {})'.format(
            original_start_method))

    # Set up multiprocessing queues
    max_queue_size = classifier_worker_threads * MAX_QUEUE_SIZE_IMAGES_PER_WORKER
    image_queue = JoinableQueue(max_queue_size)
    batch_queue = Queue()
    results_queue = Queue()

    # Start producer workers
    producers = []
    for i_worker in range(classifier_worker_threads):
        p = Process(target=_crop_producer_func,
                    args=(image_queue, batch_queue, classifier_model,
                          detection_confidence_threshold, source_folder, i_worker))
        p.start()
        producers.append(p)

    # Start consumer worker
    consumer = Process(target=_crop_consumer_func,
                       args=(batch_queue, results_queue, classifier_model,
                             classifier_batch_size, classifier_worker_threads,
                             enable_rollup, country, admin1_region))
    consumer.start()

    # This will block every time the queue reaches its maximum depth, so for
    # very small jobs, this will not be a useful progress bar.
    with tqdm(total=len(images),desc='Classification') as pbar:
        for image_data in images:
            image_queue.put(image_data)
            pbar.update()

    # Send sentinel signals to producers
    for _ in range(classifier_worker_threads):
        image_queue.put(None)

    # Wait for all work to complete
    image_queue.join()

    print('Finished waiting for input queue')

    # Wait for results
    classification_results = results_queue.get()

    # Clean up processes
    for p in producers:
        p.join()
    consumer.join()

    print('Finished waiting for workers')

    class CategoryState:
        """
        Helper class to manage classification category IDs.
        """

        def __init__(self):

            self.next_category_id = 0

            # Maps common name to string-int IDs
            self.common_name_to_id = {}

            # Maps string-ints to common names, as per format standard
            self.classification_categories = {}

            # Maps string-ints to latin taxonomy strings, as per format standard
            self.classification_category_descriptions = {}

        def _get_category_id(self, class_name):
            """
            Get an integer-valued category ID for the 7-token string [class_name],
            creating a new one if necessary.
            """

            # E.g.:
            #
            # "cb553c4e-42c9-4fe0-9bd0-da2d6ed5bfa1;mammalia;carnivora;canidae;urocyon;littoralis;island fox"
            tokens = class_name.split(';')
            assert len(tokens) == 7
            taxonomy_string = ';'.join(tokens[1:6])
            common_name = tokens[6]
            if len(common_name) == 0:
                common_name = taxonomy_string

            if common_name not in self.common_name_to_id:
                self.common_name_to_id[common_name] = str(self.next_category_id)
                self.classification_categories[str(self.next_category_id)] = common_name
                self.classification_category_descriptions[str(self.next_category_id)] = taxonomy_string
                self.next_category_id += 1

            category_id = self.common_name_to_id[common_name]

            return category_id

    # ...class CategoryState

    category_state = CategoryState()

    # Merge classification results back into detector results with proper category IDs
    for image_data in images:

        image_file = image_data['file']

        if ('detections' not in image_data) or (image_data['detections'] is None):
            continue

        detections = image_data['detections']

        if image_file not in classification_results:
            continue

        image_classifications = classification_results[image_file]

        for detection_index, detection in enumerate(detections):

            if detection_index in image_classifications:

                result = image_classifications[detection_index]

                if 'failure' in result:
                    # Add failure to the image, not the detection
                    if 'failure' not in image_data:
                        image_data['failure'] = result['failure']
                    else:
                        image_data['failure'] += ';' + result['failure']
                else:

                    # Convert class names to category IDs
                    classification_pairs = []
                    raw_classification_pairs = []

                    scores = [x[1] for x in result['classifications']]
                    assert is_list_sorted(scores, reverse=True)

                    # Only report the requested number of scores per detection
                    if len(result['classifications']) > top_n_scores:
                        result['classifications'] = \
                            result['classifications'][0:top_n_scores]

                    if len(result['raw_classifications']) > top_n_scores:
                        result['raw_classifications'] = \
                            result['raw_classifications'][0:top_n_scores]

                    for class_name, score in result['classifications']:

                        category_id = category_state._get_category_id(class_name)
                        score = round_float(score, precision=CONF_DIGITS)
                        classification_pairs.append([category_id, score])

                    for class_name, score in result['raw_classifications']:

                        category_id = category_state._get_category_id(class_name)
                        score = round_float(score, precision=CONF_DIGITS)
                        raw_classification_pairs.append([category_id, score])

                    # Add classifications to the detection
                    detection['classifications'] = classification_pairs
                    detection['raw_classifications'] = raw_classification_pairs

                # ...if this classification contains a failure

            # ...if this detection has classification information

        # ...for each detection

    # ...for each image

    # Update metadata in the results
    if 'info' not in detector_results:
        detector_results['info'] = {}

    detector_results['info']['classifier'] = classifier_model
    detector_results['info']['classification_completion_time'] = time.strftime(
        '%Y-%m-%d %H:%M:%S')

    # Add classification category mapping
    detector_results['classification_categories'] = \
        category_state.classification_categories
    detector_results['classification_category_descriptions'] = \
        category_state.classification_category_descriptions

    print('Writing output file')

    # Write results
    write_json(merged_results_file, detector_results)

    if verbose:
        print('Classification results written to {}'.format(merged_results_file))

# ...def _run_classification_step(...)


#%% Command-line driver

def main():
    """
    Command-line driver for run_md_and_speciesnet.py
    """

    if 'speciesnet' not in sys.modules:
        print('It looks like the speciesnet package is not available, try "pip install speciesnet"')
        if not is_sphinx_build():
            sys.exit(-1)

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
    parser.add_argument('--detector_model',
                        default=DEFAULT_DETECTOR_MODEL,
                        help='MegaDetector model identifier')
    parser.add_argument('--classification_model',
                        default=DEFAULT_CLASSIFIER_MODEL,
                        help='SpeciesNet classifier model identifier')
    parser.add_argument('--detector_batch_size',
                        type=int,
                        default=DEFAULT_DETECTOR_BATCH_SIZE,
                        help='Batch size for MegaDetector inference')
    parser.add_argument('--classifier_batch_size',
                        type=int,
                        default=DEFAULT_CLASSIFIER_BATCH_SIZE,
                        help='Batch size for SpeciesNet classification')
    parser.add_argument('--loader_workers',
                        type=int,
                        default=DEFAULT_LOADER_WORKERS,
                        help='Number of worker threads for preprocessing')
    parser.add_argument('--detection_confidence_threshold_for_classification',
                        type=float,
                        default=DEFAULT_DETECTION_CONFIDENCE_THRESHOLD_FOR_CLASSIFICATION,
                        help='Classify detections above this threshold')
    parser.add_argument('--detection_confidence_threshold_for_output',
                        type=float,
                        default=DEFAULT_DETECTION_CONFIDENCE_THRESHOLD_FOR_OUTPUT,
                        help='Include detections above this threshold in the output')
    parser.add_argument('--intermediate_file_folder',
                        default=None,
                        help='Folder for intermediate files (default: system temp)')
    parser.add_argument('--keep_intermediate_files',
                        action='store_true',
                        help='Keep intermediate files for debugging')
    parser.add_argument('--norollup',
                        action='store_true',
                        help='Disable taxonomic rollup')
    parser.add_argument('--country',
                        default=None,
                        help='Country code (ISO 3166-1 alpha-3) for geofencing')
    parser.add_argument('--admin1_region', '--state',
                        default=None,
                        help='Admin1 region/state code for geofencing')
    parser.add_argument('--detections_file',
                        default=None,
                        help='Path to existing MegaDetector output file (skips detection step)')
    parser.add_argument('--skip_video',
                        action='store_true',
                        help='Ignore videos, only process images')
    parser.add_argument('--skip_images',
                        action='store_true',
                        help='Ignore images, only process videos')
    parser.add_argument('--frame_sample',
                        type=int,
                        default=None,
                        help='Sample every Nth frame from videos (mutually exclusive with --time_sample)')
    parser.add_argument('--time_sample',
                        type=float,
                        default=None,
                        help='Sample frames every N seconds from videos (default {})'.\
                            format(DEAFULT_SECONDS_PER_VIDEO_FRAME) + \
                            ' (mutually exclusive with --frame_sample)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable additional debug output')

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
        raise ValueError(
            'Source folder does not exist: {}'.format(args.source))

    if args.admin1_region and not args.country:
        raise ValueError('--admin1_region requires --country to be specified')

    if args.skip_images and args.skip_video:
        raise ValueError('Cannot skip both images and videos')

    if (args.frame_sample is not None) and (args.time_sample is not None):
        raise ValueError('--frame_sample and --time_sample are mutually exclusive')
    if (args.frame_sample is None) and (args.time_sample is None):
        args.time_sample = DEAFULT_SECONDS_PER_VIDEO_FRAME

    # Set up intermediate file folder
    if args.intermediate_file_folder:
        temp_folder = args.intermediate_file_folder
        os.makedirs(temp_folder, exist_ok=True)
    else:
        temp_folder = make_temp_folder(subfolder='run_md_and_speciesnet')

    start_time = time.time()

    print('Processing folder: {}'.format(args.source))
    print('Output file: {}'.format(args.output_file))
    print('Intermediate files: {}'.format(temp_folder))

    # Determine detector output file path
    if args.detections_file:
        detector_output_file = args.detections_file
        print('Using existing detections file: {}'.format(detector_output_file))
        validation_options = ValidateBatchResultsOptions()
        validation_options.check_image_existence = True
        validation_options.relative_path_base = args.source
        validation_options.raise_errors = True
        validate_batch_results(detector_output_file,options=validation_options)
        print('Validated detections file')
    else:
        detector_output_file = os.path.join(temp_folder, 'detector_output.json')

        # Run MegaDetector
        _run_detection_step(
            source_folder=args.source,
            detector_output_file=detector_output_file,
            detector_model=args.detector_model,
            detector_batch_size=args.detector_batch_size,
            detection_confidence_threshold=args.detection_confidence_threshold_for_output,
            detector_worker_threads=args.loader_workers,
            skip_images=args.skip_images,
            skip_video=args.skip_video,
            frame_sample=args.frame_sample,
            time_sample=args.time_sample
        )

    # Run SpeciesNet
    _run_classification_step(
        detector_results_file=detector_output_file,
        merged_results_file=args.output_file,
        source_folder=args.source,
        classifier_model=args.classification_model,
        classifier_batch_size=args.classifier_batch_size,
        classifier_worker_threads=args.loader_workers,
        detection_confidence_threshold=args.detection_confidence_threshold_for_classification,
        enable_rollup=(not args.norollup),
        country=args.country,
        admin1_region=args.admin1_region,
    )

    elapsed_time = time.time() - start_time
    print(
        'Processing complete in {}'.format(humanfriendly.format_timespan(elapsed_time)))
    print('Results written to: {}'.format(args.output_file))

    # Clean up intermediate files if requested
    if (not args.keep_intermediate_files) and \
       (not args.intermediate_file_folder) and \
       (not args.detections_file):
        try:
            os.remove(detector_output_file)
        except Exception as e:
            print('Warning: error removing temporary output file {}: {}'.format(
                detector_output_file, str(e)))

# ...def main(...)


if __name__ == '__main__':
    main()
