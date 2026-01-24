"""

visualize_video_output.py

Render a folder of videos with bounding boxes to a new folder, based on a
detector output file.

"""

#%% Imports

import argparse
import os
import random
import cv2

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm
from PIL import Image
import numpy as np

from megadetector.data_management.annotations.annotation_constants import detector_bbox_category_id_to_name
from megadetector.detection.video_utils import run_callback_on_frames, default_fourcc, is_video_file
from megadetector.utils.path_utils import path_is_abs
from megadetector.utils.path_utils import clean_filename
from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file
from megadetector.visualization.visualization_utils import render_detection_bounding_boxes


#%% Constants

# This will only be used if a category mapping is not available in the results file
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

DEFAULT_CLASSIFICATION_THRESHOLD = 0.4
DEFAULT_DETECTION_THRESHOLD = 0.15

NO_FRAMES_STRING = 'No frames with detections to process'
SKIPPED_STRING = 'Skipped'


#%% Classes

class VideoVisualizationOptions:
    """
    Options controlling the behavior of visualize_video_output()
    """

    def __init__(self):

        #: Confidence threshold for including detections
        self.confidence_threshold = DEFAULT_DETECTION_THRESHOLD

        #: Sample N videos to process (-1 for all videos)
        self.sample = -1

        #: Random seed for sampling
        self.random_seed = None

        #: Confidence threshold for including classifications
        self.classification_confidence_threshold = DEFAULT_CLASSIFICATION_THRESHOLD

        #: Frame rate for output videos. Either a float (fps) or 'auto' to calculate
        #: based on detection frame intervals
        self.rendering_fs = 'auto'

        #: Fourcc codec specification for video encoding
        self.fourcc = default_fourcc

        #: Skip frames before first and after last above-threshold detection
        self.trim_to_detections = False

        #: By default, output videos use the same extension as input videos,
        #: use this to force a particular extension
        self.output_extension = None # 'mp4'

        #: By default, relative paths are preserved in the output folder; this
        #: flattens the output structure.
        self.flatten_output = False

        #: When flatten_output is True, path separators will be replaced with this
        #: string.
        self.path_separator_replacement = '#'

        #: Don't render videos below this length
        self.min_output_length_seconds = None

        #: Enable parallel processing of videos
        self.parallelize_rendering = True

        #: Number of concurrent workers (None = default based on CPU count)
        self.parallelize_rendering_n_cores = 8

        #: Use threads (True) vs processes (False) for parallelization
        self.parallelize_rendering_with_threads = True

        #: Should we include classification category names in the output filenames?
        #: Helps for finding showcase videos.  Can be "start", "end", or None.
        self.include_category_names_in_filenames = None

        #: List of category name strings to skip (e.g. "none", "bear_moose"), or None
        self.exclude_category_name_strings = None

# ...class VideoVisualizationOptions


#%% Support functions

def _get_video_output_framerate(video_entry, original_framerate, rendering_fs='auto'):
    """
    Calculate the appropriate output frame rate for a video based on detection frame numbers.

    Args:
        video_entry (dict): video entry from results file, typically containing the key
            'detections'
        original_framerate (float): original frame rate of the video
        rendering_fs (str or float): 'auto' for automatic calculation, negative float for
            speedup factor, positive float for explicit fps

    Returns:
        float: calculated output frame rate
    """

    if rendering_fs != 'auto':

        if float(rendering_fs) < 0:

            # Negative value means speedup factor
            speedup_factor = abs(float(rendering_fs))
            if ('detections' not in video_entry) or (len(video_entry['detections']) == 0):
                # This is a bit arbitrary, but a reasonable thing to do when we have no basis
                # to determine the output frame rate
                return original_framerate * speedup_factor

            frame_numbers = []
            for detection in video_entry['detections']:
                if 'frame_number' in detection:
                    frame_numbers.append(detection['frame_number'])

            if len(frame_numbers) < 2:
                # This is a bit arbitrary, but a reasonable thing to do when we have no basis
                # to determine the output frame rate
                return original_framerate * speedup_factor

            frame_numbers = sorted(set(frame_numbers))
            first_interval = frame_numbers[1] - frame_numbers[0]

            # Calculate base output frame rate based on first interval, then apply speedup
            base_output_fps = original_framerate / first_interval
            return base_output_fps * speedup_factor

        else:

            # Positive value means explicit fps
            return float(rendering_fs)

        # ...if we're using an explicit/speedup-based frame rate

    # ...if we aren't in "auto" frame rate mode

    # Auto mode
    if 'detections' not in video_entry or len(video_entry['detections']) == 0:
        return original_framerate

    frame_numbers = []
    for detection in video_entry['detections']:
        if 'frame_number' in detection:
            frame_numbers.append(detection['frame_number'])

    frame_numbers = sorted(set(frame_numbers))

    if len(frame_numbers) < 2:
        return original_framerate

    first_interval = frame_numbers[1] - frame_numbers[0]

    # Calculate output frame rate based on first interval
    output_fps = original_framerate / first_interval

    return output_fps


def _get_frames_to_process(video_entry, confidence_threshold, trim_to_detections=False):
    """
    Get list of frame numbers that have detections for this video.

    Args:
        video_entry (dict): video entry from results file
        confidence_threshold (float): minimum confidence for detections to be considered
        trim_to_detections (bool): if True, only include frames between first and last
            above-threshold detections (inclusive)

    Returns:
        list: sorted list of unique frame numbers to process
    """

    if 'detections' not in video_entry:
        return []

    if 'frames_processed' in video_entry:
        frame_numbers = set(video_entry['frames_processed'])
    else:
        frame_numbers = set()

    for detection in video_entry['detections']:

        if 'frame_number' in detection:
            # If this file includes the list of frames processed (required as of format
            # version 1.5), every frame with detections should be included in that list
            if 'frames_processed' in video_entry:
                if detection['frame_number'] not in frame_numbers:
                    print('Warning: frames_processed field present in {}, but frame {} is missing'.\
                          format(video_entry['file'],detection['frame_number']))
            frame_numbers.add(detection['frame_number'])
        else:
            print('Warning: detections in {} lack frame numbers'.format(video_entry['file']))

    # ...for each detection

    frame_numbers = sorted(list(frame_numbers))

    if trim_to_detections and (len(frame_numbers) > 0):

        # Find first and last frames with above-threshold detections

        above_threshold_frames = set()
        for detection in video_entry['detections']:
            if detection['conf'] >= confidence_threshold:
                above_threshold_frames.add(detection['frame_number'])

        if len(above_threshold_frames) > 0:

            above_threshold_frames = sorted(list(above_threshold_frames))
            first_detection_frame = above_threshold_frames[0]
            last_detection_frame = above_threshold_frames[-1]

            # Return all frames between first and last above-threshold detections (inclusive)
            trimmed_frames = []
            for frame_num in frame_numbers:
                if (first_detection_frame <= frame_num) and (frame_num <= last_detection_frame):
                    trimmed_frames.append(frame_num)
            return trimmed_frames

        else:
            # No above-threshold detections, return empty list
            return []

    # ...if we're supposed to be trimming to non-empty frames

    return frame_numbers


def _get_detections_for_frame(video_entry, frame_number, confidence_threshold):
    """
    Get all detections for a specific frame that meet confidence thresholds.

    Args:
        video_entry (dict): video entry from results file
        frame_number (int): frame number to get detections for
        confidence_threshold (float): minimum detection confidence

    Returns:
        list: list of detection dictionaries for this frame
    """

    if 'detections' not in video_entry:
        return []

    frame_detections = []

    for detection in video_entry['detections']:
        if ((detection['frame_number'] == frame_number) and
            (detection['conf'] >= confidence_threshold)):
            frame_detections.append(detection)

    return frame_detections


def _get_classification_names(video_entry,classification_label_map,options):
    """
    Return the set of above-threshold classification category names in [video_entry].

    video_entry (dict): video entry from results file, typically containing keys
            'file', 'detections', and 'frame_rate'
        detector_label_map (dict): mapping of detection category IDs to names
        classification_label_map (dict): mapping of classification category IDs to names
        options (VideoVisualizationOptions): processing options

    Returns:
        set: the above-threshold classification category names for this video,
        an empty set if no above-threshold classifications are present
    """

    classification_names = set()

    if ('detections' not in video_entry) or (video_entry['detections'] is None):
        return classification_names

    for det in video_entry['detections']:

        if det['conf'] < options.confidence_threshold:
            continue

        if 'classifications' not in det:
            continue

        for classification in det['classifications']:

            classification_category_id = classification[0]
            classification_conf = classification[1]
            if classification_conf < options.classification_confidence_threshold:
                continue
            classification_category_name = classification_label_map[classification_category_id]
            classification_category_name = clean_filename(classification_category_name,force_lower=True)
            classification_names.add(classification_category_name)

        # ...for each classification

    # ...for each detection

    return classification_names

# ...def _get_classification_names(...)


def _process_video(video_entry,
                   detector_label_map,
                   classification_label_map,
                   options,
                   video_dir,
                   out_dir):
    """
    Process a single video, rendering detections on frames and creating output video.

    Args:
        video_entry (dict): video entry from results file, typically containing keys
            'file', 'detections', and 'frame_rate'
        detector_label_map (dict): mapping of detection category IDs to names
        classification_label_map (dict): mapping of classification category IDs to names
        options (VideoVisualizationOptions): processing options
        video_dir (str): input video directory
        out_dir (str): output directory

    Returns:
        dict: processing result information, with at least keys 'file, 'error', 'success',
        'frames_processed'.
    """

    result = {
        'file': video_entry['file'],
        'success': False,
        'error': None,
        'frames_processed': 0
    }

    # Handle failure cases
    if ('failure' in video_entry) and (video_entry['failure'] is not None):
        result['error'] = 'Ignoring failed video: {}'.format(video_entry['failure'])
        return result

    # Construct input and output paths
    if video_dir is None:
        input_video_path = video_entry['file']
        assert path_is_abs(input_video_path), \
            'Absolute paths are required when no video base dir is supplied'
    else:
        assert not path_is_abs(video_entry['file']), \
            'Relative paths are required when a video base dir is supplied'
        input_video_path = os.path.join(video_dir, video_entry['file'])

    if not os.path.exists(input_video_path):
        result['error'] = 'Video not found: {}'.format(input_video_path)
        return result

    output_fn_relative = video_entry['file']

    if options.flatten_output:
        output_fn_relative = output_fn_relative.replace('\\','/')
        output_fn_relative = \
            output_fn_relative.replace('/',options.path_separator_replacement)

    if options.output_extension is not None:
        ext = options.output_extension
        if not ext.startswith('.'):
            ext = '.' + ext
        output_fn_relative = os.path.splitext(output_fn_relative)[0] + ext

    category_names = _get_classification_names(video_entry,classification_label_map,options)
    if len(category_names) > 0:
        category_names = '_'.join(sorted(list(category_names)))
    else:
        category_names = 'none'

    if options.exclude_category_name_strings is not None:
        if category_names in options.exclude_category_name_strings:
            print('Ignoring category string {} for {}'.format(
                category_names,output_fn_relative))
            result['error'] = SKIPPED_STRING + ': {}'.format(category_names)
            return result

    if options.include_category_names_in_filenames is not None:
        if options.include_category_names_in_filenames == 'start':
            output_fn_relative = category_names + '_' + output_fn_relative
        else:
            output_fn_relative = insert_before_extension(output_fn_relative,category_names)

    output_fn_abs = os.path.join(out_dir, output_fn_relative)
    parent_dir = os.path.dirname(output_fn_abs)
    if len(parent_dir) > 0:
        os.makedirs(parent_dir, exist_ok=True)

    # Get frames to process
    frames_to_process = _get_frames_to_process(video_entry,
                                               options.confidence_threshold,
                                               options.trim_to_detections)
    if len(frames_to_process) == 0:
        result['error'] = NO_FRAMES_STRING
        return result

    # Determine output frame rate
    original_framerate = video_entry['frame_rate']
    output_framerate = _get_video_output_framerate(video_entry,
                                                   original_framerate,
                                                   options.rendering_fs)

    # Bail early if this video is below the output length limit
    if options.min_output_length_seconds is not None:
        output_length = len(frames_to_process) / output_framerate
        if output_length < options.min_output_length_seconds:
            print('Skipping video {}, {}s is below minimum length ({}s)'.format(
                video_entry['file'],output_length,options.min_output_length_seconds))
            result['error'] = 'Skipped, below minimum length'
            return result

    # Storage for rendered frames
    rendered_frames = []

    def frame_callback(frame_array, frame_id):
        """
        Callback function for processing each frame.

        Args:
            frame_array (np.array): frame image data
            frame_id (str): frame identifier (unused)

        Returns:
            np.array: processed frame
        """

        # Extract frame number from the current processing context
        current_frame_idx = len(rendered_frames)
        if current_frame_idx >= len(frames_to_process):
            print('Warning: received an extra frame (index {} of {}) for video {}'.format(
                current_frame_idx,len(frames_to_process),video_entry['file']
            ))
            return frame_array

        current_frame_number = frames_to_process[current_frame_idx]

        # Convert numpy array to PIL Image
        if frame_array.dtype != np.uint8:
            frame_array = (frame_array * 255).astype(np.uint8)

        # Convert from BGR (OpenCV) to RGB (PIL) if needed
        if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_array)

        # Get detections for this frame
        frame_detections = _get_detections_for_frame(
            video_entry,
            current_frame_number,
            options.confidence_threshold
        )

        # Render detections on the frame
        if frame_detections:
            render_detection_bounding_boxes(
                frame_detections,
                pil_image,
                detector_label_map,
                classification_label_map,
                classification_confidence_threshold=options.classification_confidence_threshold
            )

        # Convert back to numpy array for video writing
        frame_array = np.array(pil_image)
        if (len(frame_array.shape) == 3) and (frame_array.shape[2] == 3):
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

        rendered_frames.append(frame_array)
        return frame_array

    # ...def frame_callback(...)

    # Process video frames
    try:
        run_callback_on_frames(
            input_video_path,
            frame_callback,
            frames_to_process=frames_to_process,
            verbose=False
        )
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        result['error'] = 'Error processing video frames: {} ({})'.format(str(e),trace)
        return result

    # Write output video
    if len(rendered_frames) > 0:

        video_writer = None

        try:

            # Get frame dimensions
            height, width = rendered_frames[0].shape[:2]

            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*options.fourcc)
            video_writer = cv2.VideoWriter(output_fn_abs, fourcc, output_framerate, (width, height))

            if not video_writer.isOpened():
                result['error'] = 'Failed to open video writer for {}'.format(output_fn_abs)
                return result

            # Write frames
            for frame in rendered_frames:
                video_writer.write(frame)

            result['success'] = True
            result['frames_processed'] = len(rendered_frames)

        except Exception as e:

            result['error'] = 'Error writing output video: {}'.format(str(e))
            return result

        finally:

            if video_writer is not None:
                try:
                    video_writer.release()
                except Exception as e:
                    print('Warning: failed to release video writer for file {}: {}'.format(
                        video_entry['file'],str(e)))

        # ...try/except

    else:

        result['error'] = 'No frames were processed for video {}'.format(video_entry['file'])

    return result

# ...def _process_video(...)


#%% Main function

def visualize_video_output(detector_output_path,
                           out_dir,
                           video_dir,
                           options=None):
    """
    Renders videos with bounding boxes based on detector output.

    Args:
        detector_output_path (str): path to .json file containing detection results
        out_dir (str): output directory for rendered videos
        video_dir (str): input video directory
        options (VideoVisualizationOptions, optional): processing options

    Returns:
        list: list of processing results for each video
    """

    if options is None:
        options = VideoVisualizationOptions()

    # Validate that input and output directories are different
    if (video_dir is not None) and (os.path.abspath(out_dir) == os.path.abspath(video_dir)):
        raise ValueError('Output directory cannot be the same as video directory')

    # Load results file
    print('Loading results from {}'.format(detector_output_path))
    results_data = load_md_or_speciesnet_file(detector_output_path)

    # Get label mappings
    detector_label_map = results_data.get('detection_categories', DEFAULT_DETECTOR_LABEL_MAP)
    classification_label_map = results_data.get('classification_categories', {})

    # Filter to video entries only
    video_entries = []
    for entry in results_data['images']:
        if is_video_file(entry['file']):
            video_entries.append(entry)

    print('Found {} videos in results file'.format(len(video_entries)))

    # Apply sampling if requested
    if (options.sample > 0) and (len(video_entries) > options.sample):
        if options.random_seed is not None:
            random.seed(options.random_seed)
        n_videos_available = len(video_entries)
        video_entries = random.sample(video_entries, options.sample)
        print('Sampled {} of {} videos for processing'.format(
            len(video_entries),n_videos_available))

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Process each video
    results = []

    if options.parallelize_rendering:

        if options.parallelize_rendering_with_threads:
            worker_string = 'threads'
        else:
            worker_string = 'processes'

        pool = None

        try:

            if options.parallelize_rendering_n_cores is None:
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool()
                else:
                    pool = Pool()
            else:
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool(options.parallelize_rendering_n_cores)
                else:
                    pool = Pool(options.parallelize_rendering_n_cores)
                print('Processing videos with {} {}'.format(options.parallelize_rendering_n_cores,
                                                            worker_string))
            results = list(tqdm(pool.imap(
                                 partial(_process_video,
                                         detector_label_map=detector_label_map,
                                         classification_label_map=classification_label_map,
                                         options=options,
                                         video_dir=video_dir,
                                         out_dir=out_dir),
                                 video_entries), total=len(video_entries), desc='Processing videos'))
        finally:

            if pool is not None:
                pool.close()
                pool.join()
                print('Pool closed and joined for video output visualization')

    else:

        for video_entry in tqdm(video_entries, desc='Processing videos'):

            result = _process_video(
                video_entry,
                detector_label_map,
                classification_label_map,
                options,
                video_dir,
                out_dir
            )
            results.append(result)

    # ...for each video

    # Print summary
    n_empty = 0
    n_failed = 0
    n_successful = 0
    n_skipped = 0

    for r in results:
        if r['success']:
            assert r['error'] is None
            n_successful += 1
        else:
            assert r['error'] is not None
            if NO_FRAMES_STRING in r['error']:
                n_empty += 1
            elif SKIPPED_STRING in r['error']:
                n_skipped += 1
            else:
                n_failed += 1

    total_frames = sum(r['frames_processed'] for r in results if r['success'])

    print('\nProcessing complete:')
    print(f'  Successfully processed: {n_successful} videos')
    print(f'  No above-threshold detections: {n_empty} videos')
    print(f'  Failed: {n_failed} videos')
    print(f'  Skipped: {n_skipped} videos')
    print(f'  Total frames rendered: {total_frames}')

    return results

# ...def visualize_video_output(...)


#%% Command-line driver

def main():
    """
    Command-line driver for visualize_video_output
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Render videos with bounding boxes predicted by a detector above '
                    'a confidence threshold, and save the rendered videos.')

    parser.add_argument(
        'detector_output_path',
        type=str,
        help='Path to json output file of the detector')

    parser.add_argument(
        'out_dir',
        type=str,
        help='Path to directory where the rendered videos will be saved. '
             'The directory will be created if it does not exist.')

    parser.add_argument(
        'video_dir',
        type=str,
        help='Path to directory containing the input videos')

    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help='Confidence threshold above which detections will be rendered')

    parser.add_argument(
        '--sample',
        type=int,
        default=-1,
        help='Number of videos to randomly sample for processing. '
             'Set to -1 to process all videos')

    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Random seed for reproducible sampling')

    parser.add_argument(
        '--classification_confidence_threshold',
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        help='Value between 0 and 1, indicating the confidence threshold '
             'above which classifications will be rendered')

    parser.add_argument(
        '--rendering_fs',
        default='auto',
        help='Frame rate for output videos. Use "auto" to calculate based on '
             'detection frame intervals, positive float for explicit fps, '
             'or negative float for speedup factor (e.g. -2.0 = 2x faster)')

    parser.add_argument(
        '--fourcc',
        type=str,
        default=default_fourcc,
        help='Fourcc codec specification for video encoding')

    parser.add_argument(
        '--trim_to_detections',
        action='store_true',
        help='Skip frames before first and after last above-threshold detection')

    args = parser.parse_args()

    # Create options object
    options = VideoVisualizationOptions()
    options.confidence_threshold = args.confidence_threshold
    options.sample = args.sample
    options.random_seed = args.random_seed
    options.classification_confidence_threshold = args.classification_confidence_threshold
    options.rendering_fs = args.rendering_fs
    options.fourcc = args.fourcc
    options.trim_to_detections = args.trim_to_detections

    # Run visualization
    visualize_video_output(
        args.detector_output_path,
        args.out_dir,
        args.video_dir,
        options
    )


if __name__ == '__main__':
    main()
