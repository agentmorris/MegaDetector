"""

visualize_video_output.py

Render a folder of videos with bounding boxes to a new folder, based on a
detector output file.

"""

#%% Imports

import argparse
import os
import random
import sys
import cv2

from tqdm import tqdm
from PIL import Image
import numpy as np

from megadetector.data_management.annotations.annotation_constants import detector_bbox_category_id_to_name
from megadetector.detection.run_detector import get_typical_confidence_threshold_from_results
from megadetector.detection.video_utils import run_callback_on_frames, default_fourcc, is_video_file
from megadetector.utils.ct_utils import get_max_conf
from megadetector.utils.path_utils import path_is_abs
from megadetector.utils.wi_utils import load_md_or_speciesnet_file
from megadetector.visualization.visualization_utils import render_detection_bounding_boxes


#%% Constants

# This will only be used if a category mapping is not available in the results file
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

DEFAULT_CLASSIFICATION_THRESHOLD = 0.4
DEFAULT_DETECTION_THRESHOLD = 0.15

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

# ...class VideoVisualizationOptions


#%% Support functions

def _get_video_output_framerate(video_entry, original_framerate):
    """
    Calculate the appropriate output frame rate for a video based on detection frame numbers.

    Args:
        video_entry (dict): video entry from results file containing detections
        original_framerate (float): original frame rate of the video

    Returns:
        float: calculated output frame rate
    """

    if 'detections' not in video_entry or len(video_entry['detections']) == 0:
        return original_framerate

    frame_numbers = []
    for detection in video_entry['detections']:
        if 'frame_number' in detection:
            frame_numbers.append(detection['frame_number'])

    if len(frame_numbers) < 2:
        return original_framerate

    frame_numbers = sorted(set(frame_numbers))
    first_interval = frame_numbers[1] - frame_numbers[0]

    # Calculate output frame rate based on first interval
    output_fps = original_framerate / first_interval

    return output_fps


def _get_frames_to_process(video_entry):
    """
    Get list of frame numbers that have detections for this video.

    Args:
        video_entry (dict): video entry from results file

    Returns:
        list: sorted list of unique frame numbers to process
    """

    if 'detections' not in video_entry:
        return []

    frame_numbers = []
    for detection in video_entry['detections']:
        if 'frame_number' in detection:
            frame_numbers.append(detection['frame_number'])

    return sorted(set(frame_numbers))


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


def _process_video(video_entry,
                   detector_label_map,
                   classification_label_map,
                   options,
                   video_dir,
                   out_dir):
    """
    Process a single video, rendering detections on frames and creating output video.

    Args:
        video_entry (dict): video entry from results file
        detector_label_map (dict): mapping of detection category IDs to names
        classification_label_map (dict): mapping of classification category IDs to names
        options (VideoVisualizationOptions): processing options
        video_dir (str): input video directory
        out_dir (str): output directory

    Returns:
        dict: processing result information
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

    # Create output path preserving directory structure
    rel_path = video_entry['file']
    output_video_path = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Get frames to process
    frames_to_process = _get_frames_to_process(video_entry)
    if len(frames_to_process) == 0:
        result['error'] = 'No frames with detections to process'
        return result

    # Determine output frame rate
    original_framerate = video_entry['frame_rate']
    if options.rendering_fs == 'auto':
        output_framerate = _get_video_output_framerate(video_entry, original_framerate)
    else:
        output_framerate = float(options.rendering_fs)

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

        try:

            # Get frame dimensions
            height, width = rendered_frames[0].shape[:2]

            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*options.fourcc)
            video_writer = cv2.VideoWriter(output_video_path, fourcc, output_framerate, (width, height))

            if not video_writer.isOpened():
                result['error'] = 'Failed to open video writer for {}'.format(output_video_path)
                return result

            # Write frames
            for frame in rendered_frames:
                video_writer.write(frame)

            video_writer.release()
            result['success'] = True
            result['frames_processed'] = len(rendered_frames)

        except Exception as e:

            result['error'] = 'Error writing output video: {}'.format(str(e))
            return result

        # ...try/except

    else:

        result['error'] = 'No frames were processed for video {}'.format(video_entry['file'])

    return result

# ...def _process_video(...)


#%% Main function

def visualize_video_output(detector_output_path, out_dir, video_dir, options=None):
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

        if not result['success']:
            print('Warning: Failed to process {}: {}'.format(result['file'],result['error']))

    # ...for each video

    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_frames = sum(r['frames_processed'] for r in results if r['success'])

    print(f'\nProcessing complete:')
    print(f'  Successfully processed: {successful} videos')
    print(f'  Failed: {failed} videos')
    print(f'  Total frames rendered: {total_frames}')

    return results

# ...def visualize_video_output(...)


#%% Command-line driver

def main():

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
             'detection frame intervals, or specify a float value for fps')

    parser.add_argument(
        '--fourcc',
        type=str,
        default=default_fourcc,
        help='Fourcc codec specification for video encoding')

    args = parser.parse_args()

    # Create options object
    options = VideoVisualizationOptions()
    options.confidence_threshold = args.confidence_threshold
    options.sample = args.sample
    options.random_seed = args.random_seed
    options.classification_confidence_threshold = args.classification_confidence_threshold
    options.rendering_fs = args.rendering_fs
    options.fourcc = args.fourcc

    # Run visualization
    visualize_video_output(
        args.detector_output_path,
        args.out_dir,
        args.video_dir,
        options
    )


if __name__ == '__main__':
    main()
