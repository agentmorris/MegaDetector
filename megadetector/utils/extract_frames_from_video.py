"""

extract_frames_from_video.py

Extracts frames from a source video or folder of videos and writes those frames to jpeg files.
For single videos, writes frame images to the destination folder.  For folders of videos, creates
subfolders in the destination folder (one per video) and writes frame images to those subfolders.

"""

#%% Constants and imports

import argparse
import inspect
import json
import os
import sys

from megadetector.detection.video_utils import \
    video_to_frames, video_folder_to_frames, is_video_file


#%% Options class

class FrameExtractionOptions:
    """
    Parameters controlling the behavior of extract_frames().
    """

    def __init__(self):

        #: Number of workers to use for parallel processing
        self.n_workers = 1

        #: Use threads for parallel processing
        self.parallelize_with_threads = False

        #: JPEG quality for extracted frames
        self.quality = 80

        #: Maximum width for extracted frames (defaults to None)
        self.max_width = None

        #: Enable additional debug output
        self.verbose = False

        #: Sample every Nth frame starting from the first frame; if this is None
        #: or 1, every frame is extracted.  If this is a negative value, it's interpreted
        #: as a sampling rate in seconds, which is rounded to the nearest frame sampling
        #: rate. Mutually exclusive with detector_output_file.
        self.frame_sample = None

        #: Path to MegaDetector .json output file. When specified, extracts frames
        #: referenced in this file.  Mutually exclusive with frame_sample. [source]
        #: must be a folder when this is specified.
        self.detector_output_file = None


#%% Core functions

def extract_frames(source, destination, options=None):
    """
    Extracts frames from a video or folder of videos.

    Args:
        source (str): path to a single video file or folder of videos
        destination (str): folder to write frame images to (will be created if it doesn't exist)
        options (FrameExtractionOptions, optional): parameters controlling frame extraction

    Returns:
        tuple: for single videos, returns (list of frame filenames, frame rate).
               for folders, returns (list of lists of frame filenames, list of frame rates, list
               of video filenames)
    """

    if options is None:
        options = FrameExtractionOptions()

    # Validate inputs
    if not os.path.exists(source):
        raise ValueError('Source path {} does not exist'.format(source))

    if os.path.abspath(source) == os.path.abspath(destination):
        raise ValueError('Source and destination cannot be the same')

    # Create destination folder if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    # Determine whether source is a file or folder
    source_is_file = os.path.isfile(source)

    if source_is_file:

        # Validate that source is a video file
        if not is_video_file(source):
            raise ValueError('Source file {} is not a video file'.format(source))

        # detector_output_file requires source to be a folder
        if options.detector_output_file is not None:
            raise ValueError('detector_output_file option requires source to be a folder, not a file')

        # Extract frames from single video
        return video_to_frames(input_video_file=source,
                               output_folder=destination,
                               overwrite=True,
                               every_n_frames=options.frame_sample,
                               verbose=options.verbose,
                               quality=options.quality,
                               max_width=options.max_width,
                               allow_empty_videos=True)

    else:

        frames_to_extract = None
        relative_paths_to_process = None

        # Handle detector output file
        if options.detector_output_file is not None:
            frames_to_extract, relative_paths_to_process = _parse_detector_output(
                options.detector_output_file, source, options.verbose)
            options.frame_sample = None

        return video_folder_to_frames(input_folder=source,
                                      output_folder_base=destination,
                                      recursive=True,
                                      overwrite=True,
                                      n_threads=options.n_workers,
                                      every_n_frames=options.frame_sample,
                                      verbose=options.verbose,
                                      parallelization_uses_threads=options.parallelize_with_threads,
                                      quality=options.quality,
                                      max_width=options.max_width,
                                      frames_to_extract=frames_to_extract,
                                      relative_paths_to_process=relative_paths_to_process,
                                      allow_empty_videos=True)

# ...def extract_frames(...)


def _parse_detector_output(detector_output_file, source_folder, verbose=False):
    """
    Parses a MegaDetector .json output file and returns frame extraction information.

    Args:
        detector_output_file (str): path to MegaDetector .json output file
        source_folder (str): folder containing the source videos
        verbose (bool, optional): enable additional debug output

    Returns:
        tuple: (frames_to_extract_dict, relative_paths_to_process) where:
               - frames_to_extract_dict maps relative video paths to lists of frame numbers
               - relative_paths_to_process is a list of relative video paths to process
    """

    print('Parsing detector output file: {}'.format(detector_output_file))

    # Load the detector results
    with open(detector_output_file, 'r') as f:
        detector_results = json.load(f)

    if 'images' not in detector_results:
        raise ValueError('Detector output file does not contain "images" field')

    images = detector_results['images']
    frames_to_extract_dict = {}
    video_files_in_results = set()

    for image_entry in images:

        file_path = image_entry['file']

        # Skip non-video files
        if not is_video_file(file_path):
            if verbose:
                print('Skipping non-video file {}'.format(file_path))
            continue

        # Check whether video file exists in source folder
        full_video_path = os.path.join(source_folder, file_path)
        if not os.path.isfile(full_video_path):
            print('Warning: video file {} not found in source folder, skipping'.format(file_path))
            continue

        video_files_in_results.add(file_path)

        # Determine which frames to extract for this video
        frames_for_this_video = []

        if 'frames_processed' in image_entry:
            # Use the frames_processed field if available
            frames_for_this_video = image_entry['frames_processed']
            if verbose:
                print('Video {}: using frames_processed field with {} frames'.format(
                    file_path, len(frames_for_this_video)))
        else:
            # Extract frames from detections
            if ('detections' in image_entry) and (image_entry['detections'] is not None):
                frame_numbers = set()
                for detection in image_entry['detections']:
                    if 'frame_number' in detection:
                        frame_numbers.add(detection['frame_number'])
                frames_for_this_video = sorted(list(frame_numbers))
                if verbose:
                    print('Video {}: extracted {} unique frame numbers from detections'.format(
                        file_path, len(frames_for_this_video)))

        if len(frames_for_this_video) > 0:
            frames_to_extract_dict[file_path] = frames_for_this_video

    # ...for each image/video in this file

    relative_paths_to_process = sorted(list(video_files_in_results))

    print('Found {} videos with frames to extract'.format(len(frames_to_extract_dict)))

    return frames_to_extract_dict, relative_paths_to_process

# ...def _parse_detector_output(...)


#%% Command-line driver

def _args_to_object(args, obj):
    """
    Copy all fields from a Namespace (i.e., the output from parse_args) to an object.
    Skips fields starting with _.  Does not check existence in the target object.
    """

    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)


def main():
    """
    Command-line driver for extract_frames_from_video
    """

    parser = argparse.ArgumentParser(
        description='Extract frames from videos and save as JPEG files')

    parser.add_argument('source', type=str,
                        help='Path to a single video file or folder containing videos')
    parser.add_argument('destination', type=str,
                        help='Output folder for extracted frames (will be created if it does not exist)')

    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of workers to use for parallel processing (default: %(default)s)')
    parser.add_argument('--parallelize_with_threads', action='store_true',
                        help='Use threads for parallel processing (default: use processes)')
    parser.add_argument('--quality', type=int, default=80,
                        help='JPEG quality for extracted frames (default: %(default)s)')
    parser.add_argument('--max_width', type=int, default=None,
                        help='Maximum width for extracted frames (default: no resizing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable additional debug output')

    # Mutually exclusive group for frame sampling options
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument('--frame_sample', type=float, default=None,
                             help='Sample every Nth frame starting from the first frame; if this is None or 1, ' +
                                  'every frame is extracted. If this is a negative value, it\'s interpreted as a ' +
                                  'sampling rate in seconds, which is rounded to the nearest frame sampling rate')
    frame_group.add_argument('--detector_output_file', type=str, default=None,
                             help='Path to MegaDetector .json output file. When specified, extracts frames ' +
                                  'referenced in this file. Source must be a folder when this is specified.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # Convert to an options object
    options = FrameExtractionOptions()
    _args_to_object(args, options)

    # Additional validation
    if options.detector_output_file is not None:
        if not os.path.isfile(options.detector_output_file):
            print('Error: detector_output_file {} does not exist'.format(options.detector_output_file))
            sys.exit(1)

    try:
        result = extract_frames(args.source, args.destination, options)

        if os.path.isfile(args.source):
            frame_filenames, frame_rate = result
            print('Extracted {} frames from {} (frame rate: {:.2f} fps)'.format(
                len(frame_filenames), args.source, frame_rate))
        else:
            frame_filenames_by_video, fs_by_video, video_filenames = result
            total_frames = sum(len(frames) for frames in frame_filenames_by_video)
            print('Processed {} videos, extracted {} total frames'.format(
                len(video_filenames), total_frames))

    except Exception as e:
        print('Error: {}'.format(str(e)))
        sys.exit(1)


if __name__ == '__main__':
    main()
