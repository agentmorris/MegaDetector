"""

process_video.py

Splits a video (or folder of videos) into frames, runs the frames through run_detector_batch.py,
and optionally stitches together results into a new video with detection boxes.

When possible, video processing happens in memory, without writing intermediate frames to disk.
If the caller requests that frames be saved, frames are written before processing, and the MD
results correspond to the frames that were written to disk (which simplifies, for example,
repeat detection elimination).

"""

#%% Imports

import os
import sys
import argparse

from megadetector.detection import run_detector_batch
from megadetector.utils.ct_utils import args_to_object
from megadetector.utils.ct_utils import dict_to_kvp_list, parse_kvp_list
from megadetector.detection.video_utils import _filename_to_frame_number
from megadetector.detection.video_utils import find_videos
from megadetector.detection.video_utils import run_callback_on_frames_for_folder
from megadetector.detection.run_detector import load_detector
from megadetector.postprocessing.validate_batch_results import \
        ValidateBatchResultsOptions, validate_batch_results

# Notes to self re: upcoming work on checkpointing
from megadetector.utils.ct_utils import split_list_into_fixed_size_chunks # noqa
from megadetector.detection.run_detector_batch import write_checkpoint, load_checkpoint # noqa


#%% Classes

class ProcessVideoOptions:
    """
    Options controlling the behavior of process_video()
    """

    def __init__(self):

        #: Can be a model filename (.pt or .pb) or a model name (e.g. "MDV5A")
        #:
        #: Use the string "no_detection" to indicate that you only want to extract frames,
        #: not run a model.  If you do this, you almost definitely want to set
        #: keep_extracted_frames to "True", otherwise everything in this module is a no-op.
        #: I.e., there's no reason to extract frames, do nothing with them, then delete them.
        self.model_file = 'MDV5A'

        #: Video (of folder of videos) to process
        self.input_video_file = ''

        #: .json file to which we should write results
        self.output_json_file = None

        #: If [input_video_file] is a folder, should we search for videos recursively?
        self.recursive = False

        #: Enable additional debug console output
        self.verbose = False

        #: Detections below this threshold will not be included in the output file.
        self.json_confidence_threshold = 0.005

        #: Sample every Nth frame; set to None (default) or 1 to sample every frame.  Typically
        #: we sample down to around 3 fps, so for typical 30 fps videos, frame_sample=10 is a
        #: typical value.  Mutually exclusive with [time_sample].
        self.frame_sample = None

        #: Sample frames every N seconds.  Mutually exclusive with [frame_sample]
        self.time_sample = None

        #: Run the model at this image size (don't mess with this unless you know what you're
        #: getting into)... if you just want to pass smaller frames to MD, use max_width
        self.image_size = None

        #: Enable image augmentation
        self.augment = False

        #: By default, a video with no frames (or no frames retrievable with the current parameters)
        #: is silently stored as a failure; this causes it to halt execution.
        self.exit_on_empty_video = False

        #: Detector-specific options
        self.detector_options = None

        #: Write a checkpoint file (to resume processing later) every N videos;
        #: set to -1 (default) to disable checkpointing
        self.checkpoint_frequency = -1

        #: Path to checkpoint file; None (default) for auto-generation based on output filename
        self.checkpoint_path = None

        #: Resume from a checkpoint file, or "auto" to use the most recent checkpoint in the
        #: output directory
        self.resume_from_checkpoint = None

# ...class ProcessVideoOptions


#%% Functions

def _validate_video_options(options):
    """
    Consistency checking for ProcessVideoOptions objects.
    """

    n_sampling_options_configured = 0
    if options.frame_sample is not None:
        n_sampling_options_configured += 1
    if options.time_sample is not None:
        n_sampling_options_configured += 1

    if n_sampling_options_configured > 1:
        raise ValueError('frame_sample and time_sample are mutually exclusive')

    return True


def process_videos(options):
    """
    Process a video or folder of videos through MD.

    Args:
        options (ProcessVideoOptions): all the parameters used to control this process,
            including filenames; see ProcessVideoOptions for details
    """

    ## Validate options

    # Check for incompatible options
    _validate_video_options(options)

    if options.output_json_file is None:
        video_file = options.input_video_file.replace('\\','/')
        if video_file.endswith('/'):
            video_file = video_file[:-1]
        options.output_json_file = video_file + '.json'
        print('Output file not specified, defaulting to {}'.format(
            options.output_json_file))

    assert options.output_json_file.endswith('.json'), \
        'Illegal output file {}'.format(options.output_json_file)

    if options.time_sample is not None:
        every_n_frames_param = -1 * options.time_sample
    else:
        every_n_frames_param = options.frame_sample

    if options.verbose:
        print('Processing videos from input source {}'.format(options.input_video_file))

    detector = load_detector(options.model_file,detector_options=options.detector_options)

    def frame_callback(image_np,image_id):
        return detector.generate_detections_one_image(image_np,
                                                      image_id,
                                                      detection_threshold=options.json_confidence_threshold,
                                                      augment=options.augment,
                                                      image_size=options.image_size,
                                                      verbose=options.verbose)

    """
    [md_results] will be dict with keys 'video_filenames' (list of str), 'frame_rates' (list of floats),
    'results' (list of list of dicts). 'video_filenames' will contain *relative* filenames.
    'results' is a list (one element per video) of lists (one element per frame) of whatever the
    callback returns, typically (but not necessarily) dicts in the MD results format.

    For failed videos, the frame rate will be represented by -1, and "results"
    will be a dict with at least the key "failure".
    """
    if os.path.isfile(options.input_video_file):

        video_folder = os.path.dirname(options.input_video_file)
        video_bn = os.path.basename(options.input_video_file)
        md_results = run_callback_on_frames_for_folder(input_video_folder=video_folder,
                                                       frame_callback=frame_callback,
                                                       every_n_frames=every_n_frames_param,
                                                       verbose=options.verbose,
                                                       files_to_process_relative=[video_bn],
                                                       error_on_empty_video=options.exit_on_empty_video)

    else:

        assert os.path.isdir(options.input_video_file), \
            '{} is neither a file nor a folder'.format(options.input_video_file)

        video_folder = options.input_video_file

        md_results = run_callback_on_frames_for_folder(input_video_folder=options.input_video_file,
                                                       frame_callback=frame_callback,
                                                       every_n_frames=every_n_frames_param,
                                                       verbose=options.verbose,
                                                       recursive=options.recursive,
                                                       error_on_empty_video=options.exit_on_empty_video)

    # ...whether we're processing a file or a folder

    print('Finished running MD on videos')

    video_results = md_results['results']
    video_filenames = md_results['video_filenames']
    video_frame_rates = md_results['frame_rates']

    assert len(video_results) == len(video_filenames)
    assert len(video_results) == len(video_frame_rates)

    video_list_md_format = []

    # i_video = 0; results_this_video = video_results[i_video]
    for i_video,results_this_video in enumerate(video_results):

        video_fn = video_filenames[i_video]

        im = {}
        im['file'] = video_fn
        im['frame_rate'] = video_frame_rates[i_video]
        im['frames_processed'] = []

        if isinstance(results_this_video,dict):

            assert 'failure' in results_this_video
            im['failure'] = results_this_video['failure']
            im['detections'] = None

        else:

            im['detections'] = []

            # results_one_frame = results_this_video[0]
            for results_one_frame in results_this_video:

                assert results_one_frame['file'].startswith(video_fn)

                frame_number = _filename_to_frame_number(results_one_frame['file'])

                assert frame_number not in im['frames_processed'], \
                    'Received the same frame twice for video {}'.format(im['file'])

                im['frames_processed'].append(frame_number)

                for det in results_one_frame['detections']:
                    det['frame_number'] = frame_number

                # This is a no-op if there were no above-threshold detections
                # in this frame
                im['detections'].extend(results_one_frame['detections'])

            # ...for each frame

        # ...was this a failed video?

        im['frames_processed'] = sorted(im['frames_processed'])

        video_list_md_format.append(im)

    # ...for each video

    run_detector_batch.write_results_to_file(
        video_list_md_format,
        options.output_json_file,
        relative_path_base=None,
        detector_file=options.model_file)

    validation_options = ValidateBatchResultsOptions()
    validation_options.raise_errors = True
    validation_options.check_image_existence = True
    validation_options.return_data = False
    validation_options.relative_path_base = video_folder
    validate_batch_results(options.output_json_file,options=validation_options)

# ...process_videos()


def options_to_command(options):
    """
    Convert a ProcessVideoOptions object to a corresponding command line.

    Args:
        options (ProcessVideoOptions): the options set to render as a command line

    Returns:
        str: the command line corresponding to [options]

    :meta private:
    """

    cmd = 'python process_video.py'
    cmd += ' "' + options.model_file + '"'
    cmd += ' "' + options.input_video_file + '"'

    if options.recursive:
        cmd += ' --recursive'
    if options.output_json_file is not None:
        cmd += ' --output_json_file' + ' "' + options.output_json_file + '"'
    if options.json_confidence_threshold is not None:
        cmd += ' --json_confidence_threshold ' + str(options.json_confidence_threshold)
    if options.frame_sample is not None:
        cmd += ' --frame_sample ' + str(options.frame_sample)
    if options.verbose:
        cmd += ' --verbose'
    if options.detector_options is not None and len(options.detector_options) > 0:
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))

    return cmd


#%% Interactive driver

if False:

    pass

    #%% Process a folder of videos

    import os
    from megadetector.detection.process_video import \
        process_videos, ProcessVideoOptions

    model_file = 'MDV5A'
    input_dir = r"G:\temp\md-test-images\video-samples"
    assert os.path.isdir(input_dir)

    output_json_file = os.path.join(input_dir,'mdv5a-video.json')

    print('Processing folder {}'.format(input_dir))

    options = ProcessVideoOptions()
    options.json_confidence_threshold = 0.05
    options.model_file = model_file
    options.input_video_file = input_dir
    options.output_json_file = output_json_file
    options.recursive = True
    # options.frame_sample = 10
    options.time_sample = 2
    options.verbose = True

    process_videos(options)


    #%% Process a single video

    import os
    from megadetector.detection.process_video import \
        process_videos, ProcessVideoOptions
    from megadetector.detection.video_utils import find_videos

    model_file = 'MDV5A'
    input_dir = r"G:\temp\md-test-images\video-samples"
    assert os.path.isdir(input_dir)
    video_fn_abs = find_videos(input_dir)[0]

    output_json_file = os.path.join(input_dir,'mdv5a-single-video.json')

    print('Processing video {}'.format(video_fn_abs))

    options = ProcessVideoOptions()
    options.json_confidence_threshold = 0.05
    options.model_file = model_file
    options.input_video_file = video_fn_abs
    options.output_json_file = output_json_file
    options.recursive = True
    # options.frame_sample = 10
    options.time_sample = 2
    options.verbose = True

    process_videos(options)


#%% Command-line driver

def main(): # noqa

    default_options = ProcessVideoOptions()

    parser = argparse.ArgumentParser(description=(
        'Run MegaDetector on each frame (or every Nth frame) in a video (or folder of videos), optionally '\
        'producing a new video with detections annotated'))

    parser.add_argument('model_file', type=str,
                        help='MegaDetector model file (.pt or .pb) or model name (e.g. "MDV5A"), '\
                             'or the string "no_detection" to run just frame extraction')

    parser.add_argument('input_video_file', type=str,
                        help='video file (or folder) to process')

    parser.add_argument('--recursive', action='store_true',
                        help='recurse into [input_video_file]; only meaningful if a folder '\
                         'is specified as input')

    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json')

    parser.add_argument('--json_confidence_threshold', type=float,
                        default=default_options.json_confidence_threshold,
                        help="don't include boxes in the .json file with confidence "\
                            'below this threshold (default {})'.format(
                                default_options.json_confidence_threshold))

    parser.add_argument('--frame_sample', type=int,
                        default=None, help='process every Nth frame (defaults to every frame), mutually exclusive '\
                            'with --time_sample.')

    parser.add_argument('--time_sample', type=float,
                        default=None, help='process frames every N seconds; this is converted to a '\
                            'frame sampling rate, so it may not be exactly the requested interval in seconds. '\
                            'mutually exclusive with --frame_sample')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable additional debug output')

    parser.add_argument('--image_size',
                        type=int,
                        default=None,
                        help=('Force image resizing to a specific integer size on the long '\
                              'axis (not recommended to change this)'))

    parser.add_argument('--augment',
                        action='store_true',
                        help='Enable image augmentation')

    parser.add_argument('--exit_on_empty_video',
                        action='store_true',
                        help=('By default, videos with no retrievable frames are stored as failures; this' \
                              'causes them to halt execution'))

    parser.add_argument(
        '--detector_options',
        nargs='*',
        metavar='KEY=VALUE',
        default='',
        help='Detector-specific options, as a space-separated list of key-value pairs')

    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=default_options.checkpoint_frequency,
        help='Write a checkpoint file (to resume processing later) every N videos; ' + \
             'set to -1 to disable checkpointing (default {})'.format(
                 default_options.checkpoint_frequency))

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to checkpoint file; defaults to a file in the same directory ' + \
             'as the output file')

    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Resume from a specific checkpoint file, or "auto" to resume from the ' + \
             'most recent checkpoint in the output directory')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    options = ProcessVideoOptions()
    args_to_object(args,options)

    options.detector_options = parse_kvp_list(args.detector_options)

    if os.path.isdir(options.input_video_file):
        process_videos(options)
    else:
        assert os.path.isfile(options.input_video_file), \
            '{} is not a valid file or folder name'.format(options.input_video_file)
        assert not options.recursive, \
            '--recursive is only meaningful when processing a folder'
        process_videos(options)

if __name__ == '__main__':
    main()
