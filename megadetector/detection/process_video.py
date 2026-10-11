"""

process_video.py

Runs MD on a video or folder of videos.

"""

#%% Constants and imports

import os
import sys
import queue
import argparse
import multiprocessing

from copy import deepcopy
from datetime import datetime
from threading import Thread
from multiprocessing import Process
from tqdm import tqdm

from megadetector.detection import run_detector
from megadetector.detection import run_detector_batch
from megadetector.utils.ct_utils import args_to_object
from megadetector.utils.ct_utils import dict_to_kvp_list, parse_kvp_list
from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.detection.video_utils import _filename_to_frame_number
from megadetector.detection.video_utils import find_videos
from megadetector.detection.video_utils import run_callback_on_frames
from megadetector.detection.video_utils import run_callback_on_frames_for_folder_batched
from megadetector.detection.video_utils import video_progress_bar_description
from megadetector.detection.run_detector import load_detector
from megadetector.detection.run_detector import is_gpu_available
from megadetector.detection.run_detector import try_download_known_detector
from megadetector.postprocessing.validate_batch_results import \
        ValidateBatchResultsOptions, validate_batch_results

# Default checkpoint files are written to [output folder]/[prefix]_[date].json.  This is
# deliberately different from the prefix used by run_detector_batch.py, so that "auto" resumption
# of image and video jobs that write results to the same folder doesn't confuse image and video
# checkpoints.
video_checkpoint_prefix = 'md_video_checkpoint'


#%% Classes

class ProcessVideoOptions:
    """
    Options controlling the behavior of process_video()
    """

    def __init__(self):

        #: Can be a model filename (.pt or .pb) or a model name (e.g. "MDV5A")
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

        #: Batch size to use for inference.  Batching only helps on a GPU, so this is
        #: automatically reduced to 1 for CPU inference.  If [use_image_queue] is False, frames
        #: are batched within each video; batches never span videos, so the last batch for each
        #: video is typically smaller than [batch_size].  If [use_image_queue] is True, batches
        #: may span videos.
        self.batch_size = 1

        #: Enable image augmentation
        self.augment = False

        #: Detector-specific options
        self.detector_options = None

        #: Force downloading the model file if a named model (e.g. "MDV5A") is supplied,
        #: even if the local file already exists (typically to overwrite a corrupted
        #: model file)
        self.force_model_download = False

        #: Decode frames on a pool of workers, which may help keep your GPU busy.  Each worker
        #: decodes whole videos.
        self.use_image_queue = False

        #: Use threads (rather than processes) for the frame decoding workers; only relevant
        #: if [use_image_queue] is True.
        self.use_threads_for_queue = False

        #: Number of frame decoding workers; only relevant if [use_image_queue] is True.  This
        #: is reduced to the number of videos if there are fewer videos than workers.
        self.loader_workers = run_detector_batch.default_loaders

        #: Should the frame decoding workers also do detector-specific preprocessing (e.g.
        #: resizing)?  Only relevant if [use_image_queue] is True.
        self.preprocess_on_image_queue = run_detector_batch.default_preprocess_on_image_queue

        #: Write a checkpoint file (to resume processing later) every N videos;
        #: set to -1 (default) to disable checkpointing
        self.checkpoint_frequency = -1

        #: Path to checkpoint file; None (default) to write checkpoints to
        #: md_video_checkpoint_[date].json in the same folder as the output file
        self.checkpoint_path = None

        #: Resume from a checkpoint file, or "auto" to resume from [checkpoint_path] if it's
        #: not None, otherwise from the most recent checkpoint in the output folder
        self.resume_from_checkpoint = None

        #: By default, process_videos() will bail if the checkpoint file already exists (unless
        #: we're resuming from a checkpoint); this allows it to overwrite existing checkpoints
        self.allow_checkpoint_overwrite = False

# ...class ProcessVideoOptions


#%% Support functions for the (optional) frame queue

def _video_producer_func(q,
                         metadata_queue,
                         video_folder,
                         video_files_relative,
                         every_n_frames,
                         producer_id=-1,
                         preprocessor=None,
                         detector_options=None,
                         verbose=False,
                         image_size=None):
    """
    Producer function for the (optional) frame queue; the video equivalent of
    run_detector_batch._producer_func().

    Decodes the sampled frames from each video in [video_files_relative], optionally preprocesses
    them, and puts them on the blocking queue [q] for processing by
    run_detector_batch._consumer_func().  Each frame is queued as [frame_id,image,producer_id],
    where frame_id is [relative video filename]/frameXXXXXX.jpg, the same frame identifiers that
    we use when we're not using the frame queue.  Sends "None" to [q] when finished.

    After queueing the frames for each video, puts a dict on [metadata_queue] with the keys:

    - 'video': the relative filename of the video
    - 'n_frames': the number of frames this function put on [q] for this video
    - 'frame_rate': the frame rate of the video (-1 for failed videos)
    - 'failure': a failure string, or None if the video was decoded successfully

    Args:
        q (Queue): queue that will hold decoded (and possibly preprocessed) frames
        metadata_queue (Queue): queue that will hold per-video information
        video_folder (str): folder containing the videos in [video_files_relative]
        video_files_relative (list): relative paths of the videos to process
        every_n_frames (int or float): frame sampling parameter; see
            video_utils.run_callback_on_frames() for details
        producer_id (int, optional): identifier for this producer worker (for logging)
        preprocessor (str, optional): model file path/identifier for preprocessing, or None to
            skip preprocessing
        detector_options (dict, optional): key/value pairs that are interpreted differently
            by different detectors
        verbose (bool, optional): enable additional debug output
        image_size (int, optional): image size to use for preprocessing
    """

    if verbose:
        print('Video producer starting: ID {}, preprocessor {}'.format(producer_id,preprocessor))
        sys.stdout.flush()

    if preprocessor is not None:
        preprocessor = run_detector_batch._load_preprocessor(preprocessor,
                                                             detector_options=detector_options,
                                                             verbose=verbose)

    # The video we're currently decoding, and the number of frames we've queued for that video;
    # these are updated by frame_callback().
    current_video = {'video_fn_relative': None, 'n_frames_queued': 0}

    def frame_callback(image_np,frame_fn):
        """
        Optionally preprocesses one frame, then puts it on the queue.
        """

        frame_id = current_video['video_fn_relative'] + '/' + frame_fn
        image = image_np

        if preprocessor is not None:
            try:
                image = run_detector_batch._preprocess_image(preprocessor,
                                                             image_np,
                                                             frame_id,
                                                             image_size=image_size,
                                                             verbose=verbose)
            except Exception as e:
                print('Producer process: frame {} cannot be preprocessed:\n{}'.format(
                    frame_id,str(e)))
                # Loader workers communicate failures by passing a string to the consumer,
                # rather than an image.
                image = run_detector.FAILURE_INFER

        q.put([frame_id,image,producer_id])
        current_video['n_frames_queued'] += 1

        # This function gets called for preprocessing only, results are discarded.
        return None

    # ...def frame_callback(...)

    for video_fn_relative in video_files_relative:

        current_video['video_fn_relative'] = video_fn_relative
        current_video['n_frames_queued'] = 0

        # Use forward slashes, for consistency with the paths that appear in failure messages
        # when we're not using the frame queue
        video_fn_abs = os.path.join(video_folder,video_fn_relative).replace('\\','/')

        video_info = {'video': video_fn_relative,
                      'frame_rate': -1.0,
                      'failure': None}

        try:

            video_results = run_callback_on_frames(input_video_file=video_fn_abs,
                                                   frame_callback=frame_callback,
                                                   every_n_frames=every_n_frames,
                                                   verbose=verbose,
                                                   frames_to_process=None,
                                                   allow_empty_videos=False)
            video_info['frame_rate'] = video_results['frame_rate']

        except Exception as e:

            # This is the same failure handling we use when we're not using the frame queue;
            # see video_utils._run_callback_on_frames_for_folder_core().
            print('Warning: error processing video {}: {}'.format(video_fn_abs,str(e)))
            video_info['failure'] = 'Failure processing video: {}'.format(str(e))

        # Frames we queued before a failure will still go through the detector, so we
        # need to account for them even if this video failed.
        video_info['n_frames'] = current_video['n_frames_queued']

        metadata_queue.put(video_info)

    # ...for each video

    # This is a signal to the consumer function that a worker has finished
    q.put(None)

    if verbose:
        print('Video loader worker {} finished'.format(producer_id))
    sys.stdout.flush()

# ...def _video_producer_func(...)


def _run_detector_on_videos_with_queue(video_folder,
                                       video_files_relative,
                                       model_file,
                                       every_n_frames,
                                       confidence_threshold,
                                       video_complete_callback,
                                       image_size=None,
                                       augment=False,
                                       detector_options=None,
                                       loader_workers=run_detector_batch.default_loaders,
                                       preprocess_on_image_queue=run_detector_batch.default_preprocess_on_image_queue,
                                       batch_size=1,
                                       use_threads_for_queue=False,
                                       verbose=False):
    """
    Driver function for the (optional) frame queue; the video equivalent of
    run_detector_batch._run_detector_with_image_queue().  Launches workers to decode (and
    optionally preprocess) frames, and runs run_detector_batch._consumer_func() in the
    calling process.

    Each worker decodes whole videos, and we don't wait for one video to finish before starting
    inference on frames from the next, so batches may span videos.

    Args:
        video_folder (str): folder containing the videos in [video_files_relative]
        video_files_relative (list): relative paths of the videos to process
        model_file (str): filename or model identifier (e.g. "MDV5A")
        every_n_frames (int or float): frame sampling parameter; see
            video_utils.run_callback_on_frames() for details
        confidence_threshold (float): minimum confidence detection to include in output
        video_complete_callback (function): function to call when each video is finished, i.e.
            when all of its frames have been through the detector.  Called with the same
            arguments that video_utils.run_callback_on_frames_for_folder_batched() passes to its
            [video_complete_callback]: the relative filename of the video, the frame rate (-1 for
            failed videos), and the results for this video (a list with one element per frame, or
            a dict with the key "failure" for failed videos).  Videos don't necessarily finish in
            the order in which they appear in [video_files_relative].
        image_size (int, optional): image size to use for inference, only mess with this
            if (a) you're using a model other than MegaDetector or (b) you know what you're
            doing
        augment (bool, optional): enable image augmentation
        detector_options (dict, optional): key/value pairs that are interpreted differently
            by different detectors
        loader_workers (int, optional): number of frame decoding workers to use
        preprocess_on_image_queue (bool, optional): should the workers handle frame decoding and
            preprocessing (True), or just frame decoding (False)?
        batch_size (int, optional): batch size for GPU processing
        use_threads_for_queue (bool, optional): use threads (rather than processes) for the
            frame decoding workers
        verbose (bool, optional): enable additional debug output
    """

    ## Validate inputs

    assert isinstance(model_file,str)

    n_videos = len(video_files_relative)

    if n_videos == 0:
        print('No videos to process')
        return

    if (loader_workers is None) or (loader_workers <= 0):
        loader_workers = 1

    # Each worker processes whole videos, so there's no reason to have more workers than videos
    if loader_workers > n_videos:
        print('Reducing the number of loader workers from {} to {} (the number of videos)'.format(
            loader_workers,n_videos))
        loader_workers = n_videos

    if detector_options is None:
        detector_options = {}

    ## Start the producers

    # This queue will hold frames that need processing by the detector
    frame_queue = multiprocessing.JoinableQueue(run_detector_batch.max_queue_size)

    # This queue will hold per-video information (filename, number of frames, frame rate).
    #
    # Specifically, it holds metadata for videos that have completely decoded, but
    # haven't been picked up by the main (inference) thread.
    metadata_queue = multiprocessing.Queue()

    producers = []

    worker_string = 'thread' if use_threads_for_queue else 'process'
    print('Starting a {} pool with {} workers'.format(worker_string,loader_workers))

    preprocessor = None

    if preprocess_on_image_queue:
        print('Enabling frame queue preprocessing')
        preprocessor = model_file

    chunks = split_list_into_n_chunks(video_files_relative, loader_workers, chunk_strategy='greedy')
    for i_chunk,chunk in enumerate(chunks):
        producer_args = (frame_queue,
                         metadata_queue,
                         video_folder,
                         chunk,
                         every_n_frames,
                         i_chunk,
                         preprocessor,
                         detector_options,
                         verbose,
                         image_size)
        if use_threads_for_queue:
            producer = Thread(target=_video_producer_func,args=producer_args)
        else:
            producer = Process(target=_video_producer_func,args=producer_args)
        producers.append(producer)

    for producer in producers:
        # We launch workers with daemon=True if we are using threads, so they don't prevent
        # the main process from exiting.  If we are using processes, we can clean up immediately
        # if the main process has an error, as long as daemon=False, so we set daemon=False if
        # we're using processes.
        #
        # Also see _stop_producers().
        producer.daemon = use_threads_for_queue
        producer.start()

    ## Keep track of which videos are finished

    # Per-video information, keyed by relative filename.  Initially empty, populated as
    # producer workers finish decoding videos and sending metadata via _receive_video_info.
    #
    # This contains only frame rate and expected/processed frame count for each video, not
    # the actual pixels, and not detection results.  A video being present in this table
    # only means that decoding is finished, not that it's been processed.
    video_info = {}

    # Results for each video that isn't finished yet, keyed by relative filename.
    #
    # When a video is finished, its results will be removed from this dict, and stored in
    # video_results_md_format (via video_complete_callback).
    frame_results_by_video = {}
    for video_fn_relative in video_files_relative:
        frame_results_by_video[video_fn_relative] = []

    finished_videos = set()

    pbar = tqdm(total=n_videos,desc=video_progress_bar_description)

    def _receive_video_info(wait):
        """
        Called every time the consumer (i.e., the inference worker) has results to report,
        then one final time when we've finished inference for the last frame in the job.

        Reads per-video metadata (filename, frame rate, number of frames, failure status)
        from the producer worker via [metadata_queue], and transfers that metadata to
        [video_info].

        If [wait] is True, blocks until we have information for every video; otherwise just
        reads whatever is available.  In practice [wait] will only be True after the consumer
        (i.e., inference) has finished, during normal calls after an inference batch, [wait]
        will be False.

        Returns the list of videos that were sitting on [metadata_queue].
        """

        videos_received = []

        while len(video_info) < n_videos:
            try:
                info = metadata_queue.get(block=wait)
            except queue.Empty:
                break
            assert info['video'] not in video_info, \
                'Received information about video {} twice'.format(info['video'])
            video_info[info['video']] = info
            videos_received.append(info['video'])

        return videos_received

    # ...def _receive_video_info(...)

    def _check_for_finished_videos(candidate_videos):
        """
        Called every time we have new inference results, to check whether any videos are
        finished, then one final time when we've finished inference for the last frame in the job.

        Calls [video_complete_callback] for any videos in [candidate_videos] that are finished,
        i.e. we've received the per-video information from the producer, and we've received
        results from the consumer for all of the frames the producer queued.
        """

        for video_fn_relative in candidate_videos:

            # It's possible to receive inference results for a video that hasn't been fully
            # decoded or hasn't been pulled from [metadata_queue] yet, in which case we wouldn't
            # have metadata in [video_info] yet.  In that case, don't call the video-finished callback.
            if (video_fn_relative not in video_info):
                continue

            assert video_fn_relative not in finished_videos, \
                'Internal error: video {} already marked as finished'.format(
                    video_fn_relative)

            info = video_info[video_fn_relative]
            frame_results = frame_results_by_video[video_fn_relative]

            if len(frame_results) < info['n_frames']:
                continue

            assert len(frame_results) == info['n_frames'], \
                'Received {} results for {} frames from video {}'.format(
                    len(frame_results),info['n_frames'],video_fn_relative)

            finished_videos.add(video_fn_relative)
            del frame_results_by_video[video_fn_relative]

            # Results for frames that were queued before a video failed are discarded, which
            # is also what happens when we're not using the frame queue.
            if info['failure'] is not None:
                results_this_video = {'failure': info['failure']}
            else:
                results_this_video = frame_results

            # In practice this will store these results in video_results_md_format (within
            # process_videos)
            video_complete_callback(video_fn_relative,info['frame_rate'],results_this_video)
            pbar.update(1)

        # ...for each candidate video

    # ...def _check_for_finished_videos(...)

    def _consumer_result_callback(new_results):
        """
        Called by the consumer each time it produces results.
        """

        candidate_videos = set(_receive_video_info(wait=False))

        for r in new_results:
            # Frame identifiers look like [relative video filename]/frameXXXXXX.jpg
            video_fn_relative = r['file'].rsplit('/',1)[0]
            assert video_fn_relative in frame_results_by_video, \
                'Received an unexpected result for {}'.format(r['file'])
            frame_results_by_video[video_fn_relative].append(r)
            candidate_videos.add(video_fn_relative)

        # Do these results mean that we finished one or more videos?
        _check_for_finished_videos(candidate_videos)

    # ...def _consumer_result_callback(...)

    ## Run the consumer

    try:

        # This will block until all frames have been processed and the frame queue
        # has had a sentinel value (None) inserted by every producer worker.  Every time
        # results are available, _consumer_result_callback will get called.
        all_frame_results = run_detector_batch._consumer_func(
            frame_queue,
            model_file,
            confidence_threshold,
            loader_workers,
            image_size=image_size,
            augment=augment,
            detector_options=detector_options,
            preprocess_on_image_queue=preprocess_on_image_queue,
            n_total_images=None,
            batch_size=batch_size,
            checkpoint_path=None,
            checkpoint_frequency=-1,
            result_callback=_consumer_result_callback)

        ## Clean up

        # All the producers have finished at this point, but some of the per-video information
        # may still be in transit.  We need to read all of it before joining the producers.
        _check_for_finished_videos(_receive_video_info(wait=True))
        pbar.close()

        assert len(finished_videos) == n_videos, \
            'Finished {} of {} videos'.format(len(finished_videos),n_videos)

    except BaseException:

        # E.g., the model couldn't be loaded, or a checkpoint couldn't be written
        pbar.close()
        run_detector_batch._stop_producers(producers,[frame_queue,metadata_queue])
        raise

    # ...try/except

    for i_producer,producer in enumerate(producers):
        producer.join()
        if verbose:
            print('Producer {} finished'.format(i_producer))

    frame_queue.join()
    if verbose:
        print('Frame queue joined')

    # We've already received all the results via _consumer_result_callback; this just
    # confirms that the consumer and the producers agree on the number of frames.
    n_frames_queued = sum([info['n_frames'] for info in video_info.values()])
    assert len(all_frame_results) == n_frames_queued, \
        'Received {} results for {} frames'.format(len(all_frame_results),n_frames_queued)

# ...def _run_detector_on_videos_with_queue(...)


#%% Other support functions

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

    if (options.batch_size is None) or (options.batch_size < 1):
        raise ValueError('Illegal batch size {}'.format(options.batch_size))

    if (options.checkpoint_frequency is not None) and \
        (options.checkpoint_frequency != -1) and \
        (options.checkpoint_frequency <= 0):
        raise ValueError('Checkpoint frequency needs to be > 0 or == -1')

    return True


def _absolute_video_path(video_folder, video_fn_relative):
    """
    Converts a relative video filename to the absolute path (with forward slashes) we use to
    identify videos in checkpoints.
    """

    return os.path.abspath(os.path.join(video_folder,video_fn_relative)).replace('\\','/')


def _video_results_to_md_format(video_fn_relative, frame_rate, results_this_video):
    """
    Converts the results for one video, from the format passed to the [video_complete_callback]
    used by video_utils.run_callback_on_frames_for_folder_batched() to a dict representing that
    video in the MD results format.

    Args:
        video_fn_relative (str): relative filename of this video
        frame_rate (float): frame rate of this video, -1 for failed videos
        results_this_video (list or dict): per-frame results, typically produced by the
            detector, or a dict with the key "failure" for failed videos

    Returns:
        dict: the results for this video in the MD results format, with [video_fn_relative]
        as the filename
    """

    im = {}
    im['file'] = video_fn_relative
    im['frame_rate'] = frame_rate
    im['frames_processed'] = []

    if isinstance(results_this_video,dict):

        assert 'failure' in results_this_video
        im['failure'] = results_this_video['failure']
        im['detections'] = None

    else:

        im['detections'] = []

        # Frames aren't necessarily processed in order when we're using the frame queue; sort
        # them so the order of detections doesn't depend on whether we used the frame queue.
        results_this_video = sorted(results_this_video,
                                    key=lambda r: _filename_to_frame_number(r['file']))

        # results_one_frame = results_this_video[0]
        for results_one_frame in results_this_video:

            assert results_one_frame['file'].startswith(video_fn_relative)

            frame_number = _filename_to_frame_number(results_one_frame['file'])

            assert frame_number not in im['frames_processed'], \
                'Received the same frame twice for video {}'.format(im['file'])

            # The MD output format has no way to represent the failure of an individual
            # frame, so we treat failed frames as if they hadn't been sampled at all.
            if ('failure' in results_one_frame) and \
               (results_one_frame['failure'] is not None):
                print('Warning: frame {} of video {} failed: {}'.format(
                    frame_number,video_fn_relative,results_one_frame['failure']))
                continue

            im['frames_processed'].append(frame_number)

            for det in results_one_frame['detections']:
                det['frame_number'] = frame_number

            # This is a no-op if there were no above-threshold detections
            # in this frame
            im['detections'].extend(results_one_frame['detections'])

        # ...for each frame

    # ...was this a failed video?

    im['frames_processed'] = sorted(im['frames_processed'])

    return im

# ...def _video_results_to_md_format(...)


#%% Main function

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


    ## Find videos to process

    if os.path.isfile(options.input_video_file):

        video_folder = os.path.dirname(os.path.abspath(options.input_video_file))
        video_files_relative = [os.path.basename(options.input_video_file)]

    else:

        assert os.path.isdir(options.input_video_file), \
            '{} is neither a file nor a folder'.format(options.input_video_file)

        video_folder = os.path.abspath(options.input_video_file)
        video_files_relative = find_videos(video_folder,
                                           recursive=options.recursive,
                                           convert_slashes=True,
                                           return_relative_paths=True)

    # ...whether we're processing a file or a folder


    ## Set up checkpointing

    output_dir = os.path.dirname(os.path.abspath(options.output_json_file))

    checkpoint_frequency = options.checkpoint_frequency
    if checkpoint_frequency is None:
        checkpoint_frequency = -1

    # Load a checkpoint if necessary
    #
    # Video filenames in checkpoints are always absolute paths; conversion to relative paths
    # happens when we write the output file.
    restored_results = []

    if options.resume_from_checkpoint is not None:

        # If resume_from_checkpoint is "auto", this resumes from checkpoint_path if it was
        # specified, otherwise from the most recent checkpoint in the output folder.
        checkpoint_file = run_detector_batch._resolve_checkpoint_to_resume(
            options.resume_from_checkpoint,
            options.checkpoint_path,
            output_dir,
            checkpoint_prefix=video_checkpoint_prefix)
        restored_results = run_detector_batch.load_checkpoint(checkpoint_file)

        for im in restored_results:
            assert 'frames_processed' in im, \
                'Checkpoint {} does not appear to contain video results'.format(checkpoint_file)

    if checkpoint_frequency != -1:

        if options.checkpoint_path is not None:
            checkpoint_path = options.checkpoint_path
        else:
            checkpoint_path = os.path.join(output_dir,
                                           '{}_{}.json'.format(
                                               video_checkpoint_prefix,
                                               datetime.now().strftime("%Y%m%d%H%M%S")))

        # Don't overwrite existing checkpoint files
        if (not options.allow_checkpoint_overwrite) and (options.resume_from_checkpoint is None):

            assert not os.path.isfile(checkpoint_path), \
                f'Checkpoint path {checkpoint_path} already exists, delete or move it before ' + \
                're-using the same checkpoint path, or specify --allow_checkpoint_overwrite'

        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)),exist_ok=True)

        print('The checkpoint file will be written to {}'.format(checkpoint_path))

    else:

        if options.checkpoint_path is not None:
            print('Warning: checkpointing disabled because checkpoint_frequency is -1, ' + \
                  'but a checkpoint path was specified')
        checkpoint_path = None

    # Don't re-process videos that we restored from the checkpoint
    already_processed = set([im['file'] for im in restored_results])

    video_files_to_process = []
    for video_fn_relative in video_files_relative:
        if _absolute_video_path(video_folder,video_fn_relative) not in already_processed:
            video_files_to_process.append(video_fn_relative)

    if len(video_files_to_process) != len(video_files_relative):
        print('Bypassing {} videos that have already been processed'.format(
            len(video_files_relative) - len(video_files_to_process)))


    ## Prepare detector options

    # Resolve model names (e.g. "MDV5A") to filenames, so we can tell (below) whether a GPU
    # is available for this model type.  We deliberately don't modify options.model_file, which
    # is recorded in the output file.
    model_file = try_download_known_detector(options.model_file,
                                             force_download=options.force_model_download,
                                             verbose=options.verbose)

    # Check this before we start any workers; this is typically a typo in a model name or filename
    if not os.path.isfile(model_file):
        raise ValueError('{} is not a known model name or an existing model file'.format(
            options.model_file))

    detector_options = options.detector_options
    if detector_options is None:
        detector_options = {}
    else:
        detector_options = deepcopy(detector_options)

    batch_size = options.batch_size

    # Batching only helps on a GPU, so we reduce the batch size to 1 if batch inference is
    # requested on a CPU.  This needs to happen before we load the model; some detectors
    # compile themselves for a specific batch size at load time.
    if batch_size > 1:

        gpu_available = is_gpu_available(model_file, context_string='process_videos')

        if not gpu_available:
            print('Batch size of {} requested, but no GPU is available, using batch size 1'.format(
                   batch_size))
            batch_size = 1

    # Some detectors (currently just RF-DETR) need to know the batch size at the time the
    # model is loaded; this is ignored by other detectors.
    if batch_size != 1:
        detector_options['batch_size'] = batch_size

    # Similarly, some detectors (currently just RF-DETR) need to know the image size at the
    # time the model is loaded, so we make sure it's in detector_options.  This is ignored by
    # other detectors.
    #
    # For future-proofing, make sure that if someone supplied image_size in the options *and*
    # in detector options, we don't have inconsistent values.  There's no reason anyone would
    # do this, this is just future-proofing.  Values in detector_options may be strings (they
    # typically come from the command line), so we compare ints.
    detector_options_image_size = detector_options.get('image_size',None)
    if (options.image_size is not None) and (detector_options_image_size is not None) and \
        (int(detector_options_image_size) != options.image_size):
        raise ValueError('Incompatible image_size values: {} vs. {}'.format(
            detector_options_image_size,options.image_size))
    if options.image_size is not None:
        detector_options['image_size'] = options.image_size


    ## Run the detector on each video

    # Results for each video in the MD results format (with absolute filenames), including
    # results restored from a checkpoint
    video_results_md_format = list(restored_results)

    # The number of videos we've processed in this run, i.e. not including videos restored
    # from a checkpoint
    n_videos_processed = 0

    def video_complete_callback(video_fn_relative, frame_rate, results_this_video):
        """
        Called when each video is finished; converts results to the MD format, and writes a
        checkpoint if necessary.
        """

        nonlocal n_videos_processed

        im = _video_results_to_md_format(video_fn_relative,frame_rate,results_this_video)
        im['file'] = _absolute_video_path(video_folder,video_fn_relative)
        video_results_md_format.append(im)

        n_videos_processed += 1

        if (checkpoint_path is not None) and \
            run_detector_batch._checkpoint_boundary_crossed(n_videos_processed - 1,
                                                            n_videos_processed,
                                                            checkpoint_frequency):
            print('Writing a new checkpoint after having processed {} videos since '
                  'last restart'.format(n_videos_processed))
            # This includes any results restored from a previous checkpoint
            run_detector_batch.write_checkpoint(checkpoint_path, video_results_md_format)

    # ...def video_complete_callback(...)

    if options.use_image_queue:

        _run_detector_on_videos_with_queue(
            video_folder=video_folder,
            video_files_relative=video_files_to_process,
            model_file=model_file,
            every_n_frames=every_n_frames_param,
            confidence_threshold=options.json_confidence_threshold,
            video_complete_callback=video_complete_callback,
            image_size=options.image_size,
            augment=options.augment,
            detector_options=detector_options,
            loader_workers=options.loader_workers,
            preprocess_on_image_queue=options.preprocess_on_image_queue,
            batch_size=batch_size,
            use_threads_for_queue=options.use_threads_for_queue,
            verbose=options.verbose)

    else:

        detector = load_detector(model_file,
                                 force_model_download=False,
                                 detector_options=detector_options)

        def frame_batch_callback(images_np,image_ids):
            return detector.generate_detections_one_batch(images_np,
                                                          image_ids,
                                                          detection_threshold=options.json_confidence_threshold,
                                                          augment=options.augment,
                                                          image_size=options.image_size,
                                                          verbose=options.verbose)

        # We use [video_complete_callback] to collect results as each video finishes, so we
        # don't need the dict this function returns.
        _ = run_callback_on_frames_for_folder_batched(
            input_video_folder=video_folder,
            frame_batch_callback=frame_batch_callback,
            batch_size=batch_size,
            every_n_frames=every_n_frames_param,
            verbose=options.verbose,
            files_to_process_relative=video_files_to_process,
            video_complete_callback=video_complete_callback)

    # ...whether we're using the frame queue

    print('Finished running MD on videos')

    assert n_videos_processed == len(video_files_to_process), \
        'Processed {} of {} videos'.format(n_videos_processed,len(video_files_to_process))
    assert len(video_results_md_format) == len(restored_results) + len(video_files_to_process)


    ## Write results

    run_detector_batch.write_results_to_file(
        video_results_md_format,
        options.output_json_file,
        relative_path_base=video_folder,
        detector_file=options.model_file)

    validation_options = ValidateBatchResultsOptions()
    validation_options.raise_errors = True
    validation_options.check_image_existence = True
    validation_options.return_data = False
    validation_options.relative_path_base = video_folder
    validate_batch_results(options.output_json_file,options=validation_options)

    if (checkpoint_path is not None) and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
        print('Deleted checkpoint file {}'.format(checkpoint_path))

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
    if (options.batch_size is not None) and (options.batch_size != 1):
        cmd += ' --batch_size ' + str(options.batch_size)
    if options.verbose:
        cmd += ' --verbose'
    if options.detector_options is not None and len(options.detector_options) > 0:
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
    if options.use_image_queue:
        cmd += ' --use_image_queue'
    if options.use_threads_for_queue:
        cmd += ' --use_threads_for_queue'
    if options.loader_workers != run_detector_batch.default_loaders:
        cmd += ' --loader_workers ' + str(options.loader_workers)
    if options.preprocess_on_image_queue:
        cmd += ' --preprocess_on_image_queue'
    if (options.checkpoint_frequency is not None) and (options.checkpoint_frequency != -1):
        cmd += ' --checkpoint_frequency ' + str(options.checkpoint_frequency)
    if options.checkpoint_path is not None:
        cmd += ' --checkpoint_path "' + options.checkpoint_path + '"'
    if options.resume_from_checkpoint is not None:
        cmd += ' --resume_from_checkpoint "' + options.resume_from_checkpoint + '"'
    if options.allow_checkpoint_overwrite:
        cmd += ' --allow_checkpoint_overwrite'

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
        'Run MegaDetector on a video (or folder of videos)'))

    parser.add_argument('model_file', type=str,
                        help='MegaDetector model file (.pt, .pth, or .pb) or model name (e.g. "MDV5A")')

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

    parser.add_argument('--batch_size',
                        type=int,
                        default=default_options.batch_size,
                        help='Batch size for GPU inference (default {}).  CPU inference will ignore this and use '\
                             'batch_size=1.'.format(default_options.batch_size))

    parser.add_argument('--augment',
                        action='store_true',
                        help='Enable image augmentation')

    parser.add_argument(
        '--detector_options',
        nargs='*',
        metavar='KEY=VALUE',
        default='',
        help='Detector-specific options, as a space-separated list of key-value pairs')

    parser.add_argument(
        '--force_model_download',
        action='store_true',
        help=('If a named model (e.g. "MDV5A") is supplied, force a download of that model even if the ' +\
              'local file already exists (typically to overwrite a corrupted model file).'))

    parser.add_argument(
        '--use_image_queue',
        action='store_true',
        help='Decode frames on a worker pool, may help keep your GPU busy.')

    parser.add_argument(
        '--preprocess_on_image_queue',
        action='store_true',
        help='Whether to do frame resizing on the worker pool (PyTorch detectors only); only relevant ' + \
             'if --use_image_queue is set')

    parser.add_argument(
        '--use_threads_for_queue',
        action='store_true',
        help='Use threads (rather than processes) for the worker pool; only relevant if --use_image_queue is set')

    parser.add_argument(
        '--loader_workers',
        type=int,
        default=default_options.loader_workers,
        help='Number of frame decoding workers to use; only relevant when --use_image_queue ' + \
            'is set (default {})'.format(default_options.loader_workers))

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
        help='File name to which checkpoints will be written if checkpoint_frequency is > 0, ' + \
             'defaults to {}_[date].json in the same folder as the output file'.format(
                 video_checkpoint_prefix))

    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to a JSON checkpoint file to resume from, or "auto" to ' + \
             'find the most recent checkpoint in the same folder as the output file.  "auto" uses ' + \
             '--checkpoint_path (rather than searching the output folder) if --checkpoint_path is specified.')

    parser.add_argument(
        '--allow_checkpoint_overwrite',
        action='store_true',
        help='By default, this script will bail if the specified checkpoint file ' + \
              'already exists; this option allows it to overwrite existing checkpoints')

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
