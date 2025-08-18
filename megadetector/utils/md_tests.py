"""

md_tests.py

A series of tests to validate basic repo functionality and verify either "correct"
inference behavior, or - when operating in environments other than the training
environment - acceptable deviation from the correct results.

This module should not depend on anything else in this repo outside of the
tests themselves, even if it means some duplicated code (e.g. for downloading files),
since much of what it tries to test is, e.g., imports.

"Correctness" is determined by agreement with a file that this script fetches from lila.science.

"""

#%% Imports and constants

### Only standard imports belong here, not MD-specific imports ###

import os
import json
import glob
import sys
import tempfile
import urllib
import urllib.request
import zipfile
import subprocess
import argparse
import inspect
import pytest

from copy import copy


#%% Classes

class MDTestOptions:
    """
    Options controlling test behavior
    """

    def __init__(self):

        ## Required ##

        #: Force CPU execution
        self.disable_gpu = False

        #: If GPU execution is requested, but a GPU is not available, should we error?
        self.cpu_execution_is_error = False

        #: Skip tests related to video processing
        self.skip_video_tests = False

        #: Skip tests related to still image processing
        self.skip_image_tests = False

        #: Skip tests launched via Python functions (as opposed to CLIs)
        self.skip_python_tests = False

        #: Skip module import tests
        self.skip_import_tests = False

        #: Skip CLI tests
        self.skip_cli_tests = False

        #: Skip download tests
        self.skip_download_tests = False

        #: Skip download tests for local URLs
        self.skip_localhost_downloads = False

        #: Skip force-CPU tests
        self.skip_cpu_tests = False

        #: Force a specific folder for temporary input/output
        self.scratch_dir = None

        #: Where does the test data live?
        self.test_data_url = 'https://lila.science/public/md-test-package.zip'

        #: Download test data even if it appears to have already been downloaded
        self.force_data_download = False

        #: Unzip test data even if it appears to have already been unzipped
        self.force_data_unzip = False

        #: By default, any unexpected behavior is an error; this forces most errors to
        #: be treated as warnings.
        self.warning_mode = False

        #: How much deviation from the expected detection coordinates should we allow before
        #: a disrepancy becomes an error?
        self.max_coord_error = 0.001

        #: How much deviation from the expected confidence values should we allow before
        #: a disrepancy becomes an error?
        self.max_conf_error = 0.005

        #: Current working directory when running CLI tests
        #:
        #: If this is None, we won't mess with the inherited working directory.
        self.cli_working_dir = None

        #: YOLOv5 installation, only relevant if we're testing run_inference_with_yolov5_val.
        #:
        #: If this is None, we'll skip that test.
        self.yolo_working_dir = None

        #: Default model to use for testing (filename, URL, or well-known model string)
        self.default_model = 'MDV5A'

        #: For comparison tests, use a model that produces slightly different output
        self.alt_model = 'MDV5B'

        #: PYTHONPATH to set for CLI tests; if None, inherits from the parent process.  Only
        #: impacts the called functions, not the parent process.
        self.cli_test_pythonpath = None

        #: IoU threshold used to determine whether boxes in two detection files likely correspond
        #: to the same box.
        self.iou_threshold_for_file_comparison = 0.85

        #: Detector options passed to PTDetector
        self.detector_options = {'compatibility_mode':'classic-test'}

        #: Used to drive a series of tests (typically with a low value for
        #: python_test_depth) over a folder of models.
        self.model_folder = None

        #: Used as a knob to control the level of Python tests, typically used when
        #: we want to run a series of simple tests on a small number of models, rather
        #: than a deep test of tests on a small number of models.  The gestalt is that
        #: this is a range from 0-100.
        self.python_test_depth = 100

        #: Currently should be 'all' or 'utils-only'
        self.test_mode = 'all'

        #: Number of cores to use for multi-CPU inference tests
        self.n_cores_for_multiprocessing_tests = 2

        #: Batch size to use when testing batches of size > 1
        self.alternative_batch_size = 3

    # ...def __init__()

# ...class MDTestOptions()


#%% Support functions

def get_expected_results_filename(gpu_is_available,
                                  model_string='mdv5a',
                                  test_type='image',
                                  augment=False,
                                  options=None):
    """
    Expected results vary just a little across inference environments, particularly
    between PT 1.x and 2.x, so when making sure things are working acceptably, we
    compare to a reference file that matches the current environment.

    This function gets the correct filename to compare to current results, depending
    on whether a GPU is available.

    Args:
        gpu_is_available (bool): whether a GPU is available
        model_string (str, optional): the model for which we're retrieving expected results
        test_type (str, optional): the test type we're running ("image" or "video")
        augment (bool, optional): whether we're running this test with image augmentation
        options (MDTestOptions, optional): additional control flow options

    Returns:
        str: relative filename of the results file we should use (within the test
        data zipfile)
    """

    if gpu_is_available:
        hw_string = 'gpu'
    else:
        hw_string = 'cpu'
    import torch
    torch_version = str(torch.__version__)
    if torch_version.startswith('1'):
        assert torch_version == '1.10.1', 'Only tested against PT 1.10.1 and PT 2.x'
        pt_string = 'pt1.10.1'
    else:
        assert torch_version.startswith('2'), 'Unknown torch version: {}'.format(torch_version)
        pt_string = 'pt2.x'

    # A hack for now to account for the fact that even with acceleration enabled and PT2
    # installed, Apple silicon appears to provide the same results as CPU/PT1 inference
    try:
        import torch
        m1_inference = torch.backends.mps.is_built and torch.backends.mps.is_available()
        if m1_inference:
            print('I appear to be running on M1/M2 hardware, using pt1/cpu as the reference results')
            hw_string = 'cpu'
            pt_string = 'pt1.10.1'
    except Exception:
        pass

    aug_string = ''
    if augment:
        aug_string = 'augment-'

    # We only have a single set of video results
    if test_type == 'image':
        fn = '{}-{}{}-{}-{}.json'.format(model_string,aug_string,test_type,hw_string,pt_string)
    else:
        fn = '{}-{}.json'.format(model_string,test_type)

    if options is not None and options.scratch_dir is not None:
        fn = os.path.join(options.scratch_dir,fn)

    return fn


def download_test_data(options=None):
    """
    Downloads the test zipfile if necessary, unzips if necessary.  Initializes
    temporary fields in [options], particularly [options.scratch_dir].

    Args:
        options (MDTestOptions, optional): see MDTestOptions for details

    Returns:
        MDTestOptions: the same object passed in as input, or the options that
        were used if [options] was supplied as None
    """

    if options is None:
        options = MDTestOptions()

    if options.scratch_dir is None:
        tempdir_base = tempfile.gettempdir()
        scratch_dir = os.path.join(tempdir_base,'md-tests')
    else:
        scratch_dir = options.scratch_dir

    os.makedirs(scratch_dir,exist_ok=True)

    # See whether we've already downloaded the data zipfile
    download_zipfile = True
    if not options.force_data_download:
        local_zipfile = os.path.join(scratch_dir,options.test_data_url.split('/')[-1])
        if os.path.isfile(local_zipfile):
            url_info = urllib.request.urlopen(options.test_data_url).info()
            remote_size = int(url_info['Content-Length'])
            target_file_size = os.path.getsize(local_zipfile)
            if remote_size == target_file_size:
                download_zipfile = False

    if download_zipfile:
        print('Downloading test data zipfile')
        urllib.request.urlretrieve(options.test_data_url, local_zipfile)
        print('Finished download to {}'.format(local_zipfile))
    else:
        print('Bypassing test data zipfile download for {}'.format(local_zipfile))


    ## Unzip data

    zipf = zipfile.ZipFile(local_zipfile)
    zip_contents = zipf.filelist

    # file_info = zip_contents[1]
    for file_info in zip_contents:

        expected_size = file_info.file_size
        if expected_size == 0:
            continue
        fn_relative = file_info.filename
        target_file = os.path.join(scratch_dir,fn_relative)
        unzip_file = True
        if (not options.force_data_unzip) and os.path.isfile(target_file):
            existing_file_size = os.path.getsize(target_file)
            if existing_file_size == expected_size:
                unzip_file = False
        if unzip_file:
            os.makedirs(os.path.dirname(target_file),exist_ok=True)
            with open(target_file,'wb') as f:
                f.write(zipf.read(fn_relative))

    # ...for each file in the zipfile

    try:
        zipf.close()
    except Exception as e:
        print('Warning: error closing zipfile:\n{}'.format(str(e)))

    # Warn if files are present that aren't expected
    test_files = glob.glob(os.path.join(scratch_dir,'**/*'), recursive=True)
    test_files = [os.path.relpath(fn,scratch_dir).replace('\\','/') for fn in test_files]
    test_files_set = set(test_files)
    expected_images_set = set(zipf.namelist())
    for fn in expected_images_set:
        if fn.endswith('/'):
            continue
        assert fn in test_files_set, 'File {} is missing from the test image folder'.format(fn)

    # Populate the test options with test data information
    options.scratch_dir = scratch_dir
    options.all_test_files = test_files
    options.test_images = [fn for fn in test_files if os.path.splitext(fn.lower())[1] in ('.jpg','.jpeg','.png')]
    options.test_videos = [fn for fn in test_files if os.path.splitext(fn.lower())[1] in ('.mp4','.avi')]
    options.test_videos = [fn for fn in options.test_videos if \
                           os.path.isfile(os.path.join(scratch_dir,fn))]

    print('Finished unzipping and enumerating test data')

    return options

# ...def download_test_data(...)


def is_gpu_available(verbose=True):
    """
    Checks whether a GPU (including M1/M2 MPS) is available, according to PyTorch.  Returns
    false if PT fails to import.

    Args:
        verbose (bool, optional): enable additional debug console output

    Returns:
        bool: whether a GPU is available
    """

    # Import torch inside this function, so we have a chance to set CUDA_VISIBLE_DEVICES
    # before checking GPU availability.
    try:
        import torch
    except Exception:
        print('Warning: could not import torch')
        return False

    gpu_available = torch.cuda.is_available()

    if gpu_available:
        if verbose:
            print('CUDA available: {}'.format(gpu_available))
            device_ids = list(range(torch.cuda.device_count()))
            if len(device_ids) > 1:
                print('Found multiple devices: {}'.format(str(device_ids)))
    else:
        try:
            gpu_available = torch.backends.mps.is_built and torch.backends.mps.is_available()
        except AttributeError:
            pass
        if gpu_available:
            print('Metal performance shaders available')

    if not gpu_available:
        print('No GPU available')

    return gpu_available

# ...def is_gpu_available(...)


def output_files_are_identical(fn1,fn2,verbose=False):
    """
    Checks whether two MD-formatted output files are identical other than file sorting.

    Args:
        fn1 (str): the first filename to compare
        fn2 (str): the second filename to compare
        verbose (bool, optional): enable additional debug output

    Returns:
        bool: whether [fn1] and [fn2] are identical other than file sorting.
    """

    if verbose:
        print('Comparing {} to {}'.format(fn1,fn2))

    with open(fn1,'r') as f:
         fn1_results = json.load(f)
    fn1_results['images'] = \
         sorted(fn1_results['images'], key=lambda d: d['file'])

    with open(fn2,'r') as f:
         fn2_results = json.load(f)
    fn2_results['images'] = \
         sorted(fn2_results['images'], key=lambda d: d['file'])

    if len(fn1_results['images']) != len(fn1_results['images']):
        if verbose:
            print('{} images in {}, {} images in {}'.format(
                len(fn1_results['images']),fn1,
                len(fn2_results['images']),fn2))
        return False

    # i_image = 0; fn1_image = fn1_results['images'][i_image]
    for i_image,fn1_image in enumerate(fn1_results['images']):

        fn2_image = fn2_results['images'][i_image]

        if fn1_image['file'] != fn2_image['file']:
            if verbose:
                print('Filename difference at {}: {} vs {} '.format(i_image,
                                                                    fn1_image['file'],
                                                                    fn2_image['file']))
            return False

        if fn1_image != fn2_image:
            if verbose:
                print('Image-level difference in image {}: {}'.format(i_image,fn1_image['file']))
            return False

    return True

# ...def output_files_are_identical(...)


def compare_detection_lists(detections_a,detections_b,options,bidirectional_comparison=True):
    """
    Compare two lists of MD-formatted detections, matching detections across lists using IoU
    criteria.  Generally used to compare detections for the same image when two sets of results
    are expected to be more or less the same.

    Args:
        detections_a (list): the first set of detection dicts
        detections_b (list): the second set of detection dicts
        options (MDTestOptions): options that determine tolerable differences between files
        bidirectional_comparison (bool, optional): reverse the arguments and make a recursive
            call.

    Returns:
        dict: a dictionary with keys 'max_conf_error' and 'max_coord_error'.
    """

    from megadetector.utils.ct_utils import get_iou

    max_conf_error = 0
    max_coord_error = 0

    max_conf_error_det_a = None
    max_conf_error_det_b = None

    max_coord_error_det_a = None
    max_coord_error_det_b = None

    # i_det_a = 0
    for i_det_a in range(0,len(detections_a)):

        det_a = detections_a[i_det_a]

        # Don't process very-low-confidence boxes
        # if det_a['conf'] < options.max_conf_error:
        #    continue

        matching_det_b = None
        highest_iou = -1

        # Find the closest match in the detections_b list

        # i_det_b = 0
        for i_det_b in range(0,len(detections_b)):

            det_b = detections_b[i_det_b]

            if det_b['category'] != det_a['category']:
                continue

            iou = get_iou(det_a['bbox'],det_b['bbox'])

            # Is this likely the same detection as det_a?
            if iou >= options.iou_threshold_for_file_comparison and iou > highest_iou:
                matching_det_b = det_b
                highest_iou = iou

        # If there are no detections in this category in detections_b
        if matching_det_b is None:
            if det_a['conf'] > max_conf_error:
                max_conf_error = det_a['conf']
                max_conf_error_det_a = det_a
            # max_coord_error = 1.0
            continue

        assert det_a['category'] == matching_det_b['category']
        conf_err = abs(det_a['conf'] - matching_det_b['conf'])
        coord_differences = []
        for i_coord in range(0,4):
            coord_differences.append(abs(det_a['bbox'][i_coord]-\
                                         matching_det_b['bbox'][i_coord]))
        coord_err = max(coord_differences)

        if conf_err >= max_conf_error:
            max_conf_error = conf_err
            max_conf_error_det_a = det_a
            max_conf_error_det_b = det_b

        if coord_err >= max_coord_error:
            max_coord_error = coord_err
            max_coord_error_det_a = det_a
            max_coord_error_det_b = det_b

    # ...for each detection in detections_a

    if bidirectional_comparison:

        reverse_comparison_results = compare_detection_lists(detections_b,
                                                             detections_a,
                                                             options,
                                                             bidirectional_comparison=False)

        if reverse_comparison_results['max_conf_error'] > max_conf_error:
            max_conf_error = reverse_comparison_results['max_conf_error']
            max_conf_error_det_a = reverse_comparison_results['max_conf_error_det_b']
            max_conf_error_det_b = reverse_comparison_results['max_conf_error_det_a']
        if reverse_comparison_results['max_coord_error'] > max_coord_error:
            max_coord_error = reverse_comparison_results['max_coord_error']
            max_coord_error_det_a = reverse_comparison_results['max_coord_error_det_b']
            max_coord_error_det_b = reverse_comparison_results['max_coord_error_det_a']

    list_comparison_results = {}

    list_comparison_results['max_coord_error'] = max_coord_error
    list_comparison_results['max_coord_error_det_a'] = max_coord_error_det_a
    list_comparison_results['max_coord_error_det_b'] = max_coord_error_det_b

    list_comparison_results['max_conf_error'] = max_conf_error
    list_comparison_results['max_conf_error_det_a'] = max_conf_error_det_a
    list_comparison_results['max_conf_error_det_b'] = max_conf_error_det_b

    return list_comparison_results

# ...def compare_detection_lists(...)


def compare_results(inference_output_file,
                    expected_results_file,
                    options,
                    expected_results_file_is_absolute=False):
    """
    Compare two MD-formatted output files that should be nearly identical, allowing small
    changes (e.g. rounding differences).  Generally used to compare a new results file to
    an expected results file.

    Args:
        inference_output_file (str): the first results file to compare
        expected_results_file (str): the second results file to compare
        options (MDTestOptions): options that determine tolerable differences between files
        expected_results_file_is_absolute (str, optional): by default,
            expected_results_file is appended to options.scratch_dir; this option
            specifies that it's an absolute path.

    Returns:
        dict: dictionary with keys 'max_coord_error' and 'max_conf_error'
    """

    # Read results
    with open(inference_output_file,'r') as f:
        results_from_file = json.load(f) # noqa

    if not expected_results_file_is_absolute:
        expected_results_file= os.path.join(options.scratch_dir,expected_results_file)

    with open(expected_results_file,'r') as f:
        expected_results = json.load(f)

    filename_to_results = {im['file'].replace('\\','/'):im for im in results_from_file['images']}
    filename_to_results_expected = {im['file'].replace('\\','/'):im for im in expected_results['images']}

    assert len(filename_to_results) == len(filename_to_results_expected), \
        'Error: comparing expected file {} to actual file {}, expected {} files in results, found {}'.format(
            expected_results_file,
            inference_output_file,
            len(filename_to_results_expected),
            len(filename_to_results))

    max_conf_error = -1
    max_conf_error_file = None
    max_conf_error_comparison_results = None

    max_coord_error = -1
    max_coord_error_file = None
    max_coord_error_comparison_results = None

    # fn = next(iter(filename_to_results.keys()))
    for fn in filename_to_results.keys():

        actual_image_results = filename_to_results[fn]
        expected_image_results = filename_to_results_expected[fn]

        if 'failure' in actual_image_results:
            # We allow some variation in how failures are represented
            assert 'failure' in expected_image_results and \
                (
                    ('detections' not in actual_image_results) or \
                    (actual_image_results['detections'] is None)
                ) and \
                (
                    ('detections' not in expected_image_results) or \
                    (expected_image_results['detections'] is None)
                )
            continue
        assert 'failure' not in expected_image_results

        actual_detections = actual_image_results['detections']
        expected_detections = expected_image_results['detections']

        comparison_results_this_image = compare_detection_lists(
            detections_a=actual_detections,
            detections_b=expected_detections,
            options=options,
            bidirectional_comparison=True)

        if comparison_results_this_image['max_conf_error'] > max_conf_error:
            max_conf_error = comparison_results_this_image['max_conf_error']
            max_conf_error_comparison_results = comparison_results_this_image
            max_conf_error_file = fn

        if comparison_results_this_image['max_coord_error'] > max_coord_error:
            max_coord_error = comparison_results_this_image['max_coord_error']
            max_coord_error_comparison_results = comparison_results_this_image
            max_coord_error_file = fn

    # ...for each image

    if not options.warning_mode:

        assert max_conf_error <= options.max_conf_error, \
            'Confidence error {} is greater than allowable ({}), on file:\n{} ({},{})'.format(
                max_conf_error,options.max_conf_error,max_conf_error_file,
                inference_output_file,expected_results_file)

        assert max_coord_error <= options.max_coord_error, \
            'Coord error {} is greater than allowable ({}), on file:\n{} ({},{})'.format(
                max_coord_error,options.max_coord_error,max_coord_error_file,
                inference_output_file,expected_results_file)

    print('Max conf error: {} (file {})'.format(
        max_conf_error,max_conf_error_file))
    print('Max coord error: {} (file {})'.format(
        max_coord_error,max_coord_error_file))

    comparison_results = {}
    comparison_results['max_conf_error'] = max_conf_error
    comparison_results['max_conf_error_comparison_results'] = max_conf_error_comparison_results
    comparison_results['max_coord_error'] = max_coord_error
    comparison_results['max_coord_error_comparison_results'] = max_coord_error_comparison_results

    return comparison_results

# ...def compare_results(...)


def _args_to_object(args, obj):
    """
    Copies all fields from a Namespace (typically the output from parse_args) to an
    object. Skips fields starting with _. Does not check existence in the target
    object.

    Args:
        args (argparse.Namespace): the namespace to convert to an object
        obj (object): object whose whose attributes will be updated

    Returns:
        object: the modified object (modified in place, but also returned)
    """

    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)

    return obj


#%% CLI functions

# These are copied from process_utils.py to avoid imports outside of the test
# functions.

os.environ["PYTHONUNBUFFERED"] = "1"

# In some circumstances I want to allow CLI tests to "succeed" even when they return
# specific non-zero output values.
allowable_process_return_codes = [0]

def execute(cmd):
    """
    Runs [cmd] (a single string) in a shell, yielding each line of output to the caller.

    Args:
        cmd (str): command to run

    Returns:
        int: the command's return code, always zero, otherwise a CalledProcessError is raised
    """

    # https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             shell=True, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code not in allowable_process_return_codes:
        raise subprocess.CalledProcessError(return_code, cmd)
    return return_code


def execute_and_print(cmd,print_output=True,catch_exceptions=False,echo_command=True):
    """
    Runs [cmd] (a single string) in a shell, capturing (and optionally printing) output.

    Args:
        cmd (str): command to run
        print_output (bool, optional): whether to print output from [cmd]
        catch_exceptions (bool, optional): whether to catch exceptions, rather than raising
            them
        echo_command (bool, optional): whether to print [cmd] to stdout prior to execution

    Returns:
        dict: a dictionary with fields "status" (the process return code) and "output"
        (the content of stdout)
    """

    if echo_command:
        print('Running command:\n{}\n'.format(cmd))

    to_return = {'status':'unknown','output':''}
    output = []
    try:
        for s in execute(cmd):
            output.append(s)
            if print_output:
                print(s,end='',flush=True)
        to_return['status'] = 0
    except subprocess.CalledProcessError as cpe:
        if not catch_exceptions:
            raise
        print('execute_and_print caught error: {}'.format(cpe.output))
        to_return['status'] = cpe.returncode
    to_return['output'] = output

    return to_return


#%% Python tests

@pytest.mark.skip(reason='Called one for each module')
def test_package_imports(package_name,exceptions=None,verbose=True):
    """
    Imports all modules in [package_name]

    Args:
        package_name (str): the package name to test
        exceptions (list, optional): exclude any modules that contain any of these strings
        verbose (bool, optional): enable additional debug output
    """
    import importlib
    import pkgutil

    package = importlib.import_module(package_name)
    package_path = package.__path__
    imported_modules = []

    if exceptions is None:
        exceptions = []

    for _, modname, _ in pkgutil.walk_packages(package_path, package_name + '.'):

        skip_module = False
        for s in exceptions:
            if s in modname:
                skip_module = True
                break
        if skip_module:
            continue

        if verbose:
            print('Testing import: {}'.format(modname))

        try:
            # Attempt to import each module
            _ = importlib.import_module(modname)
            imported_modules.append(modname)
        except ImportError as e:
            print(f"Failed to import module {modname}: {e}")
            raise


def run_python_tests(options):
    """
    Runs Python-based (as opposed to CLI-based) package tests.

    Args:
        options (MDTestOptions): see MDTestOptions for details
    """

    print('\n*** Starting module tests ***\n')


    ## Prepare data

    download_test_data(options)


    ## Import tests

    if not options.skip_import_tests:

        print('\n** Running package import tests **\n')
        test_package_imports('megadetector.visualization')
        test_package_imports('megadetector.postprocessing')
        test_package_imports('megadetector.postprocessing.repeat_detection_elimination')
        test_package_imports('megadetector.utils',exceptions=['md_tests'])
        test_package_imports('megadetector.data_management',exceptions=['lila','ocr_tools'])


    ## Return early if we're not running torch-related tests

    if options.test_mode == 'utils-only':
        return


    ## Make sure our tests are doing what we think they're doing

    from megadetector.detection import pytorch_detector
    pytorch_detector.require_non_default_compatibility_mode = True


    if not options.skip_image_tests:

        from megadetector.utils import path_utils # noqa
        image_folder = os.path.join(options.scratch_dir,'md-test-images')
        assert os.path.isdir(image_folder), 'Test image folder {} is not available'.format(image_folder)
        inference_output_file = os.path.join(options.scratch_dir,'folder_inference_output.json')
        image_file_names = path_utils.find_images(image_folder,recursive=True)


        ## Run inference on an image

        print('\n** Running MD on a single image (module) **\n')

        from megadetector.detection import run_detector
        from megadetector.visualization import visualization_utils as vis_utils # noqa
        image_fn = os.path.join(options.scratch_dir,options.test_images[0])
        model = run_detector.load_detector(options.default_model,
                                           detector_options=copy(options.detector_options))
        pil_im = vis_utils.load_image(image_fn)
        result = model.generate_detections_one_image(pil_im) # noqa

        if options.python_test_depth <= 1:
            return


        ## Run inference on a folder

        print('\n** Running MD on a folder of images (module) **\n')

        from megadetector.detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file

        results = load_and_run_detector_batch(options.default_model,
                                              image_file_names,
                                              quiet=True,
                                              detector_options=copy(options.detector_options))
        _ = write_results_to_file(results,
                                  inference_output_file,
                                  relative_path_base=image_folder,
                                  detector_file=options.default_model)

        ## Verify results

        # Verify format correctness
        from megadetector.postprocessing.validate_batch_results import validate_batch_results #noqa
        validate_batch_results(inference_output_file)

        # Verify value correctness
        expected_results_file = get_expected_results_filename(is_gpu_available(verbose=False),
                                                              options=options)
        compare_results(inference_output_file,expected_results_file,options)


        # Make note of this filename, we will use it again later
        inference_output_file_standard_inference = inference_output_file

        if options.python_test_depth <= 2:
            return


        ## Run again with a batch size > 1

        print('\n** Running MD on a folder of images with batch size > 1 (module) **\n')

        from megadetector.detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file
        from megadetector.utils.path_utils import insert_before_extension

        inference_output_file_batch = insert_before_extension(inference_output_file,'batch')
        from megadetector.detection import run_detector_batch
        run_detector_batch.verbose = True
        results = load_and_run_detector_batch(options.default_model,
                                              image_file_names,
                                              quiet=True,
                                              batch_size=options.alternative_batch_size,
                                              detector_options=copy(options.detector_options))
        run_detector_batch.verbose = False
        _ = write_results_to_file(results,
                                  inference_output_file_batch,
                                  relative_path_base=image_folder,
                                  detector_file=options.default_model)

        expected_results_file = get_expected_results_filename(is_gpu_available(verbose=False),
                                                              options=options)
        compare_results(inference_output_file_batch,expected_results_file,options)

        ## Run and verify again with augmentation enabled

        print('\n** Running MD on images with augmentation (module) **\n')

        inference_output_file_augmented = insert_before_extension(inference_output_file,'augmented')
        results = load_and_run_detector_batch(options.default_model,
                                              image_file_names,
                                              quiet=True,
                                              augment=True,
                                              detector_options=copy(options.detector_options))
        _ = write_results_to_file(results,
                                  inference_output_file_augmented,
                                  relative_path_base=image_folder,
                                  detector_file=options.default_model)

        expected_results_file_augmented = \
            get_expected_results_filename(is_gpu_available(verbose=False),
                                          augment=True,options=options)
        compare_results(inference_output_file_augmented,expected_results_file_augmented,options)


        ## Postprocess results

        print('\n** Post-processing results (module) **\n')

        from megadetector.postprocessing.postprocess_batch_results import \
            PostProcessingOptions,process_batch_results
        postprocessing_options = PostProcessingOptions()

        postprocessing_options.md_results_file = inference_output_file
        postprocessing_options.output_dir = os.path.join(options.scratch_dir,'postprocessing_output')
        postprocessing_options.image_base_dir = image_folder

        postprocessing_results = process_batch_results(postprocessing_options)
        assert os.path.isfile(postprocessing_results.output_html_file), \
            'Postprocessing output file {} not found'.format(postprocessing_results.output_html_file)


        ## Partial RDE test

        print('\n** Testing RDE (module) **\n')

        from megadetector.postprocessing.repeat_detection_elimination.repeat_detections_core import \
            RepeatDetectionOptions, find_repeat_detections

        rde_options = RepeatDetectionOptions()
        rde_options.occurrenceThreshold = 2
        rde_options.confidenceMin = 0.001
        rde_options.outputBase = os.path.join(options.scratch_dir,'rde_working_dir')
        rde_options.imageBase = image_folder
        rde_output_file = inference_output_file.replace('.json','_filtered.json')
        assert rde_output_file != inference_output_file
        rde_results = find_repeat_detections(inference_output_file, rde_output_file, rde_options)
        assert os.path.isfile(rde_results.filterFile),\
            'Could not find RDE output file {}'.format(rde_results.filterFile)


        ## Run inference on a folder (with YOLOv5 val script)

        if options.yolo_working_dir is None:

            print('Skipping YOLO val inference tests, no YOLO folder supplied')

        else:

            print('\n** Running YOLO val inference test (module) **\n')

            from megadetector.detection.run_inference_with_yolov5_val import \
                YoloInferenceOptions, run_inference_with_yolo_val
            from megadetector.utils.path_utils import insert_before_extension

            inference_output_file_yolo_val = os.path.join(options.scratch_dir,'folder_inference_output_yolo_val.json')

            yolo_inference_options = YoloInferenceOptions()
            yolo_inference_options.input_folder = os.path.join(options.scratch_dir,'md-test-images')
            yolo_inference_options.output_file = inference_output_file_yolo_val
            yolo_inference_options.yolo_working_folder = options.yolo_working_dir
            yolo_inference_options.model_filename = options.default_model
            yolo_inference_options.augment = False
            yolo_inference_options.overwrite_handling = 'overwrite'
            from megadetector.detection.run_detector import DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
            yolo_inference_options.conf_thres = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD

            run_inference_with_yolo_val(yolo_inference_options)

            ## Confirm this matches the standard inference path

            if False:
                # TODO: compare_results() isn't quite ready for this yet
                compare_results(inference_output_file=inference_output_file_yolo_val,
                                expected_results_file=inference_output_file_standard_inference,
                                options=options)

            # Run again, without symlinks this time

            inference_output_file_yolo_val_no_links = insert_before_extension(inference_output_file_yolo_val,
                                                                            'no-links')
            yolo_inference_options.output_file = inference_output_file_yolo_val_no_links
            yolo_inference_options.use_symlinks = False
            run_inference_with_yolo_val(yolo_inference_options)

            # Run again, with chunked inference and symlinks

            inference_output_file_yolo_val_checkpoints = insert_before_extension(inference_output_file_yolo_val,
                                                                                'checkpoints')
            yolo_inference_options.output_file = inference_output_file_yolo_val_checkpoints
            yolo_inference_options.use_symlinks = True
            yolo_inference_options.checkpoint_frequency = 5
            run_inference_with_yolo_val(yolo_inference_options)

            # Run again, with chunked inference and no symlinks

            inference_output_file_yolo_val_checkpoints_no_links = \
                insert_before_extension(inference_output_file_yolo_val,'checkpoints-no-links')
            yolo_inference_options.output_file = inference_output_file_yolo_val_checkpoints_no_links
            yolo_inference_options.use_symlinks = False
            yolo_inference_options.checkpoint_frequency = 5
            run_inference_with_yolo_val(yolo_inference_options)

            fn1 = inference_output_file_yolo_val

            output_files_to_compare = [
                inference_output_file_yolo_val_no_links,
                inference_output_file_yolo_val_checkpoints,
                inference_output_file_yolo_val_checkpoints_no_links
                ]

            for fn2 in output_files_to_compare:
                assert output_files_are_identical(fn1, fn2, verbose=True)

        # ...if we need to run the YOLO val inference tests

    # ...if we're not skipping image tests

    if not options.skip_video_tests:

        ## Video test (single video)

        # This test just checks non-crashing-ness; we will test correctness in the next
        # test (which runs a folder of videos)

        print('\n** Running MD on a single video (module) **\n')

        from megadetector.detection.process_video import ProcessVideoOptions, process_videos
        from megadetector.utils.path_utils import insert_before_extension

        video_options = ProcessVideoOptions()
        video_options.model_file = options.default_model
        video_options.input_video_file = os.path.join(options.scratch_dir,options.test_videos[0])
        video_options.output_json_file = os.path.join(options.scratch_dir,'single_video_output.json')
        video_options.frame_sample = 10
        video_options.detector_options = copy(options.detector_options)

        _ = process_videos(video_options)

        assert os.path.isfile(video_options.output_json_file), \
            'Python video test failed to render output .json file'


        ## Video test (folder)

        print('\n** Running MD on a folder of videos (module) **\n')

        from megadetector.detection.process_video import ProcessVideoOptions, process_videos
        from megadetector.utils.path_utils import insert_before_extension

        video_options = ProcessVideoOptions()
        video_options.model_file = options.default_model
        video_options.input_video_file = os.path.join(options.scratch_dir,
                                                      os.path.dirname(options.test_videos[0]))
        video_options.output_json_file = os.path.join(options.scratch_dir,'video_folder_output.json')
        video_options.output_video_file = None
        video_options.recursive = True
        video_options.verbose = True
        video_options.json_confidence_threshold = 0.05
        video_options.time_sample = 2
        video_options.detector_options = copy(options.detector_options)
        _ = process_videos(video_options)

        assert os.path.isfile(video_options.output_json_file), \
            'Python video test failed to render output .json file'

        ## Verify results

        expected_results_file = \
            get_expected_results_filename(is_gpu_available(verbose=False),test_type='video',options=options)
        assert os.path.isfile(expected_results_file)

        from copy import deepcopy
        options_loose = deepcopy(options)
        options_loose.max_conf_error = 0.05
        options_loose.max_coord_error = 0.01

        compare_results(video_options.output_json_file,expected_results_file,options_loose)

    # ...if we're not skipping video tests

    print('\n*** Finished module tests ***\n')

# ...def run_python_tests(...)


#%% Command-line tests

def run_cli_tests(options):
    """
    Runs CLI (as opposed to Python-based) package tests.

    Args:
        options (MDTestOptions): see MDTestOptions for details
    """

    print('\n*** Starting CLI tests ***\n')

    ## Environment management

    if options.cli_test_pythonpath is not None:
        os.environ['PYTHONPATH'] = options.cli_test_pythonpath


    ## chdir if necessary

    if options.cli_working_dir is not None:
        os.chdir(options.cli_working_dir)


    ## Prepare data

    download_test_data(options)


    ## Utility imports

    from megadetector.utils.ct_utils import dict_to_kvp_list
    from megadetector.utils.path_utils import insert_before_extension


    ## Utility tests

    # TODO: move postprocessing tests up to this point, using pre-generated .json results files


    ## Return early if we're not running torch-related tests

    if options.test_mode == 'utils-only':
        print('utils-only tests finished, returning')
        return


    if not options.skip_image_tests:

        ## Run inference on an image

        print('\n** Running MD on a single image (CLI) **\n')

        image_fn = os.path.join(options.scratch_dir,options.test_images[0])
        output_dir = os.path.join(options.scratch_dir,'single_image_test')
        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.detection.run_detector'
        else:
            cmd = 'python megadetector/detection/run_detector.py'
        cmd += ' "{}" --image_file "{}" --output_dir "{}"'.format(
            options.default_model,image_fn,output_dir)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        if options.cpu_execution_is_error:
            gpu_available_via_cli = False
            for s in cmd_results['output']:
                if 'GPU available: True' in s:
                    gpu_available_via_cli = True
                    break
            if not gpu_available_via_cli:
                raise Exception('GPU execution is required, but not available')


        ## Make sure we can also pass an absolute path to a model file, instead of, e.g. "MDV5A"

        print('\n** Running MD on a single image (CLI) (with symbolic model name) **\n')

        from megadetector.detection.run_detector import try_download_known_detector
        model_file = try_download_known_detector(options.default_model,force_download=False,verbose=False)
        cmd = cmd.replace(options.default_model,model_file)
        cmd_results = execute_and_print(cmd)


        ## Run inference on a folder

        print('\n** Running MD on a folder (CLI) **\n')

        image_folder = os.path.join(options.scratch_dir,'md-test-images')
        assert os.path.isdir(image_folder), 'Test image folder {} is not available'.format(image_folder)
        inference_output_file = os.path.join(options.scratch_dir,'folder_inference_output.json')
        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.detection.run_detector_batch'
        else:
            cmd = 'python megadetector/detection/run_detector_batch.py'
        cmd += ' "{}" "{}" "{}" --recursive'.format(
            options.default_model,image_folder,inference_output_file)
        cmd += ' --output_relative_filenames --quiet --include_image_size'
        cmd += ' --include_image_timestamp --include_exif_data'

        base_cmd = cmd

        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)


        ## Run again with a batch size > 1

        print('\n** Running MD on a folder (with a batch size > 1) (CLI) **\n')

        batch_string = ' --batch_size {}'.format(options.alternative_batch_size)
        cmd = base_cmd + batch_string
        inference_output_file_batch = insert_before_extension(inference_output_file,'batch')
        cmd = cmd.replace(inference_output_file,inference_output_file_batch)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        # Use compare_results() here rather than output_files_are_identical(), because
        # batch inference may introduce very small differences. Override the default tolerance,
        # though, because these differences should be very small compared to, e.g., differences
        # across library versions.
        batch_options = copy(options)
        batch_options.max_coord_error = 0.01
        batch_options.max_conf_error = 0.01
        compare_results(inference_output_file,inference_output_file_batch,batch_options)


        ## Run again with the image queue enabled

        print('\n** Running MD on a folder (with image queue but consumer-side preprocessing) (CLI) **\n')

        cmd = base_cmd + ' --use_image_queue'
        inference_output_file_queue = insert_before_extension(inference_output_file,'queue')
        cmd = cmd.replace(inference_output_file,inference_output_file_queue)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        assert output_files_are_identical(fn1=inference_output_file,
                                        fn2=inference_output_file_queue,
                                        verbose=True)


        ## Run again with the image queue and worker-side preprocessing enabled

        print('\n** Running MD on a folder (with image queue and worker-side preprocessing) (CLI) **\n')

        cmd = base_cmd + ' --use_image_queue --preprocess_on_image_queue'
        inference_output_file_preprocess_queue = \
            insert_before_extension(inference_output_file,'preprocess_queue')
        cmd = cmd.replace(inference_output_file,inference_output_file_preprocess_queue)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        assert output_files_are_identical(fn1=inference_output_file,
                                        fn2=inference_output_file_preprocess_queue,
                                        verbose=True)


        ## Run again with the image queue and worker-side preprocessing

        print('\n** Running MD on a folder (with image queue and preprocessing) (CLI) **\n')

        cmd = base_cmd + ' --use_image_queue --preprocess_on_image_queue'
        inference_output_file_preprocess_queue = \
            insert_before_extension(inference_output_file,'preprocess_queue')
        cmd = cmd.replace(inference_output_file,inference_output_file_preprocess_queue)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        assert output_files_are_identical(fn1=inference_output_file,
                                        fn2=inference_output_file_preprocess_queue,
                                        verbose=True)


        ## Run again with the worker-side preprocessing and an alternative batch size

        print('\n** Running MD on a folder (with worker-side preprocessing and batched inference) (CLI) **\n')

        batch_string = ' --batch_size {}'.format(options.alternative_batch_size)

        # I reduce the number of loader workers here to force batching to actually append; with a small
        # number of images and a few that are intentionally corrupt, with the default number of loader
        # workers we end up with batches that are mostly just one image.
        cmd = base_cmd + ' --use_image_queue --preprocess_on_image_queue --loader_workers 2' + batch_string
        inference_output_file_queue_batch = \
            insert_before_extension(inference_output_file,'preprocess_queue_batch')
        cmd = cmd.replace(inference_output_file,inference_output_file_queue_batch)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        compare_results(inference_output_file,inference_output_file_queue_batch,batch_options)


        ## Run again with checkpointing enabled

        print('\n** Running MD on a folder (with checkpoints) (CLI) **\n')

        checkpoint_string = ' --checkpoint_frequency 5'
        cmd = base_cmd + checkpoint_string
        inference_output_file_checkpoint = insert_before_extension(inference_output_file,'checkpoint')
        cmd = cmd.replace(inference_output_file,inference_output_file_checkpoint)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        assert output_files_are_identical(fn1=inference_output_file,
                                        fn2=inference_output_file_checkpoint,
                                        verbose=True)


        ## Run again with "modern" postprocessing, make sure the results are *not* the same as classic

        print('\n** Running MD on a folder (with modern preprocessing) (CLI) **\n')

        inference_output_file_modern = insert_before_extension(inference_output_file,'modern')
        cmd = base_cmd
        cmd = cmd.replace(inference_output_file,inference_output_file_modern)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list({'compatibility_mode':'modern'}))
        cmd_results = execute_and_print(cmd)

        assert not output_files_are_identical(fn1=inference_output_file,
                                            fn2=inference_output_file_modern,
                                            verbose=True)


        ## Run again with "modern" postprocessing and worker-side preprocessing,
        ## make sure the results are the same as modern.

        print('\n** Running MD on a folder (with worker-side modern preprocessing) (CLI) **\n')

        inference_output_file_modern_worker_preprocessing = insert_before_extension(inference_output_file,'modern')
        cmd = base_cmd + ' --use_image_queue --preprocess_on_image_queue'
        cmd = cmd.replace(inference_output_file,inference_output_file_modern_worker_preprocessing)
        cmd += ' --detector_options {}'.format(dict_to_kvp_list({'compatibility_mode':'modern'}))
        cmd_results = execute_and_print(cmd)

        # This should not be the same as the "classic" results
        assert not output_files_are_identical(fn1=inference_output_file,
                                            fn2=inference_output_file_modern_worker_preprocessing,
                                            verbose=True)

        # ...but it should be the same as the single-threaded "modern" results
        assert output_files_are_identical(fn1=inference_output_file_modern,
                                        fn2=inference_output_file_modern_worker_preprocessing,
                                        verbose=True)


        if not options.skip_cpu_tests:

            ## Run again on multiple cores

            # First run again on the CPU on a single thread if necessary, so we get a file that
            # *should* be identical to the multicore version.
            gpu_available = is_gpu_available(verbose=False)

            cuda_visible_devices = None
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            # If we already ran on the CPU, no need to run again
            if not gpu_available:

                inference_output_file_cpu = inference_output_file

            else:

                print('\n** Running MD on a folder (single CPU) (CLI) **\n')

                inference_output_file_cpu = insert_before_extension(inference_output_file,'cpu')
                cmd = base_cmd
                cmd = cmd.replace(inference_output_file,inference_output_file_cpu)
                cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
                cmd_results = execute_and_print(cmd)

            print('\n** Running MD on a folder (multiple CPUs) (CLI) **\n')

            cpu_string = ' --ncores {}'.format(options.n_cores_for_multiprocessing_tests)
            cmd = base_cmd + cpu_string
            inference_output_file_cpu_multicore = insert_before_extension(inference_output_file,'multicore')
            cmd = cmd.replace(inference_output_file,inference_output_file_cpu_multicore)
            cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
            cmd_results = execute_and_print(cmd)

            if cuda_visible_devices is not None:
                print('Restoring CUDA_VISIBLE_DEVICES')
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
            else:
                del os.environ['CUDA_VISIBLE_DEVICES']

            assert output_files_are_identical(fn1=inference_output_file_cpu,
                                            fn2=inference_output_file_cpu_multicore,
                                            verbose=True)

        # ...if we're not skipping the force-cpu tests


        ## Postprocessing

        print('\n** Testing post-processing (CLI) **\n')

        postprocessing_output_dir = os.path.join(options.scratch_dir,'postprocessing_output_cli')

        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.postprocessing.postprocess_batch_results'
        else:
            cmd = 'python megadetector/postprocessing/postprocess_batch_results.py'
        cmd += ' "{}" "{}"'.format(
            inference_output_file,postprocessing_output_dir)
        cmd += ' --image_base_dir "{}"'.format(image_folder)
        cmd_results = execute_and_print(cmd)


        ## RDE

        print('\n** Running RDE (CLI) **\n')

        rde_output_dir = os.path.join(options.scratch_dir,'rde_output_cli')

        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.postprocessing.repeat_detection_elimination.find_repeat_detections'
        else:
            cmd = 'python  megadetector/postprocessing/repeat_detection_elimination/find_repeat_detections.py'
        cmd += ' "{}"'.format(inference_output_file)
        cmd += ' --imageBase "{}"'.format(image_folder)
        cmd += ' --outputBase "{}"'.format(rde_output_dir)
        cmd += ' --occurrenceThreshold 1' # Use an absurd number here to make sure we get some suspicious detections
        cmd_results = execute_and_print(cmd)

        # Find the latest filtering folder
        filtering_output_dir = os.listdir(rde_output_dir)
        filtering_output_dir = [fn for fn in filtering_output_dir if fn.startswith('filtering_')]
        filtering_output_dir = [os.path.join(rde_output_dir,fn) for fn in filtering_output_dir]
        filtering_output_dir = [fn for fn in filtering_output_dir if os.path.isdir(fn)]
        filtering_output_dir = sorted(filtering_output_dir)[-1]

        print('Using RDE filtering folder {}'.format(filtering_output_dir))

        filtered_output_file = inference_output_file.replace('.json','_filtered.json')

        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.postprocessing.repeat_detection_elimination.remove_repeat_detections'
        else:
            cmd = 'python  megadetector/postprocessing/repeat_detection_elimination/remove_repeat_detections.py'
        cmd += ' "{}" "{}" "{}"'.format(inference_output_file,filtered_output_file,filtering_output_dir)
        cmd_results = execute_and_print(cmd)

        assert os.path.isfile(filtered_output_file), \
            'Could not find RDE output file {}'.format(filtered_output_file)


        ## Run inference on a folder (tiled)

        # This is a rather esoteric code path that I turn off when I'm testing some
        # features that it doesn't include yet, particularly compatibility mode
        # control.
        skip_tiling_tests = True

        if skip_tiling_tests:

            print('### DEBUG: skipping tiling tests ###')

        else:
            print('\n** Running tiled inference (CLI) **\n')

            image_folder = os.path.join(options.scratch_dir,'md-test-images')
            tiling_folder = os.path.join(options.scratch_dir,'tiling-folder')
            inference_output_file_tiled = os.path.join(options.scratch_dir,'folder_inference_output_tiled.json')
            if options.cli_working_dir is None:
                cmd = 'python -m megadetector.detection.run_tiled_inference'
            else:
                cmd = 'python megadetector/detection/run_tiled_inference.py'
            cmd += ' "{}" "{}" "{}" "{}"'.format(
                options.default_model,image_folder,tiling_folder,inference_output_file_tiled)
            cmd += ' --overwrite_handling overwrite'
            cmd_results = execute_and_print(cmd)

            with open(inference_output_file_tiled,'r') as f:
                results_from_file = json.load(f) # noqa


        ## Run inference on a folder (augmented, w/YOLOv5 val script)

        if options.yolo_working_dir is None:

            print('Bypassing YOLOv5 val tests, no yolo folder supplied')

        else:

            print('\n** Running YOLOv5 val tests (CLI) **\n')

            image_folder = os.path.join(options.scratch_dir,'md-test-images')
            yolo_results_folder = os.path.join(options.scratch_dir,'yolo-output-folder')
            yolo_symlink_folder = os.path.join(options.scratch_dir,'yolo-symlink_folder')
            inference_output_file_yolo_val = os.path.join(options.scratch_dir,'folder_inference_output_yolo_val.json')
            if options.cli_working_dir is None:
                cmd = 'python -m megadetector.detection.run_inference_with_yolov5_val'
            else:
                cmd = 'python megadetector/detection/run_inference_with_yolov5_val.py'
            cmd += ' "{}" "{}" "{}"'.format(
                options.default_model,image_folder,inference_output_file_yolo_val)
            cmd += ' --yolo_working_folder "{}"'.format(options.yolo_working_dir)
            cmd += ' --yolo_results_folder "{}"'.format(yolo_results_folder)
            cmd += ' --symlink_folder "{}"'.format(yolo_symlink_folder)
            cmd += ' --augment_enabled 1'
            # cmd += ' --no_use_symlinks'
            cmd += ' --overwrite_handling overwrite'
            cmd_results = execute_and_print(cmd)

            # Run again with checkpointing, make sure the outputs are identical
            cmd += ' --checkpoint_frequency 5'
            inference_output_file_yolo_val_checkpoint = \
                os.path.join(options.scratch_dir,'folder_inference_output_yolo_val_checkpoint.json')
            assert inference_output_file_yolo_val_checkpoint != inference_output_file_yolo_val
            cmd = cmd.replace(inference_output_file_yolo_val,inference_output_file_yolo_val_checkpoint)
            cmd_results = execute_and_print(cmd)

            assert output_files_are_identical(fn1=inference_output_file_yolo_val,
                                            fn2=inference_output_file_yolo_val_checkpoint,
                                            verbose=True)


        ## Run inference on a folder (with MDV5B, so we can do a comparison)

        print('\n** Running MDv5b (CLI) **\n')

        image_folder = os.path.join(options.scratch_dir,'md-test-images')
        inference_output_file_alt = os.path.join(options.scratch_dir,'folder_inference_output_alt.json')
        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.detection.run_detector_batch'
        else:
            cmd = 'python megadetector/detection/run_detector_batch.py'
        cmd += ' "{}" "{}" "{}" --recursive'.format(
            options.alt_model,image_folder,inference_output_file_alt)
        cmd += ' --output_relative_filenames --quiet --include_image_size'
        cmd += ' --include_image_timestamp --include_exif_data'
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))
        cmd_results = execute_and_print(cmd)

        with open(inference_output_file_alt,'r') as f:
            results_from_file = json.load(f) # noqa


        ## Compare the two files

        comparison_output_folder = os.path.join(options.scratch_dir,'results_comparison')
        image_folder = os.path.join(options.scratch_dir,'md-test-images')
        results_files_string = '"{}" "{}"'.format(
            inference_output_file,inference_output_file_alt)
        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.postprocessing.compare_batch_results'
        else:
            cmd = 'python megadetector/postprocessing/compare_batch_results.py'
        cmd += ' "{}" "{}" {}'.format(comparison_output_folder,image_folder,results_files_string)
        cmd_results = execute_and_print(cmd)

        assert cmd_results['status'] == 0, 'Error generating comparison HTML'
        assert os.path.isfile(os.path.join(comparison_output_folder,'index.html')), \
            'Failed to generate comparison HTML'

    # ...if we're not skipping image tests


    if not options.skip_video_tests:

        ## Video test

        print('\n** Testing video processing (CLI) **\n')

        video_inference_output_file = os.path.join(options.scratch_dir,'video_folder_output_cli.json')
        if options.cli_working_dir is None:
            cmd = 'python -m megadetector.detection.process_video'
        else:
            cmd = 'python megadetector/detection/process_video.py'

        cmd += ' "{}" "{}"'.format(options.default_model,options.scratch_dir)
        cmd += ' --output_json_file "{}"'.format(video_inference_output_file)
        cmd += ' --frame_sample 4'
        cmd += ' --verbose'
        cmd += ' --recursive'
        cmd += ' --detector_options {}'.format(dict_to_kvp_list(options.detector_options))

        cmd_results = execute_and_print(cmd)

    # ...if we're not skipping video tests

    print('\n*** Finished CLI tests ***\n')

# ...def run_cli_tests(...)


def run_download_tests(options):
    """
    Test automatic model downloads.

    Args:
        options (MDTestOptions): see MDTestOptions for details
    """

    if options.skip_download_tests or options.test_mode == 'utils-only':
        return

    from megadetector.detection.run_detector import known_models, \
        try_download_known_detector, \
        get_detector_version_from_model_file, \
        model_string_to_model_version

    # Make sure we can download models based on canonical version numbers,
    # e.g. "v5a.0.0"
    for model_name in known_models:
        url = known_models[model_name]['url']
        if ('localhost' in url) and options.skip_localhost_downloads:
            continue
        print('Testing download for known model {}'.format(model_name))
        fn = try_download_known_detector(model_name,
                                         force_download=False,
                                         verbose=False)
        version_string = get_detector_version_from_model_file(fn, verbose=False)
        # Make sure this is the same version we asked for, modulo the MDv5 re-releases
        assert (version_string.replace('.0.1','.0.0') == model_name.replace('.0.1','.0.0'))

    # Make sure we can download models based on short names, e.g. "MDV5A"
    for model_name in model_string_to_model_version:
        model_version = model_string_to_model_version[model_name]
        assert model_version in known_models
        url = known_models[model_version]['url']
        if 'localhost' in url:
            continue
        print('Testing download for model short name {}'.format(model_name))
        fn = try_download_known_detector(model_name,
                                         force_download=False,
                                         verbose=False)
        assert fn != model_name

    # Test corruption handling for .pt files
    print('Testing corruption handling for MDV5B')

    # First ensure MDV5B is downloaded
    mdv5b_file = try_download_known_detector('MDV5B',
                                             force_download=False,
                                             verbose=False)
    assert mdv5b_file is not None
    assert os.path.exists(mdv5b_file)
    assert mdv5b_file.endswith('.pt')

    # Get the original file size and MD5 hash for comparison
    original_size = os.path.getsize(mdv5b_file)
    from megadetector.utils.path_utils import compute_file_hash
    original_hash = compute_file_hash(mdv5b_file, algorithm='md5')

    # Deliberately corrupt the file by overwriting the first few bytes
    print('Corrupting model file: {}'.format(mdv5b_file))
    with open(mdv5b_file, 'r+b') as f:
        f.write(b'CORRUPTED_FILE_DATA_XXXXXX')

    # Verify the file is now corrupted (different hash)
    corrupted_hash = compute_file_hash(mdv5b_file, algorithm='md5')
    assert corrupted_hash != original_hash, 'File corruption verification failed'

    # Try to download again; this should detect corruption and re-download
    print('Testing corruption detection and re-download')
    mdv5b_file_redownloaded = try_download_known_detector('MDV5B',
                                                          force_download=False,
                                                          verbose=True)

    # Verify that the file was re-downloaded and is now valid
    assert mdv5b_file_redownloaded is not None
    assert os.path.exists(mdv5b_file_redownloaded)
    assert mdv5b_file_redownloaded == mdv5b_file

    # Verify that the file is back to its original state
    new_size = os.path.getsize(mdv5b_file_redownloaded)
    new_hash = compute_file_hash(mdv5b_file_redownloaded, algorithm='md5')

    assert new_size == original_size, \
        'Re-downloaded file size ({}) does not match original ({})'.format(new_size, original_size)
    assert new_hash == original_hash, \
        'Re-downloaded file hash ({}) does not match original ({})'.format(new_hash, original_hash)

    print('Corruption handling test passed')

# ...def run_download_tests()


#%% Main test wrapper

def run_tests(options):
    """
    Runs Python-based and/or CLI-based package tests.

    Args:
        options (MDTestOptions): see MDTestOptions for details
    """

    # Prepare data folder
    download_test_data(options)

    # Run model download tests if necessary
    run_download_tests(options)

    if options.disable_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Verify GPU
    gpu_available = is_gpu_available()

    # If the GPU is required and isn't available, error
    if options.cpu_execution_is_error and (not gpu_available):
        raise ValueError('GPU not available, and cpu_execution_is_error is set')

    # If the GPU should be disabled, verify that it is
    if options.disable_gpu:
        assert (not gpu_available), 'CPU execution specified, but the GPU appears to be available'

    # Run python tests
    if not options.skip_python_tests:

        if options.model_folder is not None:

            assert os.path.isdir(options.model_folder), \
                'Could not find model folder {}'.format(options.model_folder)

            model_files = os.listdir(options.model_folder)
            model_files = [fn for fn in model_files if fn.endswith('.pt')]
            model_files = [os.path.join(options.model_folder,fn) for fn in model_files]

            assert len(model_files) > 0, \
                'Could not find any models in folder {}'.format(options.model_folder)

            original_default_model = options.default_model

            for model_file in model_files:
                print('Running Python tests for model {}'.format(model_file))
                options.default_model = model_file
                run_python_tests(options)

            options.default_model = original_default_model

        else:

            run_python_tests(options)

    # Run CLI tests
    if not options.skip_cli_tests:
        run_cli_tests(options)


#%% Automated test entry point

def test_suite_entry_point():
    """
    This is the entry point when running tests via pytest; we run a subset of
    tests in this environment, e.g. we don't run CLI or video tests.
    """

    options = MDTestOptions()
    options.disable_gpu = False
    options.cpu_execution_is_error = False
    options.skip_video_tests = True
    options.skip_python_tests = False
    options.skip_cli_tests = True
    options.scratch_dir = None
    options.test_data_url = 'https://lila.science/public/md-test-package.zip'
    options.force_data_download = False
    options.force_data_unzip = False
    options.warning_mode = False
    options.max_coord_error = 0.01 # 0.001
    options.max_conf_error = 0.01 # 0.005
    options.skip_video_rendering_tests = True
    options.cli_working_dir = None
    options.cli_test_pythonpath = None
    options.skip_download_tests = True
    options.skip_localhost_downloads = True
    options.skip_import_tests = False

    if sys.platform == 'darwin':
        print('Detected a Mac environment, widening tolerance')
        options.max_coord_error = 0.05
        options.max_conf_error = 0.05

    options = download_test_data(options)

    run_tests(options)


#%% Interactive driver

if False:

    pass

    #%% Test Prep

    from megadetector.utils.md_tests import MDTestOptions, download_test_data

    options = MDTestOptions()

    options.disable_gpu = False
    options.cpu_execution_is_error = False
    options.skip_video_tests = True
    options.skip_python_tests = True
    options.skip_cli_tests = False
    options.scratch_dir = None
    options.test_data_url = 'https://lila.science/public/md-test-package.zip'
    options.force_data_download = False
    options.force_data_unzip = False
    options.warning_mode = False
    options.max_coord_error = 0.01 # 0.001
    options.max_conf_error = 0.01 # 0.005
    options.skip_cpu_tests = True
    options.skip_video_rendering_tests = True
    options.skip_download_tests = True
    options.skip_localhost_downloads = False

    # options.iou_threshold_for_file_comparison = 0.7

    # options.cli_working_dir = r'c:\git\MegaDetector'
    # When running in the cameratraps-detector environment
    # options.cli_test_pythonpath = r'c:\git\MegaDetector;c:\git\yolov5-md'

    # When running in the MegaDetector environment
    # options.cli_test_pythonpath = r'c:\git\MegaDetector'

    # options.cli_working_dir = os.path.expanduser('~')
    # options.yolo_working_dir = r'c:\git\yolov5-md'
    # options.yolo_working_dir = '/mnt/c/git/yolov5-md'
    options = download_test_data(options)


    #%% Environment prep

    # Add the YOLO working dir to the PYTHONPATH if necessary
    import os
    if (options.yolo_working_dir is not None) and \
        (('PYTHONPATH' not in os.environ) or (options.yolo_working_dir not in os.environ['PYTHONPATH'])):
        if ('PYTHONPATH' not in os.environ):
            os.environ['PYTHONPATH'] = options.yolo_working_dir
        else:
            os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ';' + options.yolo_working_dir


    #%% Run download tests

    from megadetector.utils.md_tests import run_download_tests
    run_download_tests(options=options)


    #%% Run all tests

    from megadetector.utils.md_tests import run_tests
    run_tests(options)


    #%% Run YOLO inference tests

    yolo_inference_options_dict = {'input_folder': '/tmp/md-tests/md-test-images',
                                   'image_filename_list': None,
                                   'model_filename': 'MDV5A',
                                   'output_file': '/tmp/md-tests/folder_inference_output_yolo_val.json',
                                   'yolo_working_folder': '/mnt/c/git/yolov5-md',
                                   'model_type': 'yolov5',
                                   'image_size': None,
                                   'conf_thres': 0.005,
                                   'batch_size': 1,
                                   'device_string': '0',
                                   'augment': False,
                                   'half_precision_enabled': None,
                                   'symlink_folder': None,
                                   'use_symlinks': True,
                                   'unique_id_strategy': 'links',
                                   'yolo_results_folder': None,
                                   'remove_symlink_folder': True,
                                   'remove_yolo_results_folder': True,
                                   'yolo_category_id_to_name': {0: 'animal', 1: 'person', 2: 'vehicle'},
                                   'overwrite_handling': 'overwrite',
                                   'preview_yolo_command_only': False,
                                   'treat_copy_failures_as_warnings': False,
                                   'save_yolo_debug_output': False,
                                   'recursive': True,
                                   'checkpoint_frequency': None}

    from megadetector.utils.ct_utils import dict_to_object
    from megadetector.detection.run_inference_with_yolov5_val import \
        YoloInferenceOptions, run_inference_with_yolo_val

    yolo_inference_options = YoloInferenceOptions()
    yolo_inference_options = dict_to_object(yolo_inference_options_dict, yolo_inference_options)

    os.makedirs(options.scratch_dir,exist_ok=True)

    inference_output_file_yolo_val = os.path.join(options.scratch_dir,'folder_inference_output_yolo_val.json')

    run_inference_with_yolo_val(yolo_inference_options)


#%% Command-line driver

def main(): # noqa

    options = MDTestOptions()

    parser = argparse.ArgumentParser(
        description='MegaDetector test suite')

    parser.add_argument(
        '--disable_gpu',
        action='store_true',
        help='Disable GPU operation')

    parser.add_argument(
        '--cpu_execution_is_error',
        action='store_true',
        help='Fail if the GPU appears not to be available')

    parser.add_argument(
        '--scratch_dir',
        default=None,
        type=str,
        help='Directory for temporary storage (defaults to system temp dir)')

    parser.add_argument(
        '--skip_image_tests',
        action='store_true',
        help='Skip tests related to still images')

    parser.add_argument(
        '--skip_video_tests',
        action='store_true',
        help='Skip tests related to video')

    parser.add_argument(
        '--skip_video_rendering_tests',
        action='store_true',
        help='Skip tests related to *rendering* video')

    parser.add_argument(
        '--skip_python_tests',
        action='store_true',
        help='Skip python tests')

    parser.add_argument(
        '--skip_cli_tests',
        action='store_true',
        help='Skip CLI tests')

    parser.add_argument(
        '--skip_download_tests',
        action='store_true',
        help='Skip model download tests')

    parser.add_argument(
        '--skip_import_tests',
        action='store_true',
        help='Skip module import tests')

    parser.add_argument(
        '--skip_cpu_tests',
        action='store_true',
        help='Skip force-CPU tests')

    parser.add_argument(
        '--force_data_download',
        action='store_true',
        help='Force download of the test data file, even if it\'s already available')

    parser.add_argument(
        '--force_data_unzip',
        action='store_true',
        help='Force extraction of all files in the test data file, even if they\'re already available')

    parser.add_argument(
        '--warning_mode',
        action='store_true',
        help='Turns numeric/content errors into warnings')

    parser.add_argument(
        '--max_conf_error',
        type=float,
        default=options.max_conf_error,
        help='Maximum tolerable confidence value deviation from expected (default {})'.format(
            options.max_conf_error))

    parser.add_argument(
        '--max_coord_error',
        type=float,
        default=options.max_coord_error,
        help='Maximum tolerable coordinate value deviation from expected (default {})'.format(
            options.max_coord_error))

    parser.add_argument(
        '--cli_working_dir',
        type=str,
        default=None,
        help='Working directory for CLI tests')

    parser.add_argument(
        '--yolo_working_dir',
        type=str,
        default=None,
        help='Working directory for yolo inference tests')

    parser.add_argument(
        '--cli_test_pythonpath',
        type=str,
        default=None,
        help='PYTHONPATH to set for CLI tests; if None, inherits from the parent process'
        )

    parser.add_argument(
        '--test_mode',
        type=str,
        default='all',
        help='Test mode: "all" or "utils-only"'
        )

    parser.add_argument(
        '--python_test_depth',
        type=int,
        default=options.python_test_depth,
        help='Used as a knob to control the level of Python tests (0-100)'
        )

    parser.add_argument(
        '--model_folder',
        type=str,
        default=None,
        help='Run Python tests on every model in this folder'
        )

    parser.add_argument(
        '--detector_options',
        nargs='*',
        metavar='KEY=VALUE',
        default='',
        help='Detector-specific options, as a space-separated list of key-value pairs')

    parser.add_argument(
        '--default_model',
        type=str,
        default=options.default_model,
        help='Default model file or well-known model name (used for most tests)')

    # The following token is used for linting, do not remove.
    #
    # no_arguments_required

    args = parser.parse_args()

    initial_detector_options = options.detector_options
    _args_to_object(args,options)
    from megadetector.utils.ct_utils import parse_kvp_list
    options.detector_options = parse_kvp_list(args.detector_options,d=initial_detector_options)

    run_tests(options)

# ...def main()

if __name__ == '__main__':
    main()

