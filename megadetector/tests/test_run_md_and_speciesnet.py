"""

Tests for run_md_and_speciesnet.py that don't run inference: argument validation,
output file overwrite handling, and the wiring of the detection and classification
steps.  The two step functions are replaced with stubs that record their arguments
and write stand-in output files.

"""

#%% Imports

import json
import os
import shutil

import pytest

from megadetector.detection import run_md_and_speciesnet as mdns
from megadetector.utils.ct_utils import make_test_folder


#%% Support functions

def _make_options(test_folder,case_name):
    """
    Create a RunMDSpeciesNetOptions object for one test case, with a real (empty) source
    folder and an output file in a subfolder of [test_folder] called [case_name].
    """

    case_folder = os.path.join(test_folder,case_name)
    source_folder = os.path.join(case_folder,'images')
    os.makedirs(source_folder)

    options = mdns.RunMDSpeciesNetOptions()
    options.source = source_folder
    options.output_file = os.path.join(case_folder,'output.json')

    return options


def _write_empty_results(output_file):
    """
    Write a minimal MD-formatted results file.
    """

    with open(output_file,'w') as f:
        json.dump({'info':{},'images':[]},f)


class _StubbedSteps:
    """
    Context manager that replaces the detection and classification steps with stubs that
    record their arguments and write minimal results files.  Returns the list of recorded
    calls, in the order the steps were run.
    """

    def __init__(self):
        self.calls = []

    def _detection_step(self,source_folder,detector_output_file,**kwargs):
        self.calls.append({'step':'detection',
                           'detector_output_file':detector_output_file,
                           **kwargs})
        _write_empty_results(detector_output_file)

    def _classification_step(self,detector_results_file,merged_results_file,source_folder,**kwargs):
        self.calls.append({'step':'classification',
                           'detector_results_file':detector_results_file,
                           'merged_results_file':merged_results_file,
                           **kwargs})
        _write_empty_results(merged_results_file)

    def __enter__(self):
        self.original_detection_step = mdns._run_detection_step
        self.original_classification_step = mdns._run_classification_step
        mdns._run_detection_step = self._detection_step
        mdns._run_classification_step = self._classification_step
        return self.calls

    def __exit__(self,exc_type,exc_value,traceback):
        mdns._run_detection_step = self.original_detection_step
        mdns._run_classification_step = self.original_classification_step

# ...class _StubbedSteps


#%% Tests

def test_argument_validation():
    """
    Check that invalid argument combinations raise a ValueError before any inference
    step runs, and that the video sampling default is applied when neither sampling
    option is set.
    """

    test_folder = make_test_folder(subfolder='run_md_and_speciesnet_tests')

    try:

        # The steps are stubbed out here too, so a missing check fails the test rather
        # than starting inference
        with _StubbedSteps() as calls:

            options = _make_options(test_folder,'missing_source')
            options.source = os.path.join(test_folder,'no-such-folder')
            with pytest.raises(ValueError, match='Source folder does not exist'):
                mdns.run_md_and_speciesnet(options)

            options = _make_options(test_folder,'admin1_without_country')
            options.admin1_region = 'US-CA'
            with pytest.raises(ValueError, match='requires --country'):
                mdns.run_md_and_speciesnet(options)

            options = _make_options(test_folder,'skip_everything')
            options.skip_images = True
            options.skip_video = True
            with pytest.raises(ValueError, match='Cannot skip both images and videos'):
                mdns.run_md_and_speciesnet(options)

            options = _make_options(test_folder,'frame_and_time_sample')
            options.frame_sample = 10
            options.time_sample = 2.0
            with pytest.raises(ValueError, match='mutually exclusive'):
                mdns.run_md_and_speciesnet(options)

            options = _make_options(test_folder,'unknown_worker_type')
            options.worker_type = 'quantum'
            with pytest.raises(ValueError, match='Unknown worker type'):
                mdns.run_md_and_speciesnet(options)

        assert len(calls) == 0

        # With neither video sampling option set, time_sample gets the default, and
        # that's what the detection step sees
        options = _make_options(test_folder,'default_time_sample')
        options.frame_sample = None
        options.time_sample = None
        with _StubbedSteps() as calls:
            mdns.run_md_and_speciesnet(options)
        assert options.time_sample == mdns.DEFAULT_SECONDS_PER_VIDEO_FRAME
        assert calls[0]['step'] == 'detection'
        assert calls[0]['time_sample'] == mdns.DEFAULT_SECONDS_PER_VIDEO_FRAME

    finally:

        shutil.rmtree(test_folder)

# ...def test_argument_validation()


def test_overwrite_handling():
    """
    Check each overwrite_handling mode when the output file already exists, and that an
    output path that is a folder is rejected.
    """

    test_folder = make_test_folder(subfolder='run_md_and_speciesnet_tests')

    try:

        options = _make_options(test_folder,'output_is_a_folder')
        os.makedirs(options.output_file)
        with pytest.raises(ValueError, match='exists, but is a directory'):
            mdns.run_md_and_speciesnet(options)

        ## "error"

        options = _make_options(test_folder,'overwrite_error')
        with open(options.output_file,'w') as f:
            f.write('existing')
        options.overwrite_handling = 'error'
        with _StubbedSteps() as calls:
            with pytest.raises(ValueError, match='overwrite_handling is "error"'):
                mdns.run_md_and_speciesnet(options)
        assert len(calls) == 0

        ## "skip": neither step runs, and the existing file is left alone

        options = _make_options(test_folder,'overwrite_skip')
        with open(options.output_file,'w') as f:
            f.write('existing')
        options.overwrite_handling = 'skip'
        with _StubbedSteps() as calls:
            mdns.run_md_and_speciesnet(options)
        assert len(calls) == 0, 'overwrite_handling "skip" should not run either step'
        with open(options.output_file,'r') as f:
            assert f.read() == 'existing', \
                'overwrite_handling "skip" should not modify the existing output file'

        ## "overwrite": both steps run, and the classification step writes the output file

        options = _make_options(test_folder,'overwrite_overwrite')
        with open(options.output_file,'w') as f:
            f.write('existing')
        options.overwrite_handling = 'overwrite'
        with _StubbedSteps() as calls:
            mdns.run_md_and_speciesnet(options)
        assert [c['step'] for c in calls] == ['detection','classification']
        assert calls[1]['merged_results_file'] == options.output_file
        with open(options.output_file,'r') as f:
            assert json.load(f) == {'info':{},'images':[]}

    finally:

        shutil.rmtree(test_folder)

# ...def test_overwrite_handling()


def test_step_wiring():
    """
    Check that an existing detections file bypasses the detection step, and that the
    intermediate detection results file is removed unless keep_intermediate_files is set.
    """

    test_folder = make_test_folder(subfolder='run_md_and_speciesnet_tests')

    try:

        ## An existing detections file bypasses the detection step

        options = _make_options(test_folder,'existing_detections')
        detections_file = os.path.join(test_folder,'existing_detections','detections.json')
        _write_empty_results(detections_file)
        options.detections_file = detections_file
        with _StubbedSteps() as calls:
            mdns.run_md_and_speciesnet(options)
        assert [c['step'] for c in calls] == ['classification']
        assert calls[0]['detector_results_file'] == detections_file
        assert os.path.isfile(detections_file), 'a supplied detections file should not be removed'

        ## By default, the intermediate detection results file is removed

        options = _make_options(test_folder,'default_cleanup')
        with _StubbedSteps() as calls:
            mdns.run_md_and_speciesnet(options)
        detector_output_file = calls[0]['detector_output_file']
        assert calls[1]['detector_results_file'] == detector_output_file
        assert not os.path.exists(detector_output_file), \
            'the intermediate detection results file should have been removed'

        ## keep_intermediate_files keeps it

        options = _make_options(test_folder,'keep_intermediate_files')
        options.keep_intermediate_files = True
        with _StubbedSteps() as calls:
            mdns.run_md_and_speciesnet(options)
        detector_output_file = calls[0]['detector_output_file']
        try:
            assert os.path.isfile(detector_output_file), \
                'the intermediate detection results file should have been kept'
        finally:
            if os.path.isfile(detector_output_file):
                os.remove(detector_output_file)

    finally:

        shutil.rmtree(test_folder)

# ...def test_step_wiring()
