"""

test_batch_inference_harness.py

Test harness to validate that batch inference produces identical results to single-image inference.

"""

#%% Imports and constants

import os
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional

from megadetector.utils import url_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector
from megadetector.utils.md_tests import compare_detection_lists, MDTestOptions
from megadetector.utils.path_utils import find_images

default_model = 'MDV5A'
default_image_folder = '/mnt/g/temp/md-test-images'


#%% Test harness class

class BatchInferenceTestHarness:
    """
    Test harness to validate batch inference implementation.
    """

    def __init__(self, test_images_dir: str = default_image_folder):
        """
        Initialize the test harness.

        Args:
            test_images_dir (str): Directory containing test images
        """

        self.test_images_dir = test_images_dir
        self.reference_results_file = None
        self.temp_dir = None

        # Test options for comparison
        self.test_options = MDTestOptions()
        self.test_options.max_conf_error = 0.001
        self.test_options.max_coord_error = 0.001

        if False:
            self.test_images = [
                'snapshot_camdeboo_CDB_S1_A05_A05_R2_CDB_S1_A05_R2_IMAG0084.JPG',
                'idaho_camera_traps_loc_0044_loc_0044_im_005629.jpg',
                'corrupt-images/irfanview-can-still-read-me-caltech_camera_traps_5a0e37cc-23d2-11e8-a6a3-ec086b02610b.jpg'
            ]
        self.test_images = find_images(test_images_dir, return_relative_paths=True, recursive=False)

    def setup_temp_directory(self):
        """
        Create temporary directory for test files.
        """

        self.temp_dir = tempfile.mkdtemp(prefix='batch_inference_test_')
        print(f'Created temporary directory: {self.temp_dir}')

    def cleanup_temp_directory(self):
        """
        Clean up temporary directory.
        """

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f'Cleaned up temporary directory: {self.temp_dir}')

    def generate_reference_results(self, model_name: str = default_model) -> str:
        """
        Generate reference results using current generate_detections_one_image().

        Args:
            model_name (str): Model name to load

        Returns:
            str: Path to reference results file
        """

        print(f'Generating reference results with model {model_name}...')

        # Load model
        model = run_detector.load_detector(model_name)

        reference_results = []

        for image_file in self.test_images:
            image_path = os.path.join(self.test_images_dir, image_file)

            if not os.path.exists(image_path):
                print(f'Warning: test image not found: {image_path}')
                continue

            print(f'Processing reference image: {image_file}')

            # Load image
            image = vis_utils.load_image(image_path)

            # Generate detections using current implementation
            result = model.generate_detections_one_image(image, image_id=image_file)

            reference_results.append({
                'image_file': image_file,
                'image_path': image_path,
                'result': result
            })

        # Save reference results
        self.reference_results_file = os.path.join(self.temp_dir, 'reference_results.json')

        with open(self.reference_results_file, 'w') as f:
            json.dump(reference_results, f, indent=2)

        print(f'Saved reference results to: {self.reference_results_file}')
        print(f'Generated {len(reference_results)} reference results')

        return self.reference_results_file

    def load_reference_results(self) -> List[Dict[str, Any]]:
        """
        Load reference results from file.

        Returns:
            List[Dict]: List of reference results
        """

        if not self.reference_results_file or not os.path.exists(self.reference_results_file):
            raise ValueError('Reference results file not found. Generate reference results first.')

        with open(self.reference_results_file, 'r') as f:
            reference_results = json.load(f)

        return reference_results

    def test_single_image_wrapper(self, model_name: str = default_model) -> bool:
        """
        Test that the new single image wrapper produces identical results.

        Args:
            model_name (str): Model name to load

        Returns:
            bool: True if test passes
        """

        print('Testing single image wrapper function...')

        reference_results = self.load_reference_results()
        model = run_detector.load_detector(model_name)

        all_tests_passed = True

        for ref_data in reference_results:
            image_file = ref_data['image_file']
            image_path = ref_data['image_path']
            expected_result = ref_data['result']

            print(f'Testing wrapper on: {image_file}')

            # Load image
            image = vis_utils.load_image(image_path)

            # Test new wrapper function (when implemented)
            # For now, just test current implementation
            actual_result = model.generate_detections_one_image(image, image_id=image_file)

            # Compare results
            test_passed = self.compare_results(expected_result, actual_result, image_file)

            if not test_passed:
                all_tests_passed = False

        if all_tests_passed:
            print('✓ Single image wrapper test PASSED')
        else:
            print('✗ Single image wrapper test FAILED')

        return all_tests_passed

    def test_batch_inference(self, model_name: str = default_model, batch_size: int = 2) -> bool:
        """
        Test batch inference against reference results.

        Args:
            model_name (str): Model name to load
            batch_size (int): Batch size to test

        Returns:
            bool: True if test passes
        """

        print(f'Testing batch inference with batch size {batch_size}...')

        reference_results = self.load_reference_results()
        model = run_detector.load_detector(model_name)

        # Prepare batch inputs
        images = []
        image_ids = []
        expected_results = []

        for ref_data in reference_results:
            image_path = ref_data['image_path']
            image_file = ref_data['image_file']
            expected_result = ref_data['result']

            if os.path.exists(image_path):
                image = vis_utils.load_image(image_path)
                images.append(image)
                image_ids.append(image_file)
                expected_results.append(expected_result)

        if len(images) == 0:
            print('No valid test images found')
            return False

        print(f'Testing batch inference on {len(images)} images...')

        # Test batch inference (when implemented)
        # For now, this will fail - we'll implement this after the batch function is ready
        try:
            if hasattr(model, 'generate_detections_one_batch'):
                batch_results = model.generate_detections_one_batch(images, image_ids)
            else:
                print('generate_detections_one_batch not implemented yet')
                return False
        except Exception as e:
            print(f'Batch inference failed: {e}')
            return False

        # Compare results
        all_tests_passed = True

        if len(batch_results) != len(expected_results):
            print(f'✗ Batch result count mismatch: expected {len(expected_results)}, got {len(batch_results)}')
            return False

        for i, (expected_result, actual_result) in enumerate(zip(expected_results, batch_results)):
            image_file = image_ids[i]
            test_passed = self.compare_results(expected_result, actual_result, image_file)

            if not test_passed:
                all_tests_passed = False

        if all_tests_passed:
            print('✓ Batch inference test PASSED')
        else:
            print('✗ Batch inference test FAILED')

        return all_tests_passed

    def compare_results(self, expected_result: Dict[str, Any], actual_result: Dict[str, Any],
                       image_file: str) -> bool:
        """
        Compare two detection results.

        Args:
            expected_result (dict): Expected detection result
            actual_result (dict): Actual detection result
            image_file (str): Image filename for logging

        Returns:
            bool: True if results match within tolerance
        """

        # Check if both have failures
        expected_failure = expected_result.get('failure')
        actual_failure = actual_result.get('failure')

        if expected_failure is not None or actual_failure is not None:
            if expected_failure == actual_failure:
                print(f'  ✓ {image_file}: Both results have same failure: {expected_failure}')
                return True
            else:
                print(f'  ✗ {image_file}: Failure mismatch - expected: {expected_failure}, actual: {actual_failure}')
                return False

        # Compare file names
        if expected_result.get('file') != actual_result.get('file'):
            print(f'  ✗ {image_file}: File name mismatch')
            return False

        # Compare detection counts
        expected_detections = expected_result.get('detections', [])
        actual_detections = actual_result.get('detections', [])

        confidence_threshold = 0.005
        expected_detections = [d for d in expected_detections if d['conf'] > confidence_threshold]
        actual_detections = [d for d in actual_detections if d['conf'] > confidence_threshold]

        if len(expected_detections) != len(actual_detections):
            print(f'  ✗ {image_file}: Detection count mismatch - expected: {len(expected_detections)}, actual: {len(actual_detections)}')
            return False

        # Use MD test utilities to compare detection lists
        try:
            comparison_result = compare_detection_lists(
                expected_detections,
                actual_detections,
                self.test_options,
                bidirectional_comparison=True
            )

            if comparison_result:
                print(f'  ✓ {image_file}: Detections match within tolerance')
                return True
            else:
                print(f'  ✗ {image_file}: Detections do not match within tolerance')
                return False

        except Exception as e:
            print(f'  ✗ {image_file}: Error comparing detections: {e}')
            return False

    def run_full_test_suite(self, model_name: str = default_model) -> bool:
        """
        Run the complete test suite.

        Args:
            model_name (str): Model name to load

        Returns:
            bool: True if all tests pass
        """

        print('='*60)
        print('BATCH INFERENCE TEST SUITE')
        print('='*60)

        try:
            # Setup
            self.setup_temp_directory()

            # Generate reference results
            self.generate_reference_results(model_name)

            # Test single image wrapper
            wrapper_test_passed = self.test_single_image_wrapper(model_name)

            # Test batch inference
            batch_test_passed = self.test_batch_inference(model_name, batch_size=2)

            # Overall result
            all_tests_passed = wrapper_test_passed and batch_test_passed

            print('='*60)
            if all_tests_passed:
                print('✓ ALL TESTS PASSED')
            else:
                print('✗ SOME TESTS FAILED')
            print('='*60)

            return all_tests_passed

        finally:
            # Cleanup
            self.cleanup_temp_directory()


#%% Test runner functions

def run_basic_test(model=default_model):
    """
    Run a basic test of the current implementation to verify the harness works.
    """

    harness = BatchInferenceTestHarness()

    try:
        harness.setup_temp_directory()
        harness.generate_reference_results(model)
        harness.test_single_image_wrapper(model)

    finally:
        harness.cleanup_temp_directory()


def run_full_test_suite(model):
    """
    Run the full test suite.
    """

    harness = BatchInferenceTestHarness()
    return harness.run_full_test_suite(model)


#%% Command-line interface

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test batch inference implementation')
    parser.add_argument('--basic', action='store_true', help='Run basic test only')
    parser.add_argument('--model', default=default_model, help='Model name to test')
    parser.add_argument('--folder', default=default_image_folder, help='Image folder')

    args = parser.parse_args()
    default_image_folder = args.folder

    if args.basic:
        run_basic_test(model=args.model)
    else:
        success = run_full_test_suite(model=args.model)
        exit(0 if success else 1)
