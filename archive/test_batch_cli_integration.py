"""

test_batch_cli_integration.py

Test harness to validate that CLI batch inference produces identical results to single-image inference.
Tests the full run_detector_batch.py integration with --batch_size argument.

"""

#%% Imports and constants

import os
import json
import tempfile
import shutil
import subprocess
import sys
from typing import List, Dict, Any, Optional

from megadetector.utils.md_tests import compare_detection_lists, MDTestOptions


#%% Test harness class

class BatchCLIIntegrationTest:
    """
    Test harness to validate batch CLI integration implementation.
    """

    def __init__(self, test_images_dir: str = '/mnt/g/temp/md-test-images'):
        """
        Initialize the test harness.

        Args:
            test_images_dir (str): Directory containing test images
        """

        self.test_images_dir = test_images_dir
        self.reference_results_file = None
        self.temp_dir = None
        self.model_name = 'MDV5A'
        
        # Test options for comparison
        self.test_options = MDTestOptions()
        self.test_options.max_conf_error = 0.001
        self.test_options.max_coord_error = 0.001

        # Path to run_detector_batch.py
        self.detector_script = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'megadetector', 'detection', 'run_detector_batch.py'
        )

    def setup_temp_directory(self):
        """
        Create temporary directory for test files.
        """

        self.temp_dir = tempfile.mkdtemp(prefix='batch_cli_test_')
        print(f'Created temporary directory: {self.temp_dir}')

    def cleanup_temp_directory(self):
        """
        Clean up temporary directory.
        """

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f'Cleaned up temporary directory: {self.temp_dir}')

    def _run_detector_batch(self, batch_size: int = 1, 
                           use_image_queue: bool = False,
                           ncores: int = 1) -> str:
        """
        Run run_detector_batch.py with specified parameters.

        Args:
            batch_size (int): Batch size to use
            use_image_queue (bool): Whether to use image queue
            ncores (int): Number of cores for CPU inference

        Returns:
            str: Path to output JSON file
        """

        output_file = os.path.join(
            self.temp_dir, 
            f'results_batch{batch_size}_queue{use_image_queue}_cores{ncores}.json'
        )

        cmd = [
            sys.executable, self.detector_script,
            self.model_name,
            self.test_images_dir,
            output_file,
            '--recursive',
            '--quiet'
        ]

        # Add batch_size argument (when implemented)
        if batch_size != 1:
            cmd.extend(['--batch_size', str(batch_size)])

        if use_image_queue:
            cmd.append('--use_image_queue')

        if ncores != 1:
            cmd.extend(['--ncores', str(ncores)])

        print(f'Running command: {" ".join(cmd)}')

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f'Command completed successfully')
            if result.stdout.strip():
                print(f'stdout: {result.stdout.strip()}')
            
            return output_file

        except subprocess.CalledProcessError as e:
            print(f'Command failed with return code {e.returncode}')
            print(f'stdout: {e.stdout}')
            print(f'stderr: {e.stderr}')
            raise

    def generate_reference_results(self) -> str:
        """
        Generate reference results using current implementation (batch_size=1).

        Returns:
            str: Path to reference results file
        """

        print(f'Generating reference results with model {self.model_name}...')

        self.reference_results_file = self._run_detector_batch(batch_size=1)

        # Verify reference results were created
        if not os.path.exists(self.reference_results_file):
            raise FileNotFoundError(f'Reference results file not created: {self.reference_results_file}')

        with open(self.reference_results_file, 'r') as f:
            reference_data = json.load(f)

        print(f'Generated reference results: {len(reference_data["images"])} images processed')
        return self.reference_results_file

    def load_results_file(self, results_file: str) -> Dict[str, Any]:
        """
        Load results from JSON file.

        Args:
            results_file (str): Path to results file

        Returns:
            dict: Results data
        """

        if not os.path.exists(results_file):
            raise FileNotFoundError(f'Results file not found: {results_file}')

        with open(results_file, 'r') as f:
            return json.load(f)

    def compare_results_files(self, expected_file: str, actual_file: str, 
                             test_name: str) -> bool:
        """
        Compare two results files for identical detection results.

        Args:
            expected_file (str): Path to expected results file
            actual_file (str): Path to actual results file
            test_name (str): Name of test for logging

        Returns:
            bool: True if results match within tolerance
        """

        print(f'Comparing results for {test_name}...')

        expected_data = self.load_results_file(expected_file)
        actual_data = self.load_results_file(actual_file)

        # Compare image counts
        expected_images = expected_data.get('images', [])
        actual_images = actual_data.get('images', [])

        if len(expected_images) != len(actual_images):
            print(f'  ✗ Image count mismatch: expected {len(expected_images)}, got {len(actual_images)}')
            return False

        # Create lookup dictionaries by file path for comparison
        expected_by_file = {img['file']: img for img in expected_images}
        actual_by_file = {img['file']: img for img in actual_images}

        all_match = True
        match_count = 0
        mismatch_count = 0

        for file_path in expected_by_file:
            if file_path not in actual_by_file:
                print(f'  ✗ Missing result for file: {file_path}')
                all_match = False
                mismatch_count += 1
                continue

            expected_result = expected_by_file[file_path]
            actual_result = actual_by_file[file_path]

            if self._compare_single_result(expected_result, actual_result, file_path):
                match_count += 1
            else:
                all_match = False
                mismatch_count += 1

        print(f'  Results: {match_count} matches, {mismatch_count} mismatches')

        if all_match:
            print(f'  ✓ {test_name} PASSED')
        else:
            print(f'  ✗ {test_name} FAILED')

        return all_match

    def _compare_single_result(self, expected_result: Dict[str, Any], 
                              actual_result: Dict[str, Any], 
                              file_path: str) -> bool:
        """
        Compare two single image results.

        Args:
            expected_result (dict): Expected result
            actual_result (dict): Actual result
            file_path (str): File path for logging

        Returns:
            bool: True if results match
        """

        # Check for failures
        expected_failure = expected_result.get('failure')
        actual_failure = actual_result.get('failure')

        if expected_failure is not None or actual_failure is not None:
            return expected_failure == actual_failure

        # Compare detection counts
        expected_detections = expected_result.get('detections', [])
        actual_detections = actual_result.get('detections', [])

        if len(expected_detections) != len(actual_detections):
            return False

        # Use MD test utilities to compare detection lists
        try:
            return compare_detection_lists(
                expected_detections,
                actual_detections,
                self.test_options,
                bidirectional_comparison=True
            )
        except Exception:
            return False

    def test_batch_sizes(self, batch_sizes: List[int] = [1, 2, 4]) -> bool:
        """
        Test various batch sizes against reference results.

        Args:
            batch_sizes (list): List of batch sizes to test

        Returns:
            bool: True if all tests pass
        """

        print(f'Testing batch sizes: {batch_sizes}')

        all_tests_passed = True

        for batch_size in batch_sizes:
            print(f'\nTesting batch_size={batch_size}...')

            try:
                actual_results_file = self._run_detector_batch(batch_size=batch_size)
                test_passed = self.compare_results_files(
                    self.reference_results_file,
                    actual_results_file,
                    f'batch_size={batch_size}'
                )

                if not test_passed:
                    all_tests_passed = False

            except Exception as e:
                print(f'  ✗ batch_size={batch_size} FAILED with exception: {e}')
                all_tests_passed = False

        return all_tests_passed

    def test_image_queue_integration(self, batch_sizes: List[int] = [1, 2]) -> bool:
        """
        Test batch processing with image queue enabled.

        Args:
            batch_sizes (list): List of batch sizes to test

        Returns:
            bool: True if all tests pass
        """

        print(f'Testing image queue integration with batch sizes: {batch_sizes}')

        all_tests_passed = True

        for batch_size in batch_sizes:
            print(f'\nTesting batch_size={batch_size} with image queue...')

            try:
                actual_results_file = self._run_detector_batch(
                    batch_size=batch_size,
                    use_image_queue=True
                )
                test_passed = self.compare_results_files(
                    self.reference_results_file,
                    actual_results_file,
                    f'batch_size={batch_size}_image_queue'
                )

                if not test_passed:
                    all_tests_passed = False

            except Exception as e:
                print(f'  ✗ batch_size={batch_size} with image queue FAILED: {e}')
                all_tests_passed = False

        return all_tests_passed

    def test_cpu_validation(self) -> bool:
        """
        Test that CPU inference properly handles batch_size > 1.
        Should either error or reduce to batch_size=1.

        Returns:
            bool: True if CPU validation works correctly
        """

        print('Testing CPU batch size validation...')

        # Force CPU inference by setting ncores > 1 (CPU-only parameter)
        try:
            # This should either fail or automatically use batch_size=1
            actual_results_file = self._run_detector_batch(
                batch_size=4,
                ncores=2  # Forces CPU inference
            )

            # If it succeeds, it should produce same results as reference
            test_passed = self.compare_results_files(
                self.reference_results_file,
                actual_results_file,
                'CPU_batch_validation'
            )

            if test_passed:
                print('  ✓ CPU validation PASSED (automatically reduced to batch_size=1)')
                return True
            else:
                print('  ✗ CPU validation FAILED (results differ)')
                return False

        except subprocess.CalledProcessError as e:
            if 'batch' in e.stderr.lower() or 'cpu' in e.stderr.lower():
                print('  ✓ CPU validation PASSED (correctly rejected batch_size > 1)')
                return True
            else:
                print(f'  ✗ CPU validation FAILED with unexpected error: {e.stderr}')
                return False

    def run_full_test_suite(self) -> bool:
        """
        Run the complete CLI integration test suite.

        Returns:
            bool: True if all tests pass
        """

        print('='*60)
        print('BATCH CLI INTEGRATION TEST SUITE')
        print('='*60)

        try:
            # Setup
            self.setup_temp_directory()

            # Generate reference results
            self.generate_reference_results()

            # Test batch sizes
            batch_test_passed = self.test_batch_sizes([1, 2, 4])

            # Test image queue integration
            queue_test_passed = self.test_image_queue_integration([1, 2])

            # Test CPU validation
            cpu_test_passed = self.test_cpu_validation()

            # Overall result
            all_tests_passed = batch_test_passed and queue_test_passed and cpu_test_passed

            print('='*60)
            if all_tests_passed:
                print('✓ ALL CLI INTEGRATION TESTS PASSED')
            else:
                print('✗ SOME CLI INTEGRATION TESTS FAILED')
            print('='*60)

            return all_tests_passed

        finally:
            # Cleanup
            self.cleanup_temp_directory()


#%% Test runner functions

def run_basic_cli_test():
    """
    Run a basic test to verify current CLI works.
    """

    harness = BatchCLIIntegrationTest()

    try:
        harness.setup_temp_directory()
        harness.generate_reference_results()
        print('Basic CLI test completed successfully')

    finally:
        harness.cleanup_temp_directory()


def run_full_cli_test_suite():
    """
    Run the full CLI integration test suite.
    """

    harness = BatchCLIIntegrationTest()
    return harness.run_full_test_suite()


#%% Command-line interface

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test batch CLI integration')
    parser.add_argument('--basic', action='store_true', help='Run basic test only')
    parser.add_argument('--model', default='MDV5A', help='Model name to test')

    args = parser.parse_args()

    if args.basic:
        run_basic_cli_test()
    else:
        success = run_full_cli_test_suite()
        exit(0 if success else 1)