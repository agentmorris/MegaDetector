#!/usr/bin/env python3

"""

test_checkpoint_at_batch_boundary.py

Specific test to verify that checkpoints are still written when batch processing
is enabled and a batch spans the last image in a checkpoint.

This is an AI-generated test script.

"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# Add the project root to path so we can import modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def test_checkpoint_boundary_bug():
    """
    Test that demonstrates the checkpoint boundary bug and verifies the fix.

    This test uses batch_size=4 and checkpoint_frequency=7 to create a scenario
    where n_images_processed will be 4, 8, 12, 16, 20, 24, 28...
    With the bug, only checkpoints at 28 would be written (28 % 7 == 0).
    With the fix, checkpoints should be written at 8, 16, 24 (crossing 7, 14, 21).
    """

    print("=== Testing Checkpoint Boundary Bug Fix ===")

    test_images_dir = "/mnt/g/temp/md-test-images"
    if not os.path.exists(test_images_dir):
        print(f"Warning: test images directory {test_images_dir} not found")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_output.json")
        checkpoint_file = os.path.join(temp_dir, "test_checkpoint.json")

        # This is the key test case: batch_size=4, checkpoint_frequency=7
        # Expected behavior:
        # - After batch 1: n_images = 4  (no checkpoint, haven't crossed 7)
        # - After batch 2: n_images = 8  (CHECKPOINT, crossed 7)
        # - After batch 3: n_images = 12 (no checkpoint)
        # - After batch 4: n_images = 16 (CHECKPOINT, crossed 14)
        # - etc.
        cmd = [
            "python", "megadetector/detection/run_detector_batch.py",
            "MDV5A",
            test_images_dir,
            output_file,
            "--use_image_queue",
            "--batch_size", "4",
            "--checkpoint_frequency", "7",
            "--checkpoint_path", checkpoint_file,
            "--verbose"  # Use verbose to see checkpoint messages
        ]

        print(f"Running: {' '.join(cmd)}")
        print("Expected: checkpoints at images 8, 16, 24, etc. (not just multiples of 7)")

        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        # Check that verbose output shows checkpoints at the right intervals
        output_lines = result.stdout.split('\n')
        checkpoint_lines = [line for line in output_lines if 'writing checkpoint after' in line]

        print(f"Found {len(checkpoint_lines)} checkpoint messages:")
        for line in checkpoint_lines:
            print(f"  {line.strip()}")

        if len(checkpoint_lines) == 0:
            print("ERROR: No checkpoint messages found in output")
            return False

        # Extract the image counts from checkpoint messages
        import re
        checkpoint_counts = []
        for line in checkpoint_lines:
            match = re.search(r'checkpoint after (\d+) images', line)
            if match:
                checkpoint_counts.append(int(match.group(1)))

        print(f"Checkpoint counts: {checkpoint_counts}")

        # Verify the checkpoints are at the expected boundary crossings
        # With batch_size=4 and checkpoint_frequency=7:
        # Batch results: 4, 8, 12, 16, 20, 24, 28...
        # Checkpoints should be at: 8 (crosses 7), 16 (crosses 14), 24 (crosses 21), 28 (crosses 28)

        expected_checkpoints = []
        batch_count = 1
        while batch_count * 4 <= 31:  # Assuming ~31 images in test set
            images_processed = batch_count * 4
            # Check if this batch crosses a checkpoint boundary
            prev_images = (batch_count - 1) * 4
            prev_checkpoint_threshold = (prev_images // 7) * 7
            current_checkpoint_threshold = (images_processed // 7) * 7

            if current_checkpoint_threshold > prev_checkpoint_threshold:
                expected_checkpoints.append(images_processed)
            batch_count += 1

        print(f"Expected checkpoint counts: {expected_checkpoints}")

        # Check if we got the expected checkpoints
        if checkpoint_counts != expected_checkpoints:
            print(f"ERROR: Checkpoint counts don't match expected values")
            print(f"Expected: {expected_checkpoints}")
            print(f"Got: {checkpoint_counts}")
            return False

        # Verify final output
        if not os.path.exists(output_file):
            print("Output file was not created")
            return False

        with open(output_file, 'r') as f:
            results = json.load(f)

        if 'images' not in results or len(results['images']) == 0:
            print("No images found in results")
            return False

        print(f"Successfully processed {len(results['images'])} images")
        print("✓ Checkpoint boundary bug fix verified!")
        return True


def main():
    """
    Run the boundary bug test.
    """

    print("Testing Checkpoint Boundary Bug Fix")
    print("=" * 40)

    success = test_checkpoint_boundary_bug()

    if success:
        print("\n✓ Test PASSED - Checkpoint boundary bug is fixed!")
        return 0
    else:
        print("\n✗ Test FAILED - Checkpoint boundary bug still exists!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
