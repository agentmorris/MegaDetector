#!/usr/bin/env python3

"""

test_video_checkpointing.py

Test script to verify checkpointing functionality in process_video.py.

This script tests:
1. Basic checkpointing during normal execution
2. Resume from checkpoint after simulated crash
3. Compatibility with different video processing options
4. Edge cases and error handling

This is an AI-generated test file.

"""

import os
import sys
import json
import subprocess
import tempfile
import time
from pathlib import Path

# Add the project root to path so we can import modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def test_basic_checkpointing():
    """
    Test basic checkpointing functionality without interruption.
    """

    print("=== Testing Basic Video Checkpointing ===")

    # Test parameters
    test_videos_dir = "/mnt/g/temp/md-test-images/video-samples"
    if not os.path.exists(test_videos_dir):
        print(f"Warning: test videos directory {test_videos_dir} not found")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_video_output.json")
        checkpoint_file = os.path.join(temp_dir, "test_video_checkpoint.json")

        # Run with checkpointing every 2 videos
        cmd = [
            "python", "megadetector/detection/process_video.py",
            "MDV5A",
            test_videos_dir,
            "--output_json_file", output_file,
            "--checkpoint_frequency", "2",
            "--checkpoint_path", checkpoint_file,
            "--frame_sample", "50",  # Sample heavily for faster testing
            "--verbose"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        # Verify output file exists and has content
        if not os.path.exists(output_file):
            print("Output file was not created")
            return False

        with open(output_file, 'r') as f:
            results = json.load(f)

        if 'images' not in results or len(results['images']) == 0:
            print("No videos found in results")
            return False

        print(f"Successfully processed {len(results['images'])} videos")

        # Checkpoint file should be deleted after successful completion
        if os.path.exists(checkpoint_file):
            print("Warning: checkpoint file still exists after completion")

        return True


def test_controlled_interruption():
    """
    Test resuming from checkpoint after controlled interruption.
    Uses timeout to simulate a crash.
    """

    print("\n=== Testing Controlled Interruption and Resume ===")

    test_videos_dir = "/mnt/g/temp/md-test-images/video-samples"
    if not os.path.exists(test_videos_dir):
        print(f"Warning: test videos directory {test_videos_dir} not found")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_video_output.json")
        checkpoint_file = os.path.join(temp_dir, "test_video_checkpoint.json")

        # First run: interrupt after a short time
        cmd1 = [
            "python", "megadetector/detection/process_video.py",
            "MDV5A",
            test_videos_dir,
            "--output_json_file", output_file,
            "--checkpoint_frequency", "1",  # Checkpoint after each video
            "--checkpoint_path", checkpoint_file,
            "--frame_sample", "50",
            "--verbose"
        ]

        print(f"Running interrupted job: {' '.join(cmd1)}")

        # Use timeout to kill the process after 20 seconds
        try:
            result1 = subprocess.run(cmd1, cwd=project_root, capture_output=True,
                                   text=True, timeout=20)
            print("Process completed before timeout - checking if checkpoint exists")
        except subprocess.TimeoutExpired:
            print("Process interrupted by timeout (as expected)")

        # Check if checkpoint was created
        if not os.path.exists(checkpoint_file):
            print("No checkpoint file was created during interrupted run")
            return False

        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        if 'images' not in checkpoint_data:
            print("Checkpoint file missing 'images' field")
            return False

        num_checkpointed = len(checkpoint_data['images'])
        print(f"Found {num_checkpointed} videos in checkpoint")

        if num_checkpointed == 0:
            print("No videos were processed before interruption")
            return False

        # Second run: resume from checkpoint
        cmd2 = [
            "python", "megadetector/detection/process_video.py",
            "MDV5A",
            test_videos_dir,
            "--output_json_file", output_file,
            "--checkpoint_frequency", "1",
            "--resume_from_checkpoint", checkpoint_file,
            "--frame_sample", "50",
            "--verbose"
        ]

        print(f"Running resume job: {' '.join(cmd2)}")
        result2 = subprocess.run(cmd2, cwd=project_root, capture_output=True, text=True)

        if result2.returncode != 0:
            print(f"Resume job failed with return code {result2.returncode}")
            print(f"STDOUT: {result2.stdout}")
            print(f"STDERR: {result2.stderr}")
            return False

        # Verify final output
        if not os.path.exists(output_file):
            print("Final output file was not created")
            return False

        with open(output_file, 'r') as f:
            final_results = json.load(f)

        final_count = len(final_results['images'])
        print(f"Final results contain {final_count} videos")

        if final_count < num_checkpointed:
            print("Resume appears to have lost some videos")
            return False

        if final_count == num_checkpointed:
            print("Resume completed with no additional videos to process")
        else:
            print(f"Successfully resumed and processed {final_count - num_checkpointed} additional videos")

        return True


def test_single_video_no_checkpointing():
    """
    Test that single video processing doesn't use checkpointing.
    """

    print("\n=== Testing Single Video (No Checkpointing) ===")

    test_videos_dir = "/mnt/g/temp/md-test-images/video-samples"
    if not os.path.exists(test_videos_dir):
        print(f"Warning: test videos directory {test_videos_dir} not found")
        return False

    # Find first video file
    from megadetector.detection.video_utils import find_videos
    video_files = find_videos(test_videos_dir)
    if len(video_files) == 0:
        print("No video files found in test directory")
        return False

    single_video = video_files[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_single_video_output.json")
        checkpoint_file = os.path.join(temp_dir, "test_single_video_checkpoint.json")

        # Run single video with checkpoint settings (should be ignored)
        cmd = [
            "python", "megadetector/detection/process_video.py",
            "MDV5A",
            single_video,
            "--output_json_file", output_file,
            "--checkpoint_frequency", "1",
            "--checkpoint_path", checkpoint_file,
            "--frame_sample", "50",
            "--verbose"
        ]

        print(f"Running single video test: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Single video test failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        # Verify output
        if not os.path.exists(output_file):
            print("Single video output file was not created")
            return False

        # Checkpoint file should NOT be created for single videos
        if os.path.exists(checkpoint_file):
            print("ERROR: Checkpoint file was created for single video processing")
            return False

        with open(output_file, 'r') as f:
            results = json.load(f)

        if 'images' not in results or len(results['images']) != 1:
            print("Single video results should contain exactly 1 video")
            return False

        print("Single video processing completed correctly (no checkpointing)")
        return True


def test_auto_resume():
    """
    Test auto-resume functionality.
    """

    print("\n=== Testing Auto Resume ===")

    test_videos_dir = "/mnt/g/temp/md-test-images/video-samples"
    if not os.path.exists(test_videos_dir):
        print(f"Warning: test videos directory {test_videos_dir} not found")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_auto_output.json")

        # Create a fake checkpoint file
        checkpoint_file = os.path.join(temp_dir, "video_checkpoint_test_auto_output.json")
        fake_checkpoint = {
            "images": [
                {
                    "file": "fake_video.mp4",
                    "frame_rate": 30.0,
                    "frames_processed": [1, 10],
                    "detections": []
                }
            ]
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(fake_checkpoint, f)

        # Test auto-resume
        cmd = [
            "python", "megadetector/detection/process_video.py",
            "MDV5A",
            test_videos_dir,
            "--output_json_file", output_file,
            "--checkpoint_frequency", "2",
            "--resume_from_checkpoint", "auto",
            "--frame_sample", "50",
            "--verbose"
        ]

        print(f"Running auto-resume test: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Auto-resume test failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        # Check that it found and loaded the checkpoint
        if "Found 1 checkpoint files" not in result.stdout:
            print("Auto-resume didn't find the checkpoint file")
            return False

        print("Auto-resume functionality working correctly")
        return True


def test_with_larger_dataset():
    """
    Test with a larger video dataset for more realistic conditions.
    """

    print("\n=== Testing with Larger Dataset ===")

    larger_videos_dir = "/mnt/g/camera_traps/camera_trap_videos/2023.06.28"
    if not os.path.exists(larger_videos_dir):
        print(f"Large dataset {larger_videos_dir} not found, skipping this test")
        return True  # Not a failure, just skipped

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_large_output.json")
        checkpoint_file = os.path.join(temp_dir, "test_large_checkpoint.json")

        # Test with checkpointing every 5 videos, timeout after 30 seconds
        cmd = [
            "python", "megadetector/detection/process_video.py",
            "MDV5A",
            larger_videos_dir,
            "--output_json_file", output_file,
            "--checkpoint_frequency", "5",
            "--checkpoint_path", checkpoint_file,
            "--frame_sample", "50",
            "--verbose"
        ]

        print(f"Running large dataset test: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=project_root, capture_output=True,
                                  text=True, timeout=60)
            print("Large dataset processing completed")
        except subprocess.TimeoutExpired:
            print("Large dataset processing interrupted by timeout (expected)")

        # Check if checkpoint was created
        if not os.path.exists(checkpoint_file):
            print("No checkpoint file was created during large dataset processing")
            return False

        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        num_videos = len(checkpoint_data['images'])
        print(f"Large dataset checkpoint contains {num_videos} videos")

        if num_videos == 0:
            print("No videos were processed in large dataset test")
            return False

        print("Large dataset checkpointing working correctly")
        return True


def main():
    """
    Run all video checkpointing tests.
    """

    print("Testing Video Processing Checkpointing Implementation")
    print("=" * 60)

    # Check if test videos directory exists
    test_videos_dir = "/mnt/g/temp/md-test-images/video-samples"
    if not os.path.exists(test_videos_dir):
        print(f"Test videos directory {test_videos_dir} not found.")
        print("Some tests will be skipped.")

    print(f"Using test videos from: {test_videos_dir}")

    tests = [
        ("Basic Video Checkpointing", test_basic_checkpointing),
        ("Controlled Interruption", test_controlled_interruption),
        ("Single Video (No Checkpointing)", test_single_video_no_checkpointing),
        ("Auto Resume", test_auto_resume),
        ("Larger Dataset", test_with_larger_dataset)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            results[test_name] = test_func()
            status = "PASSED" if results[test_name] else "FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: ERROR - {str(e)}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
