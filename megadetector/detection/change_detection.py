"""

change_detection.py

This is an experimental module intended to support change detection in cases
where camera backgrounds are stable, and MegaDetector is not suitable.

"""

#%% Imports and constants

import argparse
import random
import sys

import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from megadetector.utils.path_utils import find_images


#%% Support classes

class DetectionMethod(Enum):
    """
    Enum for different motion detection methods.
    """

    FRAME_DIFF = auto()      # Simple frame differencing
    MOG2 = auto()            # Mixture of Gaussians
    KNN = auto()             # K-nearest neighbors
    MOTION_HISTORY = auto()  # Motion history image


class ThresholdType(Enum):
    """
    Enum for different thresholding methods.
    """

    GLOBAL = auto()      # Global thresholding
    ADAPTIVE = auto()    # Adaptive thresholding
    OTSU = auto()        # Otsu's method


@dataclass
class ChangeDetectionOptions:
    """
    Class to store options for change detection
    """

    # Core parameters
    min_area: int = 500
    threshold: int = 25

    # Method selection
    detection_method: DetectionMethod = DetectionMethod.FRAME_DIFF
    threshold_type: ThresholdType = ThresholdType.GLOBAL

    # Pre-processing of the raw images, before differencing
    blur_kernel_size: int = 21

    # Post-processing of the difference image, to fill holes
    dilate_kernel_size: int = 5

    # Background subtractor parameters (MOG2, KNN)
    history: int = 25             # Number of frames to build background model
    var_threshold: float = 16     # Threshold for MOG2
    detect_shadows: bool = False  # Whether to detect shadows (MOG2)

    # Adaptive threshold parameters
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # Motion history parameters
    mhi_duration: float = 10.0    # Duration in frames for motion to persist
    mhi_threshold: int = 30       # Threshold for motion detection
    mhi_buffer_size: int = 20     # Number of frames to keep in buffer

    # Region of interest parameters
    ignore_fraction: Optional[float] = None  # Fraction of image to ignore (-1.0 to 1.0)

    # Processing parameters
    dilate_iterations: int = 2

    # Number of concurrent workers (for parallelizing over folders, not images)
    workers: int = 4

    # Enable additional debug output
    verbose: bool = False

    # Debugging tools
    stop_at_token: str = None

# ...def ChangeDetectionOptions


@dataclass
class MotionHistoryState:
    """
    Class to maintain state for motion history detection across frames
    """

    buffer_size: int = 10
    frame_buffer: list = field(default_factory=list)
    mhi: Optional[np.ndarray] = None  # Motion History Image
    last_timestamp: float = 0.0
    frame_interval: float = 1.0       # Time between frames in "seconds" (nominal)
    frame_shape: Optional[tuple] = None

    def initialize(self, frame):
        """
        Initialize the motion history state with the first frame

        Args:
            frame (np array): First frame to initialize the state
        """

        if self.mhi is None and frame is not None:
            self.frame_shape = frame.shape
            self.mhi = np.zeros(self.frame_shape, dtype=np.float32)


    def update(self, frame, options):
        """
        Update the motion history with a new frame

        Args:
            frame (np array): New frame to update the motion history
            options (ChangeDetectionOptions): detection settings

        Returns:
            Motion mask based on the updated motion history
        """

        self.initialize(frame)

        # Update timestamp
        curr_timestamp = self.last_timestamp + self.frame_interval

        # Update buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)  # Remove oldest frame

        # Check if we have enough frames for motion history
        if len(self.frame_buffer) < 2:
            return np.zeros(frame.shape, dtype=np.uint8)

        # Get the previous frame (most recent in buffer before current)
        prev_frame = self.frame_buffer[-2]

        # Detect motion between frames
        frame_diff = cv2.absdiff(prev_frame, frame)
        _, motion_mask = cv2.threshold(frame_diff, options.mhi_threshold, 1, cv2.THRESH_BINARY)

        # Manual implementation of motion history update (replacing cv2.updateMotionHistory)
        # Decrease the existing MHI values by the time that has passed
        decay_factor = self.frame_interval / options.mhi_duration
        self.mhi = np.maximum(0, self.mhi - decay_factor * 255)

        # Set the MHI to maximum value where motion is detected
        self.mhi[motion_mask > 0] = 255.0

        # Normalize MHI to 0-255 for visualization and further processing
        normalized_mhi = np.uint8(self.mhi)

        self.last_timestamp = curr_timestamp

        return normalized_mhi

# ...def MotionHistoryState


#%% Functions

def create_background_subtractor(options=None):
    """
    Create a background subtractor

    Args:
        options (ChangeDetectionOptions, optional): detection settings

    Returns:
        Background subtractor object
    """

    if options is None:
        options = ChangeDetectionOptions()

    if options.detection_method == DetectionMethod.MOG2:
        return cv2.createBackgroundSubtractorMOG2(
            history=options.history,
            varThreshold=options.var_threshold,
            detectShadows=options.detect_shadows
        )

    elif options.detection_method == DetectionMethod.KNN:
        return cv2.createBackgroundSubtractorKNN(
            history=options.history,
            dist2Threshold=options.var_threshold,
            detectShadows=options.detect_shadows
        )

    return None

# ...def create_background_subtractor(...)


def detect_motion(prev_image_path,
                  curr_image_path,
                  options=None,
                  motion_state=None,
                  bg_subtractor=None):
    """
    Detect motion between two consecutive images.

    Args:
        prev_image_path (str): path to the previous image
        curr_image_path (str): path to the current image
        options (ChangeDetectionOptions, optional): detection settings
        motion_state (MotionHistoryState, optional): state for motion history
        bg_subtractor (cv2 background subtractor object): background subtractor model for
            MOG2/KNN methods

    Returns:
        tuple: (motion_result, updated_motion_state)
            motion_result: dict with keys:
                motion_detected: bool indicating whether motion was detected
                motion_regions: list of bounding boxes of motion regions
                diff_percentage: percentage of the image that changed
                debug_images: dict of intermediate images for debugging (if requested)
    """

    # Helpful debug line for plotting images in IPython
    # im = mask; cv2.imshow('im',im); cv2.waitKey(0); cv2.destroyAllWindows()

    ##%% Argument handling

    if options is None:
        options = ChangeDetectionOptions()

    to_return = {
        'motion_detected': False,
        'motion_regions': [],
        'diff_percentage': 0.0,
        'debug_images': {}
    }


    ##%% Image reading

    # Read images
    curr_image = cv2.imread(str(curr_image_path))

    if curr_image is None:
        print(f"Could not read image: {curr_image_path}")
        return to_return, motion_state

    # Read previous image if available (used for frame diff mode)
    prev_image = None
    if prev_image_path is not None:
        prev_image = cv2.imread(str(prev_image_path))
        if prev_image is None:
            print(f"Could not read image: {prev_image_path}")
            return to_return, motion_state


    ##%% Preprocessing

    # Apply region of interest masking if specified
    roi_mask = None
    if options.ignore_fraction is not None:
        h, w = curr_image.shape[0], curr_image.shape[1]
        roi_mask = np.ones((h, w), dtype=np.uint8)

        # Calculate the number of rows to ignore
        ignore_rows = int(abs(options.ignore_fraction) * h)

        # Negative fraction: ignore top portion
        if options.ignore_fraction < 0:
            roi_mask[0:ignore_rows, :] = 0
        # Positive fraction: ignore bottom portion
        elif options.ignore_fraction > 0:
            roi_mask[h-ignore_rows:h, :] = 0

    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    # Apply ROI mask if specified
    if roi_mask is not None:
        curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=roi_mask)

    # Apply Gaussian blur to reduce noise
    curr_gray = cv2.GaussianBlur(curr_gray, (options.blur_kernel_size, options.blur_kernel_size), 0)


    ##%% Differencing

    # Simple frame differencing
    if options.detection_method == DetectionMethod.FRAME_DIFF:

        # Need previous image for frame differencing
        if prev_image is None:
            return to_return, motion_state

        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        # Apply ROI mask if specified
        if roi_mask is not None:
            prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=roi_mask)

        prev_gray = cv2.GaussianBlur(prev_gray, (options.blur_kernel_size, options.blur_kernel_size), 0)

        # Compute absolute difference between frames
        mask = cv2.absdiff(prev_gray, curr_gray)

    # Background subtractors (MOG2 or KNN)
    elif options.detection_method in [DetectionMethod.MOG2, DetectionMethod.KNN]:

        # Use the provided background subtractor
        if bg_subtractor is None:
            print("Warning: No background subtractor provided, creating a new one")
            bg_subtractor = create_background_subtractor(options)

        # Get foreground mask from current image
        mask = bg_subtractor.apply(curr_gray)

        # Apply ROI mask again after background subtraction if needed
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # Motion history image
    elif options.detection_method == DetectionMethod.MOTION_HISTORY:

        # Initialize motion state if not provided
        if motion_state is None:
            motion_state = MotionHistoryState(buffer_size=options.mhi_buffer_size)
            motion_state.frame_interval = 0.1  # Default interval between frames

        # Apply ROI mask if needed (motion state will handle the masked image)
        if roi_mask is not None:
            masked_curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=roi_mask)
            mask = motion_state.update(masked_curr_gray, options)
        else:
            mask = motion_state.update(curr_gray, options)

    # Fall back to frame differencing
    else:
        if prev_image is None:
            return to_return, motion_state

        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        # Apply ROI mask if specified
        if roi_mask is not None:
            prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=roi_mask)

        prev_gray = cv2.GaussianBlur(prev_gray, (options.blur_kernel_size, options.blur_kernel_size), 0)

        mask = cv2.absdiff(prev_gray, curr_gray)


    ##%% Debugging

    if options.stop_at_token is not None and options.stop_at_token in curr_image_path:
        import IPython; IPython.embed()


    ##%% Thresholding the mask

    # Adaptive thresholding
    if options.threshold_type == ThresholdType.ADAPTIVE:
        thresh = cv2.adaptiveThreshold(
            mask, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            options.adaptive_block_size,
            options.adaptive_c
        )

    # Otsu
    elif options.threshold_type == ThresholdType.OTSU:
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Global thresholding
    else:
        assert options.threshold_type == ThresholdType.GLOBAL
        _, thresh = cv2.threshold(mask, options.threshold, 255, cv2.THRESH_BINARY)


    ##%% Postprocessing the thresholded mask

    # Ensure ROI mask is applied after thresholding
    if roi_mask is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    # Dilate the threshold image to fill in holes
    kernel = np.ones((options.dilate_kernel_size, options.dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=options.dilate_iterations)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    significant_contours = [c for c in contours if cv2.contourArea(c) > options.min_area]

    # Calculate changed percentage (only consider the ROI area)
    if roi_mask is not None:
        roi_area = np.sum(roi_mask > 0)
        diff_percentage = (np.sum(thresh > 0) / roi_area) * 100 if roi_area > 0 else 0
    else:
        diff_percentage = (np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])) * 100

    # Get bounding boxes for significant motion regions
    motion_regions = [cv2.boundingRect(c) for c in significant_contours]

    # Populate return values
    to_return['motion_detected'] = len(significant_contours) > 0
    to_return['motion_regions'] = motion_regions
    to_return['diff_percentage'] = diff_percentage

    # Add debug images if verbose
    if options.verbose:

        to_return['debug_images'] = {
            'curr_gray': curr_gray,
            'mask': mask,
            'thresh': thresh,
            'dilated': dilated
        }

        if prev_image is not None:
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (options.blur_kernel_size, options.blur_kernel_size), 0)
            to_return['debug_images']['prev_gray'] = prev_gray

    return to_return, motion_state

# ...def detect_motion(...)


def process_camera_folder(folder_path, options=None):
    """
    Process all images in a camera folder to detect motion.

    Args:
        folder_path (str): path to the folder containing images from one camera
        options (ChangeDetectionOptions, optional): detection settings

    Returns:
        DataFrame with motion detection results for all images in the folder
    """

    if options is None:
        options = ChangeDetectionOptions()

    folder_path = Path(folder_path)
    camera_name = folder_path.name

    # Find images
    image_files = find_images(folder_path, recursive=True, return_relative_paths=False)

    if len(image_files) == 0:
        print(f'"No images found in {folder_path}"')
        return pd.DataFrame()

    print(f"Processing {len(image_files)} images in {camera_name}")

    # Initialize results
    results = []

    # Initialize motion state and background subtractor for this camera
    motion_state = None
    bg_subtractor = None

    if options.detection_method == DetectionMethod.MOTION_HISTORY:
        motion_state = MotionHistoryState(buffer_size=options.mhi_buffer_size)

    if options.detection_method in [DetectionMethod.MOG2, DetectionMethod.KNN]:
        bg_subtractor = create_background_subtractor(options)

    # Add first image with no motion (no previous image to compare)
    if len(image_files) > 0:
        first_image = image_files[0]
        results.append({
            'camera': camera_name,
            'image': str(first_image),
            'prev_image': None,
            'motion_detected': False,
            'diff_percentage': 0.0,
            'num_regions': 0,
            'regions': ''
        })

        # If using background subtractor, initialize it with the first image
        if bg_subtractor is not None and options.detection_method in \
            [DetectionMethod.MOG2, DetectionMethod.KNN]:
            first_img = cv2.imread(str(first_image))
            if first_img is not None:
                first_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
                first_gray = cv2.GaussianBlur(first_gray,
                                              (options.blur_kernel_size,
                                               options.blur_kernel_size), 0)
                bg_subtractor.apply(first_gray, learningRate=1.0)  # Initialize with this frame

    # Process pairs of consecutive images
    for i_image in tqdm(range(1, len(image_files)), total=len(image_files),
                        disable=(not options.verbose)):
        prev_image = image_files[i_image-1]
        curr_image = image_files[i_image]

        motion_result, motion_state = detect_motion(
            prev_image, curr_image, options, motion_state, bg_subtractor
        )

        motion_detected = motion_result['motion_detected']
        regions = motion_result['motion_regions']
        diff_percentage = motion_result['diff_percentage']

        # Format regions as semicolon-separated list of "x,y,w,h"
        regions_str = ';'.join([f"{x},{y},{w},{h}" for x, y, w, h in regions])

        # Add result for current image
        results.append({
            'camera': camera_name,
            'image': str(curr_image),
            'prev_image': str(prev_image),
            'motion_detected': motion_detected,
            'diff_percentage': diff_percentage,
            'num_regions': len(regions),
            'regions': regions_str
        })

    # ...for each image

    return pd.DataFrame(results)

# ...def process_camera_folder(...)


def process_folders(folders, options=None, output_csv=None):
    """
    Process multiple folders of images.

    Args:
        folders (list): list of folder paths to process
        options (ChangeDetectionOptions, optional): detection settings
        output_csv (str, optional): path to save results as CSV

    Returns:
        DataFrame with motion detection results for all folders
    """

    if options is None:
        options = ChangeDetectionOptions()

    # Convert folders to list if it's a single string
    if isinstance(folders, str):
        folders = [folders]

    # Convert to Path objects
    folders = [Path(folder) for folder in folders]

    all_results = []

    if options.workers == 1:
        for folder in folders:
            folder_results = process_camera_folder(folder, options)
            all_results.append(folder_results)
    else:
        # Process each camera folder in parallel
        with ProcessPoolExecutor(max_workers=options.workers) as executor:
            future_to_folder = {executor.submit(process_camera_folder, folder, options): folder
                            for folder in folders}

            for future in future_to_folder:
                folder = future_to_folder[future]
                try:
                    folder_results = future.result()
                    all_results.append(folder_results)
                    print(f"Finished processing {folder}")
                except Exception as e:
                    print(f"Error processing {folder}: {e}")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save to CSV if requested
        if output_csv:
            combined_results.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")

        return combined_results
    else:
        return pd.DataFrame()

# ...def process_folders(...)


def create_change_previews(motion_results, output_folder, num_samples=10, random_seed=None):
    """
    Create side-by-side previews of images with detected motion

    Args:
        motion_results (DataFrame): DataFrame with motion detection results
        output_folder (str): folder where preview images will be saved
        num_samples (int, optional): number of random samples to create
        random_seed (int, optional): seed for random sampling (for reproducibility)

    Returns:
        List of paths to created preview images
    """

    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Filter results to only include rows with motion detected
    motion_detected = motion_results[motion_results['motion_detected'] == True] # noqa

    if len(motion_detected) == 0:
        print("No motion detected in any images")
        return []


    if random_seed is not None:
        random.seed(random_seed)

    if num_samples is None:
        samples = motion_detected
    else:
        # Sample rows (or take all if fewer than requested)
        sample_size = min(num_samples, len(motion_detected))
        sample_indices = random.sample(range(len(motion_detected)), sample_size)
        samples = motion_detected.iloc[sample_indices]

    preview_paths = []

    for i_sample, row in samples.iterrows():

        curr_image_path = row['image']
        prev_image_path = row['prev_image']

        # Read images
        curr_image = cv2.imread(curr_image_path)
        prev_image = cv2.imread(prev_image_path)

        if curr_image is None or prev_image is None:
            print(f"Could not read images: {prev_image_path} or {curr_image_path}")
            continue

        # Ensure that both images have the same dimensions
        if curr_image.shape != prev_image.shape:
            # Resize to match dimensions
            prev_image = cv2.resize(prev_image, (curr_image.shape[1], curr_image.shape[0]))

        # Create side-by-side comparison
        combined = np.hstack((prev_image, curr_image))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'Before', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'After', (curr_image.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)

        # Add details at the bottom
        camera = row['camera']
        diff_pct = row['diff_percentage']
        info_text = f"Camera: {camera} | Change: {diff_pct:.2f}%"
        cv2.putText(combined, info_text, (10, combined.shape[0] - 10), font, 0.7, (0, 255, 0), 2)

        # Draw bounding boxes on the 'after' image if regions exist
        if row['regions']:
            regions = row['regions'].split(';')
            for region in regions:
                if region:
                    try:
                        x, y, w, h = map(int, region.split(','))
                        cv2.rectangle(combined,
                                    (curr_image.shape[1] + x, y),
                                    (curr_image.shape[1] + x + w, y + h),
                                    (0, 0, 255), 2)
                    except ValueError:
                        print(f"Invalid region format: {region}")

        # Save the combined image
        camera_name = Path(curr_image_path).parent.name
        image_name = Path(curr_image_path).name
        output_path = output_folder / f"preview_{camera_name}_{image_name}"
        cv2.imwrite(str(output_path), combined)

        preview_paths.append(str(output_path))

    # ...for each image
    return preview_paths

# ...def create_change_previews(...)


#%% Command-line driver

def main(): # noqa
    parser = argparse.ArgumentParser(description='Detect motion in timelapse camera images')
    parser.add_argument('--root_dir', required=True, help='Root directory containing camera folders')
    parser.add_argument('--output_csv', default=None, help='Optional output CSV file')

    # Core parameters
    parser.add_argument('--min_area', type=int, default=500,
                        help='Minimum contour area to consider as significant motion')
    parser.add_argument('--threshold', type=int, default=25,
                        help='Threshold for binary image creation')

    # Method selection
    parser.add_argument('--detection_method', type=str, default='frame_diff',
                        choices=['frame_diff', 'mog2', 'knn', 'motion_history'],
                        help='Method to use for change detection')
    parser.add_argument('--threshold_type', type=str, default='global',
                        choices=['global', 'adaptive', 'otsu'],
                        help='Type of thresholding to apply')

    # Background subtractor parameters
    parser.add_argument('--history', type=int, default=500,
                        help='Number of frames used to build the background model')
    parser.add_argument('--var_threshold', type=float, default=16,
                        help='Threshold for MOG2/KNN background subtraction')
    parser.add_argument('--detect_shadows', action='store_true',
                        help='Detect and mark shadows in background subtraction')

    # Adaptive threshold parameters
    parser.add_argument('--adaptive_block_size', type=int, default=11,
                        help='Block size for adaptive thresholding')
    parser.add_argument('--adaptive_c', type=int, default=2,
                        help='Constant subtracted from the mean for adaptive thresholding')

    # Motion history parameters
    parser.add_argument('--mhi_duration', type=float, default=1.0,
                        help='Duration in seconds for the motion history image')
    parser.add_argument('--mhi_threshold', type=int, default=30,
                        help='Threshold for motion detection in the motion history image')
    parser.add_argument('--mhi_buffer_size', type=int, default=10,
                        help='Number of frames to keep in motion history buffer')

    # Region of interest parameters
    parser.add_argument('--ignore_fraction', type=float, default=None,
                        help='Fraction of image to ignore: negative = top, positive = bottom, range [-1.0, 1.0]')

    # Processing parameters
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable additional debug output')

    # Preview generation
    parser.add_argument('--create_previews', action='store_true',
                        help='Create side-by-side previews of detected motion')
    parser.add_argument('--preview_folder', default='change_previews',
                        help='Folder to save preview images')
    parser.add_argument('--num_previews', type=int, default=10,
                        help='Number of random preview images to create')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # Validate ignore_fraction
    if args.ignore_fraction is not None and (args.ignore_fraction < -1.0 or args.ignore_fraction > 1.0):
        print("Error: ignore_fraction must be between -1.0 and 1.0")
        return

    # Create options object
    options = ChangeDetectionOptions(
        min_area=args.min_area,
        threshold=args.threshold,
        detection_method=getattr(DetectionMethod, args.detection_method.upper()),
        threshold_type=getattr(ThresholdType, args.threshold_type.upper()),
        history=args.history,
        var_threshold=args.var_threshold,
        detect_shadows=args.detect_shadows,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        mhi_duration=args.mhi_duration,
        mhi_threshold=args.mhi_threshold,
        mhi_buffer_size=args.mhi_buffer_size,
        ignore_fraction=args.ignore_fraction,
        workers=args.workers,
        verbose=args.verbose
    )

    # Get camera folders
    root_dir = Path(args.root_dir)
    camera_folders = [f for f in root_dir.iterdir() if f.is_dir()]

    print(f"Found {len(camera_folders)} camera folders")

    # Process all folders
    results = process_folders(
        camera_folders,
        options=options,
        output_csv=args.output_csv
    )

    # Create previews if requested
    if args.create_previews:
        preview_paths = create_change_previews(
            results,
            args.preview_folder,
            num_samples=args.num_previews
        )
        print(f"Created {len(preview_paths)} preview images in {args.preview_folder}")

    print("Motion detection completed")

    # Display summary
    motion_detected_count = results['motion_detected'].sum()
    total_images = len(results)
    print(f"Motion detected in {motion_detected_count} out of {total_images} images "
          f"({motion_detected_count/total_images*100:.2f}%)")


if __name__ == "__main__":
    main()
