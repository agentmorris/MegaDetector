"""

change_detection.py

This is an experimental module intended to support change detection in cases
where camera backgrounds are stable, and MegaDetector is not suitable.

"""

#%% Imports and constants

import argparse
import random

import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from megadetector.utils.path_utils import find_images

default_min_area = 500
default_threshold = 25
blur_kernel_size = 21
dilate_kernel_size = 5
default_workers = 4


#%% Functions

def detect_motion(prev_image_path, 
                  curr_image_path, 
                  min_area=default_min_area, 
                  threshold=default_threshold):
    """
    Detect motion between two consecutive images.
    
    Args:
        prev_image_path (str): path to the previous image
        curr_image_path (str): path to the current image
        min_area (float, optional): minimum contour area to be considered significant motion
        threshold (float, optional): threshold for binary image creation after absolute difference
        
    Returns:
        dict, with keys:
            motion_detected: bool indicating whether motion was detected
            motion_regions: list of bounding boxes of motion regions
            diff_percentage: percentage of the image that changed
    """

    # Helpful debug line for plotting images in IPython
    # im = x; cv2.imshow('im',im); cv2.waitKey(0); cv2.destroyAllWindows()

    to_return = {'motion_detected':None,'motion_regions':None,'diff_percentage':None}

    # Read images
    prev_image = cv2.imread(str(prev_image_path))
    curr_image = cv2.imread(str(curr_image_path))
    
    if prev_image is None or curr_image is None:
        print(f"Could not read images: {prev_image_path} or {curr_image_path}")
        return to_return
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (blur_kernel_size, blur_kernel_size), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # Compute absolute difference between frames
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate the threshold image to fill in holes
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Calculate changed percentage
    diff_percentage = (np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])) * 100
    
    # Get bounding boxes for significant motion regions
    motion_regions = [cv2.boundingRect(c) for c in significant_contours]
    
    # {'motion_detected':None,'motion_regions':None,'diff_percentage':None}
    to_return['motion_detected'] = len(significant_contours) > 0
    to_return['motion_regions'] = motion_regions
    to_return['diff_percentage'] = diff_percentage

    return to_return

# ...def detect_motion(...)


def process_camera_folder(folder_path, 
                          min_area=default_min_area, 
                          threshold=default_threshold,
                          verbose=False):
    """
    Process all images in a camera folder to detect motion.
    
    Args:
        folder_path (str): path to the folder containing images from one camera
        min_area (float, optional): minimum contour area to consider as motion
        threshold (float, optional): threshold for binary image creation
        verbose (bool, optional): enable additional debug output
    
    Returns:
        DataFrame with motion detection results for all images in the folder
    """

    folder_path = Path(folder_path)
    camera_name = folder_path.name
    
    # Find images
    image_files = find_images(folder_path,recursive=True,return_relative_paths=False)
    
    if len(image_files) == 0:
        print(f'"No images found in {folder_path}"')
        return pd.DataFrame()
    
    print(f"Processing {len(image_files)} images in {camera_name}")
    
    # Initialize results
    results = []
    
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
    
    # Process pairs of consecutive images
    for i_image in tqdm(range(1, len(image_files)),total=len(image_files),disable=(not verbose)):

        prev_image = image_files[i_image-1]
        curr_image = image_files[i_image]
        
        motion_result = detect_motion(prev_image, curr_image, min_area, threshold)

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
    
    # ...for each image (pair)

    return pd.DataFrame(results)

# ...def process_camera_folder(...)


def process_folders(folders, 
                    min_area=default_min_area,
                    threshold=default_threshold, 
                    output_csv=None, 
                    workers=default_workers,
                    verbose=False):
    """
    Process multiple folders of timelapse images.
    
    Args:
        folders (list): list of folder paths to process
        min_area (float, optional): minimum contour area to consider as motion
        threshold (float, optional): threshold for binary image creation
        output_csv (str, optional): optional path to save results as CSV
        workers (int, optional): number of parallel workers
        verbose (bool, optional): enable additional debug output

    Returns:
        DataFrame with motion detection results for all folders
    """

    # Convert folders to list if it's a single string
    if isinstance(folders, str):
        folders = [folders]
    
    # Convert to Path objects
    folders = [Path(folder) for folder in folders]
    
    all_results = []
    
    if workers == 1:
        for folder in folders:
            folder_results = process_camera_folder(folder,
                                                min_area=min_area,
                                                threshold=threshold,
                                                verbose=verbose)
            all_results.append(folder_results)
    else:
        # Process each camera folder in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_folder = {executor.submit(process_camera_folder, folder, min_area, threshold, verbose): folder 
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
    Create side-by-side previews of images with detected motion.
    
    Args:
        motion_results: DataFrame with motion detection results
        output_folder: Folder where preview images will be saved
        num_samples: Number of random samples to create
        random_seed: Seed for random sampling (for reproducibility)
        
    Returns:
        List of paths to created preview images
    """

    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Filter results to only include rows with motion detected
    motion_detected = motion_results[motion_results['motion_detected'] == True]
    
    if len(motion_detected) == 0:
        print("No motion detected in any images")
        return []
    
    # Sample rows (or take all if fewer than requested)
    sample_size = min(num_samples, len(motion_detected))
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Take a random sample
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
    
    # ...for each sampled image

    return preview_paths

# ...def create_change_previews(...)


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description='Detect motion in timelapse camera images')
    parser.add_argument('--root_dir', required=True, help='Root directory containing camera folders')
    parser.add_argument('--output_csv', default=None, help='Optional output CSV file')
    parser.add_argument('--min_area', type=int, default=500, 
                        help='Minimum contour area to consider as significant motion')
    parser.add_argument('--threshold', type=int, default=25, 
                        help='Threshold for binary image creation')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel workers')
    parser.add_argument('--create_previews', action='store_true',
                        help='Create side-by-side previews of detected motion')
    parser.add_argument('--preview_folder', default='change_previews',
                        help='Folder to save preview images')
    parser.add_argument('--num_previews', type=int, default=10,
                        help='Number of random preview images to create')
    args = parser.parse_args()
    
    # Get camera folders
    root_dir = Path(args.root_dir)
    camera_folders = [f for f in root_dir.iterdir() if f.is_dir()]
    
    print(f"Found {len(camera_folders)} camera folders")
    
    # Process all folders
    results = process_folders(
        camera_folders,
        min_area=args.min_area,
        threshold=args.threshold,
        output_csv=args.output_csv,
        workers=args.workers
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
    print(f"Motion detected in {motion_detected_count} out of {total_images} images ({motion_detected_count/total_images*100:.2f}%)")

if __name__ == "__main__":
    main()
