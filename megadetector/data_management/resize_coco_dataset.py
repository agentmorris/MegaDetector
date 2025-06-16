"""

resize_coco_dataset.py

Given a COCO-formatted dataset, resizes all the images to a target size,
scaling bounding boxes accordingly.

"""

#%% Imports and constants

import os
import json
import shutil
import argparse
import sys
import tempfile
from PIL import Image
import pytest
from copy import deepcopy

from tqdm import tqdm
from collections import defaultdict
from multiprocessing.pool import Pool, ThreadPool
from functools import partial

from megadetector.utils.path_utils import insert_before_extension
from megadetector.visualization.visualization_utils import \
    open_image, resize_image, exif_preserving_save
from megadetector.utils.ct_utils import make_test_folder


#%% Functions

def _process_single_image_for_resize(image_data,
                                     input_folder,
                                     output_folder,
                                     target_size,
                                     correct_size_image_handling,
                                     image_id_to_annotations_map):
    """
    Processes a single image: loads, resizes/copies, updates metadata, and scales annotations.
    """
    current_image_data = image_data.copy()
    annotations_for_this_image = image_id_to_annotations_map.get(image_data['id'], [])
    # Deepcopy annotations if they are dicts, otherwise simple copy for basic types
    current_annotations = [ann.copy() if isinstance(ann, dict) else ann for ann in annotations_for_this_image]

    input_fn_relative = current_image_data['file_name']
    input_fn_abs = os.path.join(input_folder, input_fn_relative)
    # The main function already asserts this, but good for a helper to be robust
    if not os.path.isfile(input_fn_abs):
        print(f"Warning: Image file {input_fn_abs} not found, skipping.")
        # Return original data if image not found, or handle as error
        return image_data, annotations_for_this_image


    output_fn_abs = os.path.join(output_folder, input_fn_relative)
    os.makedirs(os.path.dirname(output_fn_abs), exist_ok=True)

    pil_im = open_image(input_fn_abs)
    input_w = pil_im.width
    input_h = pil_im.height

    image_is_already_target_size = \
        (input_w == target_size[0]) and (input_h == target_size[1])
    preserve_original_size = \
        (target_size[0] == -1) and (target_size[1] == -1)

    if image_is_already_target_size or preserve_original_size:
        output_w = input_w
        output_h = input_h
        if correct_size_image_handling == 'copy':
            if input_fn_abs != output_fn_abs: # only copy if src and dst are different
                 shutil.copyfile(input_fn_abs, output_fn_abs)
        elif correct_size_image_handling == 'rewrite':
            exif_preserving_save(pil_im, output_fn_abs)
        else:
            raise ValueError(f'Unrecognized value {correct_size_image_handling} for correct_size_image_handling')
    else:
        pil_im = resize_image(pil_im, target_size[0], target_size[1])
        output_w = pil_im.width
        output_h = pil_im.height
        exif_preserving_save(pil_im, output_fn_abs)

    current_image_data['width'] = output_w
    current_image_data['height'] = output_h

    for ann in current_annotations:
        if 'bbox' in ann:
            bbox = ann['bbox']
            if (output_w != input_w) or (output_h != input_h):
                width_scale = output_w / input_w
                height_scale = output_h / input_h
                bbox = [
                    bbox[0] * width_scale,
                    bbox[1] * height_scale,
                    bbox[2] * width_scale,
                    bbox[3] * height_scale
                ]
            ann['bbox'] = bbox

    return current_image_data, current_annotations


def resize_coco_dataset(input_folder,
                        input_filename,
                        output_folder,
                        output_filename,
                        target_size=(-1,-1),
                        correct_size_image_handling='copy',
                        n_workers: int = 1,
                        pool_type: str = 'thread'):
    """
    Given a COCO-formatted dataset (images in input_folder, data in input_filename), resizes
    all the images to a target size (in output_folder) and scales bounding boxes accordingly.

    Args:
        input_folder (str): the folder where images live; filenames in [input_filename] should
            be relative to [input_folder]
        input_filename (str): the (input) COCO-formatted .json file containing annotations
        output_folder (str): the folder to which we should write resized images; can be the
            same as [input_folder], in which case images are over-written
        output_filename (str): the COCO-formatted .json file we should generate that refers to
            the resized images
        target_size (list or tuple of ints, optional): this should be tuple/list of ints, with length 2 (w,h).
            If either dimension is -1, aspect ratio will be preserved.  If both dimensions are -1, this means
            "keep the original size".  If  both dimensions are -1 and correct_size_image_handling is copy, this
            function is basically a no-op.
        correct_size_image_handling (str, optional): what to do in the case where the original size
            already matches the target size.  Can be 'copy' (in which case the original image is just copied
            to the output folder) or 'rewrite' (in which case the image is opened via PIL and re-written,
            attempting to preserve the same quality).  The only reason to do use 'rewrite' 'is the case where
            you're superstitious about biases coming from images in a training set being written by different
            image encoders.
        n_workers (int, optional): number of workers to use for parallel processing.
            Defaults to 1 (no parallelization). If <= 1, processing is sequential.
        pool_type (str, optional): type of multiprocessing pool to use ('thread' or 'process').
            Defaults to 'thread'. Only used if n_workers > 1.

    Returns:
        dict: the COCO database with resized images, identical to the content of [output_filename]
    """

    # Read input data
    with open(input_filename,'r') as f:
        d = json.load(f)

    # Map image IDs to annotations
    image_id_to_annotations = defaultdict(list)
    for ann in d['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)

    # TODO: this is trivially parallelizable
    #
    # Process images (sequentially or in parallel)
    # We will replace this loop with pool.map or a sequential call
    # to _process_single_image_for_resize later.

    original_images = d['images']
    processed_results = []

    if n_workers <= 1:
        # Sequential processing
        print("Processing images sequentially...")
        for im_data in tqdm(original_images, desc="Resizing images sequentially"):
            result = _process_single_image_for_resize(
                im_data,
                input_folder,
                output_folder,
                target_size,
                correct_size_image_handling,
                image_id_to_annotations
            )
            processed_results.append(result)
    else:
        # Parallel processing
        assert pool_type in ('process', 'thread'), f'Illegal pool type {pool_type}'
        SelectedPool = ThreadPool if pool_type == 'thread' else Pool

        print(f'Starting a {pool_type} pool of {n_workers} workers for image resizing')
        pool = SelectedPool(n_workers)

        p_process_image = partial(_process_single_image_for_resize,
                                  input_folder=input_folder,
                                  output_folder=output_folder,
                                  target_size=target_size,
                                  correct_size_image_handling=correct_size_image_handling,
                                  image_id_to_annotations_map=image_id_to_annotations)

        processed_results = list(tqdm(pool.imap(p_process_image, original_images), total=len(original_images), desc=f"Resizing images with {pool_type} pool"))

        pool.close()
        pool.join()
        print(f"{pool_type.capitalize()} pool closed and joined.")

    # Reconstruct the 'images' and 'annotations' lists from processed_results
    # This part will be handled in the next step as per the plan.
    # For now, ensure processed_results is populated correctly.
    # The following lines from the previous version need to be removed from here
    # and added after this if/else block, operating on 'processed_results'.

    new_images_list = []
    new_annotations_list = []
    for res_im_data, res_annotations in processed_results:
        new_images_list.append(res_im_data)
        new_annotations_list.extend(res_annotations)

    d['images'] = new_images_list
    d['annotations'] = new_annotations_list

    # Write output file
    with open(output_filename,'w') as f:
        json.dump(d,f,indent=1)

    return d

# ...def resize_coco_dataset(...)


#%% Interactive driver

if False:

    pass

    #%% Test resizing

    input_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
    input_filename = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training.json')
    target_size = (1600,-1)

    output_filename = insert_before_extension(input_filename,'resized-test')
    output_folder = input_folder + '-resized-test'

    correct_size_image_handling = 'rewrite'

    resize_coco_dataset(input_folder,input_filename,
                        output_folder,output_filename,
                        target_size=target_size,
                        correct_size_image_handling=correct_size_image_handling)


    #%% Preview

    from megadetector.visualization import visualize_db
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000

    html_file,_ = visualize_db.visualize_db(output_filename,
                                              os.path.expanduser('~/tmp/resize_coco_preview'),
                                              output_folder,options)


    from megadetector.utils import path_utils # noqa
    path_utils.open_file(html_file)


#%% Command-line driver

def main():
    """
    Command-line execution function.
    """
    parser = argparse.ArgumentParser(
        description='Resize images in a COCO dataset and scale annotations.'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to the folder containing original images.'
    )
    parser.add_argument(
        'input_filename',
        type=str,
        help='Path to the input COCO .json file.'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to the folder where resized images will be saved.'
    )
    parser.add_argument(
        'output_filename',
        type=str,
        help='Path to the output COCO .json file for resized data.'
    )
    parser.add_argument(
        '--target_size',
        type=str,
        default='-1,-1',
        help='Target size as "width,height". Use -1 to preserve aspect ratio for a dimension. E.g., "800,600" or "1024,-1".'
    )
    parser.add_argument(
        '--correct_size_image_handling',
        type=str,
        default='copy',
        choices=['copy', 'rewrite'],
        help='How to handle images already at target size.'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=1,
        help='Number of workers for parallel processing. <=1 for sequential.'
    )
    parser.add_argument(
        '--pool_type',
        type=str,
        default='thread',
        choices=['thread', 'process'],
        help='Type of multiprocessing pool if n_workers > 1.'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    try:
        target_size_parts = args.target_size.split(',')
        if len(target_size_parts) != 2:
            raise ValueError("target_size must have two comma-separated parts (width,height).")
        parsed_target_size = (int(target_size_parts[0]), int(target_size_parts[1]))
    except ValueError as e:
        print(f"Error parsing target_size: {e}")
        parser.print_help()
        parser.exit()

    resize_coco_dataset(
        args.input_folder,
        args.input_filename,
        args.output_folder,
        args.output_filename,
        target_size=parsed_target_size,
        correct_size_image_handling=args.correct_size_image_handling,
        n_workers=args.n_workers,
        pool_type=args.pool_type
    )
    print("Dataset resizing complete.")

if __name__ == '__main__':
    main()


#%% Tests

class TestResizeCocoDataset:
    def set_up(self):
        # Create a unique base directory for this test run
        # make_test_folder itself creates a unique subfolder if not given a specific name,
        # but here we give it a specific subfolder name for clarity.
        # Using a timestamp or UUID could also make self.test_dir unique per run if needed,
        # but make_test_folder handles uniqueness if its default behavior is used.
        # For simplicity here, we'll assume make_test_folder('resize_coco_tests') is sufficient
        # or that ct_utils.make_test_folder handles creating a unique enough base.
        # To be absolutely sure for multiple test classes/runs, one might add a unique suffix.
        # However, following the example of url_utils.py, it seems a fixed subfolder name is okay
        # if tests are run in a way that teardown always occurs.
        self.test_dir = make_test_folder(subfolder='resize_coco_tests')

        self.input_images_dir_seq = os.path.join(self.test_dir, 'input_images_seq')
        os.makedirs(self.input_images_dir_seq, exist_ok=True)

        self.input_images_dir_par = os.path.join(self.test_dir, 'input_images_par')
        os.makedirs(self.input_images_dir_par, exist_ok=True)

        self.output_images_dir_seq = os.path.join(self.test_dir, 'output_images_seq')
        os.makedirs(self.output_images_dir_seq, exist_ok=True)

        self.output_images_dir_par = os.path.join(self.test_dir, 'output_images_par')
        os.makedirs(self.output_images_dir_par, exist_ok=True)

    def tear_down(self):
        # Ensure shutil is imported if not already globally in the file
        # (it is, under '#%% Imports and constants')
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_dummy_image_and_coco_json(self,
                                          image_dir,
                                          json_filename_base="input_coco.json",
                                          num_images=2,
                                          original_size=(100, 100),
                                          num_annotations_per_image=2):
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "test_category"}]
        }

        annotation_id_counter = 1

        for i in range(num_images):
            image_name = f"image_{i}.png"
            image_path = os.path.join(image_dir, image_name)

            # Create a dummy image
            try:
                img = Image.new('RGB', original_size, color='red')
                img.save(image_path)
            except Exception as e:
                # In some environments, font loading for default PIL text might fail.
                # For a simple color image, this shouldn't be an issue.
                # If it is, consider a simpler save or pre-creating a tiny PNG.
                print(f"Warning: Could not create dummy image {image_path}: {e}")
                # Fallback: create an empty file, though this will fail later steps
                # open(image_path, 'a').close()

            image_entry = {
                "id": i + 1,
                "file_name": image_name, # Filename only, not path
                "width": original_size[0],
                "height": original_size[1]
            }
            coco_data["images"].append(image_entry)

            for j in range(num_annotations_per_image):
                annotation_entry = {
                    "id": annotation_id_counter,
                    "image_id": image_entry["id"],
                    "category_id": 1, # Corresponds to "test_category"
                    # Simple, non-overlapping bbox for testing scaling
                    "bbox": [10 + j*30, 10 + j*5, 20, 15]
                }
                coco_data["annotations"].append(annotation_entry)
                annotation_id_counter += 1

        json_file_path = os.path.join(self.test_dir, json_filename_base)
        with open(json_file_path, 'w') as f:
            json.dump(coco_data, f, indent=1)

        return json_file_path, coco_data

    def test_resize_sequential_vs_parallel(self):
        self.set_up()

        try:
            num_images_to_test = 3
            original_w, original_h = 120, 80
            target_w, target_h = 60, 40
            target_size_test = (target_w, target_h)

            # Sequential run
            input_json_path_seq, _ = self._create_dummy_image_and_coco_json(
                image_dir=self.input_images_dir_seq,
                json_filename_base="input_coco_seq.json",
                num_images=num_images_to_test,
                original_size=(original_w, original_h)
            )
            output_json_path_seq = os.path.join(self.test_dir, 'output_coco_seq.json')

            print(f"Test: Starting sequential resize (1 worker)...")
            resize_coco_dataset(
                input_folder=self.input_images_dir_seq,
                input_filename=input_json_path_seq,
                output_folder=self.output_images_dir_seq,
                output_filename=output_json_path_seq,
                target_size=target_size_test,
                n_workers=1
            )
            print(f"Test: Sequential resize complete. Output: {output_json_path_seq}")

            # Parallel run
            # For the parallel run, we use different input/output directories but can reuse the same logic
            # for creating the dummy dataset structure. The image files will be new.
            input_json_path_par, _ = self._create_dummy_image_and_coco_json(
                image_dir=self.input_images_dir_par,
                json_filename_base="input_coco_par.json",
                num_images=num_images_to_test,
                original_size=(original_w, original_h)
            )
            output_json_path_par = os.path.join(self.test_dir, 'output_coco_par.json')

            print(f"Test: Starting parallel resize (2 workers, thread pool)...")
            resize_coco_dataset(
                input_folder=self.input_images_dir_par,
                input_filename=input_json_path_par,
                output_folder=self.output_images_dir_par,
                output_filename=output_json_path_par,
                target_size=target_size_test,
                n_workers=2, # Using 2 workers for testing parallelism
                pool_type='thread'
            )
            print(f"Test: Parallel resize complete. Output: {output_json_path_par}")

            # Load results
            with open(output_json_path_seq, 'r') as f:
                data_seq = json.load(f)
            with open(output_json_path_par, 'r') as f:
                data_par = json.load(f)

            # Compare COCO JSON data
            # Compare images
            assert len(data_seq['images']) == num_images_to_test
            assert len(data_seq['images']) == len(data_par['images']), "Number of images differs"

            sorted_images_seq = sorted(data_seq['images'], key=lambda x: x['id'])
            sorted_images_par = sorted(data_par['images'], key=lambda x: x['id'])

            for img_s, img_p in zip(sorted_images_seq, sorted_images_par):
                assert img_s['id'] == img_p['id'], f"Image IDs differ: {img_s['id']} vs {img_p['id']}"
                # Filenames are generated independently, so we only check structure, not exact name matching across seq/par runs' inputs
                # but output structure should be consistent if input names were e.g. image_0, image_1
                assert img_s['file_name'] == img_p['file_name']
                assert img_s['width'] == target_w, f"Seq image {img_s['id']} width incorrect"
                assert img_s['height'] == target_h, f"Seq image {img_s['id']} height incorrect"
                assert img_p['width'] == target_w, f"Par image {img_p['id']} width incorrect"
                assert img_p['height'] == target_h, f"Par image {img_p['id']} height incorrect"

            # Compare annotations
            assert len(data_seq['annotations']) == len(data_par['annotations']), "Number of annotations differs"
            # Assuming _create_dummy_image_and_coco_json creates the same number of annotations for each test run

            sorted_anns_seq = sorted(data_seq['annotations'], key=lambda x: x['id'])
            sorted_anns_par = sorted(data_par['annotations'], key=lambda x: x['id'])

            for ann_s, ann_p in zip(sorted_anns_seq, sorted_anns_par):
                assert ann_s['id'] == ann_p['id'], f"Annotation IDs differ: {ann_s['id']} vs {ann_p['id']}"
                assert ann_s['image_id'] == ann_p['image_id'], f"Annotation image_ids differ for ann_id {ann_s['id']}"
                assert ann_s['category_id'] == ann_p['category_id'], f"Annotation category_ids differ for ann_id {ann_s['id']}"

                # Check bbox scaling (example: original width 120, target 60 -> scale 0.5)
                # Original bbox: [10, 10, 20, 15] -> Scaled: [5, 5, 10, 7.5] (Floats possible)
                # Need to compare with tolerance or ensure rounding is handled if expecting ints
                # For this test, let's assume direct comparison works due to simple scaling.
                # If PIL's resize causes slight pixel shifts affecting precise sub-pixel bbox calculations,
                # then a tolerance (pytest.approx) would be better.
                # Given the current resize_coco_dataset logic, it's direct multiplication.
                for i in range(4):
                    assert abs(ann_s['bbox'][i] - ann_p['bbox'][i]) < 1e-5, \
                        f"Bbox element {i} differs for ann_id {ann_s['id']}: {ann_s['bbox']} vs {ann_p['bbox']}"

            # Compare actual image files
            seq_files = sorted(os.listdir(self.output_images_dir_seq))
            par_files = sorted(os.listdir(self.output_images_dir_par))

            assert len(seq_files) == num_images_to_test, "Incorrect number of output images (sequential)"
            assert len(seq_files) == len(par_files), "Number of output image files differs"

            for fname_s, fname_p in zip(seq_files, par_files):
                assert fname_s == fname_p, "Output image filenames differ between seq and par runs"
                img_s_path = os.path.join(self.output_images_dir_seq, fname_s)
                img_p_path = os.path.join(self.output_images_dir_par, fname_p)

                with Image.open(img_s_path) as img_s_pil:
                    assert img_s_pil.size == target_size_test, f"Image {fname_s} (seq) has wrong dimensions: {img_s_pil.size}"
                with Image.open(img_p_path) as img_p_pil:
                    assert img_p_pil.size == target_size_test, f"Image {fname_p} (par) has wrong dimensions: {img_p_pil.size}"

            print("Test test_resize_sequential_vs_parallel PASSED")

        finally:
            self.tear_down()

# End of TestResizeCocoDataset class definition

def test_resize_coco_dataset_main():
    print("Starting TestResizeCocoDataset main runner...")
    test_runner = TestResizeCocoDataset()
    test_runner.test_resize_sequential_vs_parallel()
    # If there were more test methods, they would be called here:
    # test_runner.test_another_feature()
    print("TestResizeCocoDataset main runner finished.")

if __name__ == '__main__':
    # This allows running the test directly from the command line.
    # For more comprehensive testing, pytest should be used if available in the project.
    print("Executing test_resize_coco_dataset_main() due to __main__ guard...")
    # This will only run if the script is executed directly, *not* when imported.
    # However, the main CLI driver above this Test cell also uses an __name__ == '__main__'
    # guard. If this script is run directly, Python executes the *first* __main__ block
    # it encounters top-to-bottom. So, the main CLI driver will run, not these tests,
    # if the script is invoked as `python resize_coco_dataset.py`.
    # To run these tests directly, one would typically comment out the primary __main__ block
    # or use a test runner like pytest that discovers test classes and methods.
    # For the purpose of this task, we are adding the block as requested, assuming
    # it might be used in specific debugging scenarios or if pytest is configured
    # to find and run such main test functions.
    # A better way for pytest is to just let pytest discover the Test... class.
    # A common pattern for a simple script-based runner is to have a specific CLI arg for tests.

    # Given the file structure, if __name__ == '__main__' is used for the CLI driver
    # at the end of the file (but before this test cell), that block will take precedence.
    # For this test runner to be effective via `python script.py`, the main CLI's
    # `if __name__ == '__main__':` block would need to be conditional or removed.
    # Let's assume for now this is for explicit programmatic invocation or pytest discovery.

    # If the primary CLI driver is intended to be the default __main__ action,
    # then this test __main__ block is more for documentation or specific invocation.
    # We'll proceed with adding it as requested by the plan.
    test_resize_coco_dataset_main()

