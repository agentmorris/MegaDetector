"""

create_crop_folder.py

Given a MegaDetector .json file and a folder of images, creates a new folder
of images representing all above-threshold crops from the original folder.

"""

#%% Constants and imports

import os
import json
import argparse

from tqdm import tqdm

from multiprocessing.pool import Pool, ThreadPool
from collections import defaultdict
from functools import partial

from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import is_list_sorted
from megadetector.visualization.visualization_utils import crop_image
from megadetector.visualization.visualization_utils import exif_preserving_save


#%% Support classes

class CreateCropFolderOptions:
    """
    Options used to parameterize create_crop_folder().
    """

    def __init__(self):

        #: Confidence threshold determining which detections get written
        self.confidence_threshold = 0.1

        #: Number of pixels to expand each crop
        self.expansion = 0

        #: JPEG quality to use for saving crops (None for default)
        self.quality = 95

        #: Whether to overwrite existing images
        self.overwrite = True

        #: Number of concurrent workers
        self.n_workers = 8

        #: Whether to use processes ('process') or threads ('thread') for parallelization
        self.pool_type = 'thread'

        #: Include only these categories, or None to include all
        #:
        #: options.category_names_to_include = ['animal']
        self.category_names_to_include = None


#%% Support functions

def _get_crop_filename(image_fn,crop_id):
    """
    Generate crop filenames in a consistent way.
    """

    if isinstance(crop_id,int):
        crop_id = str(crop_id).zfill(3)
    assert isinstance(crop_id,str)
    return insert_before_extension(image_fn,'crop_' + crop_id)


def _generate_crops_for_single_image(crops_this_image,
                                     input_folder,
                                     output_folder,
                                     options):
    """
    Generate all the crops required for a single image.

    Args:
        crops_this_image (list of dict): list of dicts with at least keys
            'image_fn_relative', 'crop_id'
        input_folder (str): input folder (whole images)
        output_folder (crops): output folder (crops)
        options (CreateCropFolderOptions): cropping options
    """

    if len(crops_this_image) == 0:
        return

    image_fn_relative = crops_this_image[0]['image_fn_relative']
    input_fn_abs = os.path.join(input_folder,image_fn_relative)
    assert os.path.isfile(input_fn_abs)

    detections_to_crop = [c['detection'] for c in crops_this_image]

    cropped_images = crop_image(detections_to_crop,
                                input_fn_abs,
                                confidence_threshold=0,
                                expansion=options.expansion)

    assert len(cropped_images) == len(crops_this_image)

    # i_crop = 0; crop_info = crops_this_image[0]
    for i_crop,crop_info in enumerate(crops_this_image):

        assert crop_info['image_fn_relative'] == image_fn_relative
        crop_filename_relative = _get_crop_filename(image_fn_relative, crop_info['crop_id'])
        crop_filename_abs = os.path.join(output_folder,crop_filename_relative).replace('\\','/')

        if os.path.isfile(crop_filename_abs) and not options.overwrite:
            continue

        cropped_image = cropped_images[i_crop]
        os.makedirs(os.path.dirname(crop_filename_abs),exist_ok=True)
        exif_preserving_save(cropped_image,crop_filename_abs,quality=options.quality)

    # ...for each crop


#%% Main function

def crop_results_to_image_results(image_results_file_with_crop_ids,
                                  crop_results_file,
                                  output_file,
                                  delete_crop_information=True,
                                  require_identical_detection_categories=True,
                                  restrict_to_top_n=-1,
                                  crop_results_prefix=None,
                                  detections_without_classification_handling='error'):
    """
    This function is intended to be run after you have:

        1. Run MegaDetector on a folder
        2. Generated a crop folder using create_crop_folder
        3. Run a species classifier on those crops

    This function will take the crop-level results and transform them back
    to the original images.  Classification categories, if available, are taken
    from [crop_results_file].

    Args:
        image_results_file_with_crop_ids (str): results file for the original images,
            containing crop IDs, likely generated via create_crop_folder.  All
            non-standard fields in this file will be passed along to [output_file].
        crop_results_file (str): results file for the crop folder
        output_file (str): output .json file, containing crop-level classifications
            mapped back to the image level.
        delete_crop_information (bool, optional): whether to delete the "crop_id" and
            "crop_filename_relative" fields from each detection, if present.
        require_identical_detection_categories (bool, optional): if True, error if
            the image-level and crop-level detection categories are different.  If False,
            ignore the crop-level detection categories.
        restrict_to_top_n (int, optional): If >0, removes all but the top N classification
            results for each detection.
        crop_results_prefix (str, optional): if not None, removes this prefix from crop
            results filenames.  Intended to support the case where the crop results
            use absolute paths.
        detections_without_classification_handling (str, optional): what to do when we
            encounter a crop that doesn't appear in classification results: 'error',
            or 'include' ("include" means "leave the detection alone, without classifications"
    """

    ##%% Validate inputs

    assert os.path.isfile(image_results_file_with_crop_ids), \
        'Could not find image-level input file {}'.format(image_results_file_with_crop_ids)
    assert os.path.isfile(crop_results_file), \
        'Could not find crop results file {}'.format(crop_results_file)
    output_dir = os.path.dirname(output_file)
    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)


    ##%% Read input files

    print('Reading input...')

    with open(image_results_file_with_crop_ids,'r') as f:
        image_results_with_crop_ids = json.load(f)
    with open(crop_results_file,'r') as f:
        crop_results = json.load(f)

    # Find all the detection categories that need to be consistent
    used_detection_category_ids = set()
    for im in tqdm(image_results_with_crop_ids['images']):
        if 'detections' not in im or im['detections'] is None:
            continue
        for det in im['detections']:
            if 'crop_id' in det:
                used_detection_category_ids.add(det['category'])

    # Make sure the detection categories that matter are consistent across the two files
    if require_identical_detection_categories:
        for category_id in used_detection_category_ids:
            category_name = image_results_with_crop_ids['detection_categories'][category_id]
            assert category_id in crop_results['detection_categories'] and \
                category_name == crop_results['detection_categories'][category_id], \
                    'Crop results and detection results use incompatible categories'

    crop_filename_to_results = {}

    # im = crop_results['images'][0]
    for im in crop_results['images']:
        fn = im['file']
        # Possibly remove a prefix from each filename
        if (crop_results_prefix is not None) and (crop_results_prefix in fn):
            if fn.startswith(crop_results_prefix):
                fn = fn.replace(crop_results_prefix,'',1)
                im['file'] = fn
        crop_filename_to_results[fn] = im

    if 'classification_categories' in crop_results:
        image_results_with_crop_ids['classification_categories'] = \
            crop_results['classification_categories']

    if 'classification_category_descriptions' in crop_results:
        image_results_with_crop_ids['classification_category_descriptions'] = \
            crop_results['classification_category_descriptions']


    ##%% Read classifications from crop results, merge into image-level results

    print('Reading classification results...')

    n_skipped_detections = 0

    # Loop over the original image-level detections
    #
    # im = image_results_with_crop_ids['images'][0]
    for i_image,im in tqdm(enumerate(image_results_with_crop_ids['images']),
                           total=len(image_results_with_crop_ids['images'])):

        if 'detections' not in im or im['detections'] is None:
            continue

        # i_det = 0; det = im['detections'][i_det]
        for det in im['detections']:

            if 'classifications' in det:
                del det['classifications']

            if 'crop_id' in det:

                # We may be skipping detections with no classification results
                skip_detection = False

                # Find the corresponding crop in the classification results
                crop_filename_relative = det['crop_filename_relative']
                if crop_filename_relative not in crop_filename_to_results:
                    if detections_without_classification_handling == 'error':
                        raise ValueError('Crop lookup error: {}'.format(crop_filename_relative))
                    elif detections_without_classification_handling == 'include':
                        # Leave this detection unclassified
                        skip_detection = True
                    else:
                        raise ValueError(
                            'Illegal value for detections_without_classification_handling: {}'.format(
                                detections_without_classification_handling
                        ))

                if skip_detection:

                    n_skipped_detections += 1

                else:

                    crop_results_this_detection = crop_filename_to_results[crop_filename_relative]

                    # Consistency checking
                    assert crop_results_this_detection['file'] == crop_filename_relative, \
                        'Crop filename mismatch'
                    assert len(crop_results_this_detection['detections']) == 1, \
                        'Multiple crop results for a single detection'
                    assert crop_results_this_detection['detections'][0]['bbox'] == [0,0,1,1], \
                        'Invalid crop bounding box'

                    # This check was helpful for the case where crop-level results had already
                    # taken detection confidence values from detector output by construct, but this isn't
                    # really meaningful for most cases.
                    # assert abs(crop_results_this_detection['detections'][0]['conf'] - det['conf']) < 0.01

                    if require_identical_detection_categories:
                        assert crop_results_this_detection['detections'][0]['category'] == det['category']

                    # Copy the crop-level classifications
                    det['classifications'] = crop_results_this_detection['detections'][0]['classifications']
                    confidence_values = [x[1] for x in det['classifications']]
                    assert is_list_sorted(confidence_values,reverse=True)
                    if restrict_to_top_n > 0:
                        det['classifications'] = det['classifications'][0:restrict_to_top_n]

            if delete_crop_information:
                if 'crop_id' in det:
                    del det['crop_id']
                if 'crop_filename_relative' in det:
                    del det['crop_filename_relative']

        # ...for each detection

    # ...for each image

    if n_skipped_detections > 0:
        print('Skipped {} detections'.format(n_skipped_detections))


    ##%% Write output file

    print('Writing output file...')

    with open(output_file,'w') as f:
        json.dump(image_results_with_crop_ids,f,indent=1)

# ...def crop_results_to_image_results(...)


def create_crop_folder(input_file,
                       input_folder,
                       output_folder,
                       output_file=None,
                       crops_output_file=None,
                       options=None):
    """
    Given a MegaDetector .json file and a folder of images, creates a new folder
    of images representing all above-threshold crops from the original folder.

    Optionally writes a new .json file that attaches unique IDs to each detection.

    Args:
        input_file (str): MD-formatted .json file to process
        input_folder (str): Input image folder
        output_folder (str): Output (cropped) image folder
        output_file (str, optional): new .json file that attaches unique IDs to each detection.
        crops_output_file (str, optional): new .json file that includes whole-image detections
            for each of the crops, using confidence values from the original results
        options (CreateCropFolderOptions, optional): crop parameters
    """

    ## Validate options, prepare output folders

    if options is None:
        options = CreateCropFolderOptions()

    assert os.path.isfile(input_file), 'Input file {} not found'.format(input_file)
    assert os.path.isdir(input_folder), 'Input folder {} not found'.format(input_folder)
    os.makedirs(output_folder,exist_ok=True)

    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if len(output_dir) > 0:
            os.makedirs(output_dir,exist_ok=True)


    ##%% Read input

    print('Reading MD results file...')
    with open(input_file,'r') as f:
        detection_results = json.load(f)

    category_ids_to_include = None

    if options.category_names_to_include is not None:
        category_id_to_name = detection_results['detection_categories']
        category_name_to_id = invert_dictionary(category_id_to_name)
        category_ids_to_include = set()
        for category_name in options.category_names_to_include:
            assert category_name in category_name_to_id, \
                'Unrecognized category name {}'.format(category_name)
            category_ids_to_include.add(category_name_to_id[category_name])

    ##%% Make a list of crops that we need to create

    # Maps input images to list of dicts, with keys 'crop_id','detection'
    image_fn_relative_to_crops = defaultdict(list)
    n_crops = 0

    n_detections_excluded_by_category = 0

    # im = detection_results['images'][0]
    for i_image,im in enumerate(detection_results['images']):

        if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
            continue

        detections_this_image = im['detections']

        image_fn_relative = im['file']

        for i_detection,det in enumerate(detections_this_image):

            if det['conf'] < options.confidence_threshold:
                continue

            if (category_ids_to_include is not None) and \
                (det['category'] not in category_ids_to_include):
                n_detections_excluded_by_category += 1
                continue

            det['crop_id'] = i_detection

            crop_info = {'image_fn_relative':image_fn_relative,
                         'crop_id':i_detection,
                         'detection':det}

            crop_filename_relative = _get_crop_filename(image_fn_relative,
                                                        crop_info['crop_id'])
            det['crop_filename_relative'] = crop_filename_relative

            image_fn_relative_to_crops[image_fn_relative].append(crop_info)
            n_crops += 1

    # ...for each input image

    print('Prepared a list of {} crops from {} of {} input images'.format(
        n_crops,len(image_fn_relative_to_crops),len(detection_results['images'])))

    if n_detections_excluded_by_category > 0:
        print('Excluded {} detections by category'.format(n_detections_excluded_by_category))

    ##%% Generate crops

    if options.n_workers <= 1:

        # image_fn_relative = next(iter(image_fn_relative_to_crops))
        for image_fn_relative in tqdm(image_fn_relative_to_crops.keys()):
            crops_this_image = image_fn_relative_to_crops[image_fn_relative]
            _generate_crops_for_single_image(crops_this_image=crops_this_image,
                                             input_folder=input_folder,
                                             output_folder=output_folder,
                                             options=options)

    else:

        print('Creating a {} pool with {} workers'.format(options.pool_type,options.n_workers))
        pool = None
        try:
            if options.pool_type == 'thread':
                pool = ThreadPool(options.n_workers)
            else:
                assert options.pool_type == 'process'
                pool = Pool(options.n_workers)

            # Each element in this list is the list of crops for a single image
            crop_lists = list(image_fn_relative_to_crops.values())

            with tqdm(total=len(image_fn_relative_to_crops)) as pbar:
                for i,_ in enumerate(pool.imap_unordered(partial(
                            _generate_crops_for_single_image,
                                input_folder=input_folder,
                                output_folder=output_folder,
                                options=options),
                            crop_lists)):
                    pbar.update()
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                print("Pool closed and joined for crop folder creation")

    # ...if we're using parallel processing


    ##%% Write output file

    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(detection_results,f,indent=1)

    if crops_output_file is not None:

        original_images = detection_results['images']

        detection_results_cropped = detection_results
        detection_results_cropped['images'] = []

        # im = original_images[0]
        for im in original_images:

            if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
                continue

            detections_this_image = im['detections']
            image_fn_relative = im['file']

            for i_detection,det in enumerate(detections_this_image):

                if 'crop_id' in det:
                    im_out = {}
                    im_out['file'] = det['crop_filename_relative']
                    det_out = {}
                    det_out['category'] = det['category']
                    det_out['conf'] = det['conf']
                    det_out['bbox'] = [0, 0, 1, 1]
                    im_out['detections'] = [det_out]
                    detection_results_cropped['images'].append(im_out)

                # ...if we need to include this crop in the new .json file

            # ...for each crop

        # ...for each original image

        with open(crops_output_file,'w') as f:
            json.dump(detection_results_cropped,f,indent=1)

# ...def create_crop_folder()


#%% Command-line driver

def main():
    """
    Command-line interface for creating a crop folder from MegaDetector results.
    """

    parser = argparse.ArgumentParser(
        description='Create a folder of crops from MegaDetector results'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the MegaDetector .json results file'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to the folder containing the original images'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to the folder where cropped images will be saved'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save the modified MegaDetector .json file (with crop IDs and filenames)'
    )
    parser.add_argument(
        '--crops_output_file',
        type=str,
        default=None,
        help='Path to save a new .json file for the crops themselves (with full-image detections for each crop)'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.1,
        help='Confidence threshold for detections to be cropped (default: 0.1)'
    )
    parser.add_argument(
        '--expansion',
        type=int,
        default=0,
        help='Number of pixels to expand each crop (default: 0)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality for saving crops (default: 95)'
    )
    parser.add_argument(
        '--overwrite',
        type=str,
        default='true',
        choices=['true', 'false'],
        help="Overwrite existing crop images (default: 'true')"
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=8,
        help='Number of concurrent workers (default: 8)'
    )
    parser.add_argument(
        '--pool_type',
        type=str,
        default='thread',
        choices=['thread', 'process'],
        help="Type of parallelism to use ('thread' or 'process', default: 'thread')"
    )
    parser.add_argument(
        '--category_names',
        type=str,
        default=None,
        help="Comma-separated list of category names to include " + \
             "(e.g., 'animal,person'). If None (default), all categories are included."
    )

    args = parser.parse_args()

    options = CreateCropFolderOptions()
    options.confidence_threshold = args.confidence_threshold
    options.expansion = args.expansion
    options.quality = args.quality
    options.overwrite = (args.overwrite.lower() == 'true')
    options.n_workers = args.n_workers
    options.pool_type = args.pool_type

    if args.category_names:
        options.category_names_to_include = [name.strip() for name in args.category_names.split(',')]
    else:
        options.category_names_to_include = None

    print('Starting crop folder creation...')
    print('Input MD results: {}'.format(args.input_file))
    print('Input image folder: {}'.format(args.input_folder))
    print('Output crop folder: {}'.format(args.output_folder))

    if args.output_file:
        print('Modified MD results will be saved to {}'.format(args.output_file))
    if args.crops_output_file:
        print('Crops .json output will be saved to {}'.format(args.crops_output_file))

    create_crop_folder(
        input_file=args.input_file,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        output_file=args.output_file,
        crops_output_file=args.crops_output_file,
        options=options
    )

if __name__ == '__main__':
    main()
