"""

visualize_detector_output.py

Render images with bounding boxes annotated on them to a folder, based on a
detector output result file (.json), optionally writing an HTML index file.

"""

#%% Imports

import argparse
import os
import random
import sys

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm

from megadetector.data_management.annotations.annotation_constants import detector_bbox_category_id_to_name
from megadetector.detection.run_detector import get_typical_confidence_threshold_from_results
from megadetector.utils.ct_utils import get_max_conf
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.utils import write_html_image_list
from megadetector.utils.path_utils import path_is_abs
from megadetector.utils.path_utils import open_file
from megadetector.utils.wi_taxonomy_utils import load_md_or_speciesnet_file
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.visualization.visualization_utils import \
    blur_detections, DEFAULT_BOX_THICKNESS, DEFAULT_LABEL_FONT_SIZE

default_box_sort_order = 'confidence'


#%% Constants

# This will only be used if a category mapping is not available in the results file.
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}


#%% Support functions

def _render_image(entry,
                  detector_label_map,
                  classification_label_map,
                  confidence_threshold,
                  classification_confidence_threshold,
                  render_detections_only,
                  preserve_path_structure,
                  out_dir,
                  images_dir,
                  output_image_width,
                  box_sort_order=default_box_sort_order,
                  category_names_to_blur=None,
                  box_thickness=DEFAULT_BOX_THICKNESS,
                  box_expansion=0,
                  label_font_size=DEFAULT_LABEL_FONT_SIZE,
                  rounded_corners=False,
                  label_font='arial.ttf'):
    """
    Internal function for rendering a single image.
    """

    rendering_result = {'failed_image':False,
                        'missing_image':False,
                        'skipped_image':False,
                        'annotated_image_path':None,
                        'max_conf':None,
                        'image_filename_in_abs':None,
                        'file':entry['file']}

    image_id = entry['file']

    if 'failure' in entry and entry['failure'] is not None:
        rendering_result['failed_image'] = True
        return rendering_result

    assert 'detections' in entry and entry['detections'] is not None

    max_conf = get_max_conf(entry)
    rendering_result['max_conf'] = max_conf

    if (max_conf < confidence_threshold) and render_detections_only:
        rendering_result['skipped_image'] = True
        return rendering_result

    if images_dir is None:
        image_filename_in_abs = image_id
        assert path_is_abs(image_filename_in_abs), \
            'Absolute paths are required when no image base dir is supplied'
    else:
        assert not path_is_abs(image_id), \
            'Relative paths are required when an image base dir is supplied'
        image_filename_in_abs = os.path.join(images_dir, image_id)
    if not os.path.exists(image_filename_in_abs):
        print(f'Image {image_id} not found')
        rendering_result['missing_image'] = True
        return rendering_result

    rendering_result['image_filename_in_abs'] = image_filename_in_abs

    # Load the image
    image = vis_utils.open_image(image_filename_in_abs)

    # Find categories we're supposed to blur
    category_ids_to_blur = []
    if category_names_to_blur is not None:
        if isinstance(category_names_to_blur,str):
            category_names_to_blur = [category_names_to_blur]
        for category_id in detector_label_map:
            if detector_label_map[category_id] in category_names_to_blur:
                category_ids_to_blur.append(category_id)

    detections_to_blur = []
    for d in entry['detections']:
        if d['conf'] >= confidence_threshold and d['category'] in category_ids_to_blur:
            detections_to_blur.append(d)
    if len(detections_to_blur) > 0:
        blur_detections(image,detections_to_blur)

    # Resize if necessary
    #
    # If output_image_width is -1 or None, this will just return the original image
    image = vis_utils.resize_image(image, output_image_width)

    vis_utils.render_detection_bounding_boxes(
        entry['detections'], image,
        label_map=detector_label_map,
        classification_label_map=classification_label_map,
        confidence_threshold=confidence_threshold,
        classification_confidence_threshold=classification_confidence_threshold,
        box_sort_order=box_sort_order,
        thickness=box_thickness,
        expansion=box_expansion,
        label_font_size=label_font_size,
        rounded_corners=rounded_corners,
        label_font=label_font)

    if not preserve_path_structure:
        for char in ['/', '\\', ':']:
            image_id = image_id.replace(char, '~')
        annotated_img_path = os.path.join(out_dir, f'anno_{image_id}')
    else:
        assert not os.path.isabs(image_id), "Can't preserve paths when operating on absolute paths"
        annotated_img_path = os.path.join(out_dir, image_id)
        os.makedirs(os.path.dirname(annotated_img_path),exist_ok=True)

    image.save(annotated_img_path)
    rendering_result['annotated_image_path'] = annotated_img_path

    return rendering_result

# ...def _render_image(...)


#%% Main function

def visualize_detector_output(detector_output_path,
                              out_dir,
                              images_dir=None,
                              confidence_threshold=0.15,
                              sample=-1,
                              output_image_width=1000,
                              random_seed=None,
                              render_detections_only=False,
                              classification_confidence_threshold=0.1,
                              html_output_file=None,
                              html_output_options=None,
                              preserve_path_structure=False,
                              parallelize_rendering=False,
                              parallelize_rendering_n_cores=10,
                              parallelize_rendering_with_threads=True,
                              box_sort_order=default_box_sort_order,
                              category_names_to_blur=None,
                              link_images_to_originals=False,
                              detector_label_map=None,
                              box_thickness=DEFAULT_BOX_THICKNESS,
                              box_expansion=0,
                              label_font_size=DEFAULT_LABEL_FONT_SIZE,
                              rounded_corners=False,
                              label_font='arial.ttf'):
    """
    Draws bounding boxes on images given the output of a detector.

    Args:
        detector_output_path (str): path to detector output .json file, or a loaded MD results
            dict
        out_dir (str): path to directory for saving annotated images
        images_dir (str, optional): folder where the images live; filenames in
            [detector_output_path] should be relative to [image_dir].  Can be None if paths are
            absolute.
        confidence_threshold (float, optional): threshold above which detections will be rendered
        sample (int, optional): maximum number of images to render, -1 for all
        output_image_width (int, optional): width in pixels to resize images for display,
            preserving aspect ration; set to -1 to use original image width
        random_seed (int, optional): seed to use for choosing images when sample != -1
        render_detections_only (bool, optional): only render images with above-threshold detections.
            Empty images are discarded after sampling, so if you want to see, e.g., 1000 non-empty
            images, you can set [render_detections_only], but you need to sample more than 1000 images.
        classification_confidence_threshold (float, optional): only show classifications
            above this threshold; does not impact whether images are rendered, only whether
            classification labels (not detection categories) are displayed
        html_output_file (str, optional): output path for an HTML index file (not written
            if None)
        html_output_options (dict, optional): HTML formatting options; see write_html_image_list
            for details.  The most common option you may want to supply here is
            'maxFiguresPerHtmlFile'.
        preserve_path_structure (bool, optional): if False (default), writes images to unique
            names in a flat structure in the output folder; if True, preserves relative paths
            within the output folder
        parallelize_rendering (bool, optional): whether to use concurrent workers for rendering
        parallelize_rendering_n_cores (int, optional): number of concurrent workers to use
            (ignored if parallelize_rendering is False)
        parallelize_rendering_with_threads (bool, optional): determines whether we use
            threads (True) or processes (False) for parallelization (ignored if parallelize_rendering
            is False)
        box_sort_order (str, optional): sorting scheme for detection boxes, can be None, "confidence", or
            "reverse_confidence"
        category_names_to_blur (list of str, optional): category names for which we should blur detections,
            most commonly ['person']
        link_images_to_originals (bool, optional): include a link from every rendered image back to
            the corresponding original image
        detector_label_map (dict, optional): mapping from category IDs to labels; by default (None) uses
            the values in the detector file.  If this is the string 'no_detection_labels', hides labels.
        box_thickness (int or float, optional): box thickness in pixels.  If this is a float less than
            1.0, it's treated as a fraction of the image width.
        box_expansion (int or float , optional): box expansion in pixels.  If this is a float less
            than 1.0, it's treated as a fraction of the image width.
        label_font_size (float, optional): label font size in pixels.  If this is a float less
            than 1.0, it's treated as a fraction of the image width.
        rounded_corners (bool, optional): use rounded-rectangle style for boxes and labels
            (default False)
        label_font (str, optional): font filename to use for label text (default 'arial.ttf')

    Returns:
        list: list of paths to annotated images
    """

    if isinstance(detector_output_path,str):
        assert os.path.exists(detector_output_path), \
            'Detector output file does not exist at {}'.format(detector_output_path)
    else:
        assert isinstance(detector_output_path,dict), \
            'detector_output_path is neither a filename nor a results dict'

    if images_dir is not None:
        assert os.path.isdir(images_dir), \
            'Image folder {} is not available'.format(images_dir)

    os.makedirs(out_dir, exist_ok=True)


    ##%% Load detector output

    if isinstance(detector_output_path,dict):
        detector_output = detector_output_path
    else:
        detector_output = load_md_or_speciesnet_file(detector_output_path)

    images = detector_output['images']

    if confidence_threshold is None:
        confidence_threshold = get_typical_confidence_threshold_from_results(detector_output)

    assert confidence_threshold >= 0 and confidence_threshold <= 1, \
        f'Confidence threshold {confidence_threshold} is invalid, must be in (0, 1).'

    if isinstance(detector_label_map,str):
        assert detector_label_map == 'no_detection_labels', \
            'Unrecognized detection label string {}'.format(detector_label_map)
        detector_label_map = None
    elif detector_label_map is not None:
        assert isinstance(detector_label_map,dict), \
            'Invalid detector label maps'
    elif 'detection_categories' in detector_output:
        detector_label_map = detector_output['detection_categories']
    else:
        detector_label_map = DEFAULT_DETECTOR_LABEL_MAP

    num_images = len(images)
    print(f'Detector output file contains {num_images} entries.')

    if (sample is not None) and (sample > 0) and (num_images > sample):

        if random_seed is not None:
            images = sorted(images, key=lambda x: x['file'])
            random.seed(random_seed)

        random.shuffle(images)
        images = sorted(images[:sample], key=lambda x: x['file'])
        print(f'Sampled {len(images)} entries from the detector output file.')


    ##%% Load images, annotate them and save

    print('Rendering detections above a confidence threshold of {}'.format(
        confidence_threshold))

    classification_label_map = None

    if 'classification_categories' in detector_output:
        classification_label_map = detector_output['classification_categories']

    rendering_results = []

    if parallelize_rendering:

        if parallelize_rendering_with_threads:
            worker_string = 'threads'
        else:
            worker_string = 'processes'

        pool = None
        try:
            if parallelize_rendering_n_cores is None:
                if parallelize_rendering_with_threads:
                    pool = ThreadPool()
                else:
                    pool = Pool()
            else:
                if parallelize_rendering_with_threads:
                    pool = ThreadPool(parallelize_rendering_n_cores)
                else:
                    pool = Pool(parallelize_rendering_n_cores)
                print('Rendering images with {} {}'.format(parallelize_rendering_n_cores,
                                                           worker_string))
            rendering_results = list(tqdm(pool.imap(
                                     partial(_render_image,detector_label_map=detector_label_map,
                                             classification_label_map=classification_label_map,
                                             confidence_threshold=confidence_threshold,
                                             classification_confidence_threshold=classification_confidence_threshold,
                                             render_detections_only=render_detections_only,
                                             preserve_path_structure=preserve_path_structure,
                                             out_dir=out_dir,
                                             images_dir=images_dir,
                                             output_image_width=output_image_width,
                                             box_sort_order=box_sort_order,
                                             category_names_to_blur=category_names_to_blur,
                                             box_thickness=box_thickness,
                                             box_expansion=box_expansion,
                                             label_font_size=label_font_size,
                                             rounded_corners=rounded_corners,
                                             label_font=label_font),
                                     images), total=len(images)))
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                print('Pool closed and joined for detector output visualization')

    else:

        for entry in tqdm(images):

            rendering_result = _render_image(entry,
                                             detector_label_map,
                                             classification_label_map,
                                             confidence_threshold,
                                             classification_confidence_threshold,
                                             render_detections_only,
                                             preserve_path_structure,
                                             out_dir,
                                             images_dir,
                                             output_image_width,
                                             box_sort_order,
                                             category_names_to_blur=category_names_to_blur,
                                             box_thickness=box_thickness,
                                             box_expansion=box_expansion,
                                             label_font_size=label_font_size,
                                             rounded_corners=rounded_corners,
                                             label_font=label_font)
            rendering_results.append(rendering_result)

    # ...for each image

    failed_images = [r for r in rendering_results if r['failed_image']]
    missing_images = [r for r in rendering_results if r['missing_image']]
    skipped_images = [r for r in rendering_results if r['skipped_image']]

    print('Skipped {} failed images (of {})'.format(len(failed_images),len(images)))
    print('Skipped {} missing images (of {})'.format(len(missing_images),len(images)))
    print('Skipped {} below-threshold images (of {})'.format(len(skipped_images),len(images)))

    print(f'Rendered detection results to {out_dir}')

    annotated_image_paths = [r['annotated_image_path'] for r in rendering_results if \
                             r['annotated_image_path'] is not None]

    if html_output_file is not None:

        html_dir = os.path.dirname(html_output_file)

        html_image_info = []

        for r in rendering_results:
            d = {}
            if r['annotated_image_path'] is None:
                assert r['failed_image'] or r['missing_image'] or r['skipped_image']
                continue
            annotated_image_path_relative = os.path.relpath(r['annotated_image_path'],html_dir)
            d['filename'] = annotated_image_path_relative
            # For sorting
            d['filename_lower'] = annotated_image_path_relative.lower()
            d['imageStyle'] = 'max-width:95%;'
            d['textStyle'] = \
             'font-family:verdana,arial,calibri;font-size:80%;' + \
                 'text-align:left;margin-top:20;margin-bottom:5'
            d['title'] = '{} (max conf: {})'.format(r['file'],r['max_conf'])
            if link_images_to_originals:
                d['linkTarget'] = r['image_filename_in_abs']
            html_image_info.append(d)

        html_image_info = sort_list_of_dicts_by_key(html_image_info,'filename_lower')
        _ = write_html_image_list.write_html_image_list(html_output_file,
                                                        html_image_info,
                                                        options=html_output_options)

    # ...if we're supposed to write HTML info

    return annotated_image_paths

# ...def visualize_detector_output(...)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Annotate the bounding boxes predicted by a detector above '
                    'some confidence threshold, and save the annotated images.')
    parser.add_argument(
        'detector_output_path', type=str,
        help='Path to json output file of the detector')
    parser.add_argument(
        'out_dir', type=str,
        help='Path to directory where the annotated images will be saved. '
             'The directory will be created if it does not exist.')
    parser.add_argument(
        '--confidence', type=float, default=0.15,
        help='Value between 0 and 1, indicating the confidence threshold '
             'above which to visualize bounding boxes')
    parser.add_argument(
        '--images_dir', type=str, default=None,
        help='Path to a local directory where images are stored. This '
             'serves as the root directory for image paths in '
             'detector_output_path.  Omit if image paths are absolute.')
    parser.add_argument(
        '--sample', type=int, default=-1,
        help='Number of images to be annotated and rendered. Set to -1 '
             '(default) to annotate all images in the detector output file. '
             'There may be fewer images if some are not found in images_dir.')
    parser.add_argument(
        '--output_image_width', type=int, default=1000,
        help='Integer, desired width in pixels of the output annotated images. '
             'Use -1 to not resize. Default: 1000.')
    parser.add_argument(
        '--random_seed', type=int, default=None,
        help='Integer, for deterministic order of image sampling')
    parser.add_argument(
        '--html_output_file', type=str, default=None,
        help='Filename to which we should write an HTML image index (off by default)')
    parser.add_argument(
        '--open_html_output_file', action='store_true',
        help='Open the .html output file when done')
    parser.add_argument(
        '--detections_only', action='store_true',
        help='Only render images with above-threshold detections (by default, '
             'both empty and non-empty images are rendered).')
    parser.add_argument(
        '--preserve_path_structure', action='store_true',
        help='Preserve relative image paths (otherwise flattens and assigns unique file names)')
    parser.add_argument(
        '--category_names_to_blur', default=None, type=str,
        help='Comma-separated list of category names to blur (or a single category name, typically "person")')
    parser.add_argument(
        '--classification_confidence', type=float, default=0.3,
        help='If classification results are present, render results above this threshold')
    parser.add_argument(
        '--box_thickness', type=float, default=DEFAULT_BOX_THICKNESS,
        help='Line thickness in pixels for box rendering.  If this is less than 1.0, '
             'it is treated as a fraction of the image width.')
    parser.add_argument(
        '--box_expansion', type=float, default=0,
        help='Number of pixels to expand bounding boxes on each side.  If this is less than 1.0, '
             'it is treated as a fraction of the image width.')
    parser.add_argument(
        '--label_font_size', type=float, default=DEFAULT_LABEL_FONT_SIZE,
        help='Font size in pixels for detection labels.  If this is less than 1.0, '
             'it is treated as a fraction of the image width.')
    parser.add_argument(
        '--rounded_corners', action='store_true', default=False,
        help='Use rounded-rectangle style for boxes and labels.')
    parser.add_argument(
        '--label_font', type=str, default='arial.ttf',
        help='Font filename to use for label text (default arial.ttf).')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    category_names_to_blur = args.category_names_to_blur
    if category_names_to_blur is not None:
        category_names_to_blur = category_names_to_blur.split(',')

    visualize_detector_output(
        detector_output_path=args.detector_output_path,
        out_dir=args.out_dir,
        confidence_threshold=args.confidence,
        images_dir=args.images_dir,
        sample=args.sample,
        output_image_width=args.output_image_width,
        random_seed=args.random_seed,
        render_detections_only=args.detections_only,
        classification_confidence_threshold=args.classification_confidence,
        preserve_path_structure=args.preserve_path_structure,
        html_output_file=args.html_output_file,
        category_names_to_blur=category_names_to_blur,
        box_thickness=args.box_thickness,
        box_expansion=args.box_expansion,
        label_font_size=args.label_font_size,
        rounded_corners=args.rounded_corners,
        label_font=args.label_font)

    if (args.html_output_file is not None) and args.open_html_output_file:
        print('Opening output file {}'.format(args.html_output_file))
        open_file(args.html_output_file)

if __name__ == '__main__':
    main()


#%% Interactive driver

if False:

    pass

    #%%

    detector_output_path = os.path.expanduser('detections.json')
    out_dir = r'g:\temp\preview'
    images_dir = r'g:\camera_traps\camera_trap_images'
    confidence_threshold = 0.15
    sample = 50
    output_image_width = 1000
    random_seed = 1
    render_detections_only = True
    classification_confidence_threshold = 0.1
    html_output_file = os.path.join(out_dir,'index.html')
    html_output_options = None
    preserve_path_structure = False
    parallelize_rendering = True
    parallelize_rendering_n_cores = 10
    parallelize_rendering_with_threads = False

    _ = visualize_detector_output(detector_output_path,
                                  out_dir,
                                  images_dir,
                                  confidence_threshold,
                                  sample,
                                  output_image_width,
                                  random_seed,
                                  render_detections_only,
                                  classification_confidence_threshold,
                                  html_output_file,
                                  html_output_options,
                                  preserve_path_structure,
                                  parallelize_rendering,
                                  parallelize_rendering_n_cores,
                                  parallelize_rendering_with_threads)

    from megadetector.utils.path_utils import open_file
    open_file(html_output_file)
