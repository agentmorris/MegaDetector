"""

visualize_db.py

Outputs an HTML page visualizing annotations (class labels and/or bounding boxes)
on a sample of images in a database in the COCO Camera Traps format.

"""

#%% Imports

import argparse
import inspect
import random
import json
import math
import os
import sys
import time

import pandas as pd
import numpy as np

import humanfriendly

from itertools import compress
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from tqdm import tqdm

from megadetector.utils.write_html_image_list import write_html_image_list
from megadetector.data_management.cct_json_utils import IndexedJsonDb
from megadetector.visualization import visualization_utils as vis_utils

def _isnan(x):
    return (isinstance(x,float) and np.isnan(x))


#%% Settings

class DbVizOptions:
    """
    Parameters controlling the behavior of visualize_db().
    """

    def __init__(self):

        #: Number of images to sample from the database, or None to visualize all images
        self.num_to_visualize = None

        #: Target size for rendering; set either dimension to -1 to preserve aspect ratio.
        #:
        #: If viz_size is None or (-1,-1), the original image size is used.
        self.viz_size = (1000, -1)

        #: HTML rendering options; see write_html_image_list for details
        #:
        #:The most relevant option one might want to set here is:
        #:
        #: html_options['maxFiguresPerHtmlFile']
        #:
        #: ...which can be used to paginate previews to a number of images that will load well
        #: in a browser (5000 is a reasonable limit).
        self.html_options = write_html_image_list()

        #: Whether to sort images by filename (True) or randomly (False)
        self.sort_by_filename = True

        #: Only show images that contain bounding boxes
        self.trim_to_images_with_bboxes = False

        #: Random seed to use for sampling images
        self.random_seed = 0

        #: Should we include Web search links for each category name?
        self.add_search_links = False

        #: Should each thumbnail image link back to the original image?
        self.include_image_links = False

        #: Should there be a text link back to each original image?
        self.include_filename_links = False

        #: Line width in pixels
        self.box_thickness = 4

        #: Number of pixels to expand each bounding box
        self.box_expansion = 0

        #: Only include images that contain annotations with these class names (not IDs) (list)
        #:
        #: Mutually exclusive with classes_to_exclude
        self.classes_to_include = None

        #: Exclude images that contain annotations with these class names (not IDs) (list)
        #:
        #: Mutually exclusive with classes_to_include
        self.classes_to_exclude = None

        #: Special tag used to say "show me all images with multiple categories"
        #:
        #: :meta private:
        self.multiple_categories_tag = '*multiple*'

        #: We sometimes flatten image directories by replacing a path separator with
        #: another character.  Leave blank for the typical case where this isn't necessary.
        self.pathsep_replacement = '' # '~'

        #: Parallelize rendering across multiple workers
        self.parallelize_rendering = False

        #: In theory, whether to parallelize with threads (True) or processes (False), but
        #: process-based parallelization in this function is currently unsupported
        self.parallelize_rendering_with_threads = True

        #: Number of workers to use for parallelization; ignored if parallelize_rendering
        #: is False
        self.parallelize_rendering_n_cores = 25

        #: Should we show absolute (True) or relative (False) paths for each image?
        self.show_full_paths = False

        #: List of additional fields in the image struct that we should print in image headers
        self.extra_image_fields_to_print = None

        #: List of additional fields in the annotation struct that we should print in image headers
        self.extra_annotation_fields_to_print = None

        #: Set to False to skip existing images
        self.force_rendering = True

        #: Enable additionald debug console output
        self.verbose = False

        #: COCO files used for evaluation may contain confidence scores, this
        #: determines the field name used for confidence scores
        self.confidence_field_name = 'score'

        #: Optionally apply a confidence threshold; this requires that [confidence_field_name]
        #: be present in all detections.
        self.confidence_threshold = None


#%% Helper functions

def _image_filename_to_path(image_file_name, image_base_dir, pathsep_replacement=''):
    """
    Translates the file name in an image entry in the json database to a path, possibly doing
    some manipulation of path separators.
    """

    if len(pathsep_replacement) > 0:
        image_file_name = os.path.normpath(image_file_name).replace(os.pathsep,pathsep_replacement)
    return os.path.join(image_base_dir, image_file_name)


#%% Core functions

def visualize_db(db_path, output_dir, image_base_dir, options=None):
    """
    Writes images and html to output_dir to visualize the annotations in a .json file.

    Args:
        db_path (str or dict): the .json filename to load, or a previously-loaded database
        output_dir (str): the folder to which we should write annotated images
        image_base_dir (str): the folder where the images live; filenames in [db_path] should
            be relative to this folder.
        options (DbVizOptions, optional): See DbVizOptions for details

    Returns:
        tuple: A length-two tuple containing (the html filename) and (the loaded database).
    """

    if options is None:
        options = DbVizOptions()

    # Consistency checking for fields with specific format requirements

    # This should be a list, but if someone specifies a string, do a reasonable thing
    if isinstance(options.extra_image_fields_to_print,str):
        options.extra_image_fields_to_print = [options.extra_image_fields_to_print]

    if not options.parallelize_rendering_with_threads:
        print('Warning: process-based parallelization is not yet supported by visualize_db')
        options.parallelize_rendering_with_threads = True

    if image_base_dir.startswith('http'):
        if not image_base_dir.endswith('/'):
            image_base_dir += '/'
    else:
        assert(os.path.isdir(image_base_dir))

    os.makedirs(os.path.join(output_dir, 'rendered_images'), exist_ok=True)

    if isinstance(db_path,str):
        assert(os.path.isfile(db_path))
        print('Loading database from {}...'.format(db_path))
        image_db = json.load(open(db_path))
        print('...done')
    elif isinstance(db_path,dict):
        print('Using previously-loaded DB')
        image_db = db_path
    else:
        raise ValueError('Illegal dictionary or filename')

    annotations = image_db['annotations']
    images = image_db['images']
    categories = image_db['categories']

    # Optionally remove all images without bounding boxes, *before* sampling
    if options.trim_to_images_with_bboxes:

        b_has_bbox = [False] * len(annotations)
        for i_ann,ann in enumerate(annotations):
            if 'bbox' in ann or 'bbox_relative' in ann:
                if 'bbox' in ann:
                    assert isinstance(ann['bbox'],list)
                else:
                    assert isinstance(ann['bbox_relative'],list)
                b_has_bbox[i_ann] = True
        annotations_with_boxes = list(compress(annotations, b_has_bbox))

        image_ids_with_boxes = [x['image_id'] for x in annotations_with_boxes]
        image_ids_with_boxes = set(image_ids_with_boxes)

        image_has_box = [False] * len(images)
        for i_image,image in enumerate(images):
            image_id = image['id']
            if image_id in image_ids_with_boxes:
                image_has_box[i_image] = True
        images_with_bboxes = list(compress(images, image_has_box))
        images = images_with_bboxes

    # Optionally include/remove images with specific labels, *before* sampling

    assert (not ((options.classes_to_exclude is not None) and \
                 (options.classes_to_include is not None))), \
                 'Cannot specify an inclusion and exclusion list'

    if options.classes_to_exclude is not None:
        assert isinstance(options.classes_to_exclude,list), \
            'If supplied, classes_to_exclude should be a list'

    if options.classes_to_include is not None:
        assert isinstance(options.classes_to_include,list), \
            'If supplied, classes_to_include should be a list'

    if (options.classes_to_exclude is not None) or (options.classes_to_include is not None):

        print('Indexing database')
        indexed_db = IndexedJsonDb(image_db)
        b_valid_class = [True] * len(images)
        for i_image,image in enumerate(images):
            classes = indexed_db.get_classes_for_image(image)
            if options.classes_to_exclude is not None:
                for excluded_class in options.classes_to_exclude:
                    if excluded_class in classes:
                       b_valid_class[i_image] = False
                       break
            elif options.classes_to_include is not None:
                b_valid_class[i_image] = False
                if options.multiple_categories_tag in options.classes_to_include:
                    if len(classes) > 1:
                        b_valid_class[i_image] = True
                if not b_valid_class[i_image]:
                    for c in classes:
                        if c in options.classes_to_include:
                            b_valid_class[i_image] = True
                            break
            else:
                raise ValueError('Illegal include/exclude combination')

        images_with_valid_classes = list(compress(images, b_valid_class))
        images = images_with_valid_classes

    # ...if we need to include/exclude categories

    # Put the annotations in a dataframe so we can select all annotations for a given image
    print('Creating data frames')
    df_anno = pd.DataFrame(annotations)
    df_img = pd.DataFrame(images)

    # Construct label map
    label_map = {}
    for cat in categories:
        label_map[int(cat['id'])] = cat['name']

    # Take a sample of images
    if options.num_to_visualize is not None:
        if options.num_to_visualize > len(df_img):
            print('Warning: asked to visualize {} images, but only {} are available, keeping them all'.\
                  format(options.num_to_visualize,len(df_img)))
        else:
            df_img = df_img.sample(n=options.num_to_visualize,random_state=options.random_seed)

    images_html = []

    # Set of dicts representing inputs to render_db_bounding_boxes:
    #
    # bboxes, box_classes, image_path
    rendering_info = []

    print('Preparing rendering list')

    for i_image,img in tqdm(df_img.iterrows(),total=len(df_img)):

        img_id = img['id']
        assert img_id is not None

        img_relative_path = img['file_name']

        if image_base_dir.startswith('http'):
            img_path = image_base_dir + img_relative_path
        else:
            img_path = os.path.join(image_base_dir,
                                    _image_filename_to_path(img_relative_path, image_base_dir))

        annos_i = df_anno.loc[df_anno['image_id'] == img_id, :] # all annotations on this image

        bboxes = []
        box_classes = []
        box_score_strings = []

        # All the class labels we've seen for this image (with or without bboxes)
        image_categories = set()

        extra_annotation_field_string = ''
        annotation_level_for_image = ''

        # Did this image come with already-normalized bounding boxes?
        boxes_are_normalized = None

        # Iterate over annotations for this image
        # i_ann = 0; anno = annos_i.iloc[i_ann]
        for i_ann,anno in annos_i.iterrows():

            if options.extra_annotation_fields_to_print is not None:
                field_names = list(anno.index)
                for field_name in field_names:
                    if field_name in options.extra_annotation_fields_to_print:
                        field_value = anno[field_name]
                        if (field_value is not None) and (not _isnan(field_value)):
                            extra_annotation_field_string += ' ({}:{})'.format(
                                field_name,field_value)

            if options.confidence_threshold is not None:
                assert options.confidence_field_name in anno, \
                    'Error: confidence thresholding requested, ' + \
                        'but at least one annotation does not have the {} field'.format(
                            options.confidence_field_name)
                if anno[options.confidence_field_name] < options.confidence_threshold:
                    continue

            if 'sequence_level_annotation' in anno:
                b_sequence_level_annotation = anno['sequence_level_annotation']
                if b_sequence_level_annotation:
                    annotation_level = 'sequence'
                else:
                    annotation_level = 'image'
                if annotation_level_for_image == '':
                    annotation_level_for_image = annotation_level
                elif annotation_level_for_image != annotation_level:
                    annotation_level_for_image = 'mixed'

            category_id = anno['category_id']
            category_name = label_map[category_id]
            if options.add_search_links:
                category_name = category_name.replace('"','')
                category_name = '<a href="https://www.google.com/search?tbm=isch&q={}">{}</a>'.format(
                    category_name,category_name)

            image_categories.add(category_name)

            assert not ('bbox' in anno and 'bbox_relative' in anno), \
                "An annotation can't have both an absolute and a relative bounding box"

            box_field = 'bbox'
            if 'bbox_relative' in anno:
                box_field = 'bbox_relative'
                assert (boxes_are_normalized is None) or (boxes_are_normalized), \
                    "An image can't have both absolute and relative bounding boxes"
                boxes_are_normalized = True
            elif 'bbox' in anno:
                assert (boxes_are_normalized is None) or (not boxes_are_normalized), \
                    "An image can't have both absolute and relative bounding boxes"
                boxes_are_normalized = False

            if box_field in anno:
                bbox = anno[box_field]
                if isinstance(bbox,float):
                    assert math.isnan(bbox), "I shouldn't see a bbox that's neither a box nor NaN"
                    continue
                bboxes.append(bbox)
                box_classes.append(anno['category_id'])

                box_score_string = ''
                if options.confidence_field_name is not None and \
                   options.confidence_field_name in anno:
                       score = anno[options.confidence_field_name]
                       box_score_string = '({}%)'.format(round(100 * score))
                box_score_strings.append(box_score_string)

        # ...for each of this image's annotations

        image_classes = ', '.join(image_categories)

        img_id_string = str(img_id).lower()
        file_name = '{}_gt.jpg'.format(os.path.splitext(img_id_string)[0])

        # Replace characters that muck up image links
        illegal_characters = ['/','\\',':','\t','#',' ','%']
        for c in illegal_characters:
            file_name = file_name.replace(c,'~')

        rendering_info_this_image = {'bboxes':bboxes,
                                     'box_classes':box_classes,
                                     'tags':box_score_strings,
                                     'img_path':img_path,
                                     'output_file_name':file_name,
                                     'boxes_are_normalized':boxes_are_normalized}
        rendering_info.append(rendering_info_this_image)

        label_level_string = ''
        if len(annotation_level_for_image) > 0:
            label_level_string = ' (annotation level: {})'.format(annotation_level_for_image)

        if 'frame_num' in img and 'seq_num_frames' in img:
            frame_string = ' frame: {} of {},'.format(img['frame_num'],img['seq_num_frames'])
        elif 'frame_num' in img:
            frame_string = ' frame: {},'.format(img['frame_num'])
        else:
            frame_string = ''

        if options.show_full_paths:
            filename_text = img_path
        else:
            filename_text = img_relative_path
        if options.include_filename_links:
            filename_text = '<a href="{}">{}</a>'.format(img_path,filename_text)

        flag_string = ''

        if ('flags' in img) and (not _isnan(img['flags'])):
            flag_string = ', flags: {}'.format(str(img['flags']))

        extra_field_string = ''

        if options.extra_image_fields_to_print is not None:
            for field_name in options.extra_image_fields_to_print:
                if field_name in img:
                    # Always include a leading comma; this either separates us from the
                    # previous field in [extra_fields_to_print] or from the rest of the string
                    extra_field_string += ', {}: {}'.format(
                        field_name,str(img[field_name]))

        # We're adding html for an image before we render it, so it's possible this image will
        # fail to render.  For applications where this script is being used to debua a database
        # (the common case?), this is useful behavior, for other applications, this is annoying.
        image_dict = \
        {
            'filename': '{}/{}'.format('rendered_images', file_name),
            'title': '{}<br/>{}, num boxes: {},{} class labels: {}{}{}{}{}'.format(
                filename_text, img_id, len(bboxes), frame_string, image_classes,
                label_level_string, flag_string, extra_field_string, extra_annotation_field_string),
            'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;' + \
                'text-align:left;margin-top:20;margin-bottom:5'
        }
        if options.include_image_links:
            image_dict['linkTarget'] = img_path

        images_html.append(image_dict)

    # ...for each image

    def render_image_info(rendering_info):

        img_path = rendering_info['img_path']
        bboxes = rendering_info['bboxes']
        bbox_classes = rendering_info['box_classes']
        boxes_are_normalized = rendering_info['boxes_are_normalized']
        bbox_tags = None
        if 'tags' in rendering_info:
            bbox_tags = rendering_info['tags']
        output_file_name = rendering_info['output_file_name']
        output_full_path = os.path.join(output_dir, 'rendered_images', output_file_name)

        if (os.path.isfile(output_full_path)) and (not options.force_rendering):
            if options.verbose:
                print('Skipping existing image {}'.format(output_full_path))
            return True

        if not img_path.startswith('http'):
            if not os.path.exists(img_path):
                print('Image {} cannot be found'.format(img_path))
                return False

        try:
            original_image = vis_utils.open_image(img_path)
            original_size = original_image.size
            if (options.viz_size is None) or \
                (options.viz_size[0] == -1 and options.viz_size[1] == -1):
                image = original_image
            else:
                image = vis_utils.resize_image(original_image,
                                               options.viz_size[0],
                                               options.viz_size[1],
                                               no_enlarge_width=True)
        except Exception as e:
            print('Image {} failed to open, error: {}'.format(img_path, e))
            return False

        vis_utils.render_db_bounding_boxes(boxes=bboxes,
                                           classes=bbox_classes,
                                           image=image,
                                           original_size=original_size,
                                           label_map=label_map,
                                           thickness=options.box_thickness,
                                           expansion=options.box_expansion,
                                           tags=bbox_tags,
                                           boxes_are_normalized=boxes_are_normalized)

        image.save(output_full_path)

        return True

    # ...def render_image_info(...)

    print('Rendering images')
    start_time = time.time()

    if options.parallelize_rendering:

        if options.parallelize_rendering_with_threads:
            worker_string = 'threads'
        else:
            worker_string = 'processes'

        pool = None
        try:
            if options.parallelize_rendering_n_cores is None:
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool()
                else:
                    pool = Pool()
            else:
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool(options.parallelize_rendering_n_cores)
                else:
                    pool = Pool(options.parallelize_rendering_n_cores)
                print('Rendering images with {} {}'.format(options.parallelize_rendering_n_cores,
                                                           worker_string))
            rendering_success = list(tqdm(pool.imap(render_image_info, rendering_info),
                                     total=len(rendering_info)))
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                print("Pool closed and joined for DB visualization")

    else:

        rendering_success = []
        for file_info in tqdm(rendering_info):
            rendering_success.append(render_image_info(file_info))

    elapsed = time.time() - start_time

    print('Rendered {} images in {} ({} successful)'.format(
        len(rendering_info),humanfriendly.format_timespan(elapsed),sum(rendering_success)))

    if options.sort_by_filename:
        images_html = sorted(images_html, key=lambda x: x['filename'])
    else:
        random.shuffle(images_html)

    html_output_file = os.path.join(output_dir, 'index.html')

    html_options = options.html_options
    if isinstance(db_path,str):
        html_options['headerHtml'] = '<h1>Sample annotations from {}</h1>'.format(db_path)
    else:
        html_options['headerHtml'] = '<h1>Sample annotations</h1>'

    write_html_image_list(
            filename=html_output_file,
            images=images_html,
            options=html_options)

    print('Visualized {} images, wrote results to {}'.format(len(images_html),html_output_file))

    return html_output_file,image_db

# ...def visualize_db(...)


#%% Command-line driver

# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.
#
# Skips fields starting with _.  Does not check existence in the target object.
def _args_to_object(args, obj):

    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)

def main():
    """
    Command-line driver for visualize_db
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', action='store', type=str,
                        help='.json file to visualize')
    parser.add_argument('output_dir', action='store', type=str,
                        help='Output directory for html and rendered images')
    parser.add_argument('image_base_dir', action='store', type=str,
                        help='Base directory (or URL) for input images')

    parser.add_argument('--num_to_visualize', action='store', type=int, default=None,
                        help='Number of images to visualize (randomly drawn) (defaults to all)')
    parser.add_argument('--random_sort', action='store_true',
                        help='Sort randomly (rather than by filename) in output html')
    parser.add_argument('--trim_to_images_with_bboxes', action='store_true',
                        help='Only include images with bounding boxes (defaults to false)')
    parser.add_argument('--random_seed', action='store', type=int, default=None,
                        help='Random seed for image selection')
    parser.add_argument('--pathsep_replacement', action='store', type=str, default='',
                        help='Replace path separators in relative filenames with another ' + \
                             'character (frequently ~)')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # Convert to an options object
    options = DbVizOptions()
    _args_to_object(args, options)
    if options.random_sort:
        options.sort_by_filename = False

    visualize_db(options.db_path,options.output_dir,options.image_base_dir,options)

if __name__ == '__main__':
    main()
