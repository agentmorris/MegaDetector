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

import numpy as np

import humanfriendly

from itertools import compress
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from copy import deepcopy
from tqdm import tqdm

from megadetector.utils.write_html_image_list import write_html_image_list
from megadetector.utils.path_utils import clean_filename
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
        #:
        #: This is ignored if max_sequence_length is not None, in which case
        #: we always sort by sequence ID, then frame number.
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

        #: Parallelize rendering across multiple workers
        self.parallelize_rendering = False

        #: In theory, whether to parallelize with threads (True) or processes (False), but
        #: process-based parallelization in this function is currently unsupported
        self.parallelize_rendering_with_threads = True

        #: Number of workers to use for parallelization; ignored if parallelize_rendering
        #: is False
        self.parallelize_rendering_n_cores = 16

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

        #: Custom mapping from category IDs to labels, replacing what's in the .json file
        self.custom_category_mapping = None

        #: JPEG quality to use for saving images (None for Pillow default)
        self.quality = None

        #: List of PIL color names, which will be indexed by category IDs, or None
        #: to use the default color map.
        #:
        #: For example: ['AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse']
        self.colormap = None

        #: Should we create separate pages for each category (within the sampled set)?
        #:
        #: Images with multiple categories will be included in all relevant pages.
        self.create_category_pages = False

        #: If this is None, we just sample images, and show images.  If this is
        #: not None, we sample images, but we also show the other images in the sequences
        #: containing our sampled images.  If this is <=0, there is no limit on the
        #: number of images we'll show per sequences.  If this is >0, we will cap the number
        #: of images shown per sequence; no guarantee is made about which images will
        #: be selected in that case.  This only impacts the number of images added as
        #: "sequence friends" of images that get sampled.
        self.max_sequence_length = None


#%% Core functions

def visualize_db(db_path, output_dir, image_base_dir, options=None):
    """
    Writes images and html to output_dir to visualize the images and annotations in a
    COCO-formatted .json file.

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

    # These should be a lists, but if someone specifies a string, do a reasonable thing
    if isinstance(options.extra_image_fields_to_print,str):
        options.extra_image_fields_to_print = [options.extra_image_fields_to_print]
    if isinstance(options.extra_annotation_fields_to_print,str):
        options.extra_annotation_fields_to_print = [options.extra_annotation_fields_to_print]

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
        print('...done, loaded {} images'.format(len(image_db['images'])))
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

    # Construct label map
    label_map = {}
    for cat in categories:
        label_map[int(cat['id'])] = cat['name']

    # Map sequence IDs to images if necessary
    sequence_id_to_images = defaultdict(list)

    if options.max_sequence_length is not None:
        for im in images:
            if 'seq_id' in im:
                sequence_id_to_images[im['seq_id']].append(im)

    # Take a sample of images
    if options.num_to_visualize is not None:
        if options.num_to_visualize > len(images):
            print('Warning: asked to visualize {} images, but only {} are available, keeping them all'.\
                  format(options.num_to_visualize,len(images)))
        else:
            random.seed(options.random_seed)
            images = random.sample(images,options.num_to_visualize)

    # List of dicts containing HTML image information suitable for passing to
    # write_html_image_list
    images_html = []

    # Set of dicts representing inputs to _render_image_info, with fields:
    #
    # bboxes, box_classes, tags, img_path, output_file_name, boxes_are_normalies
    rendering_info = []

    image_id_to_annotations = defaultdict(list)

    for ann in annotations:
        image_id_to_annotations[ann['image_id']].append(ann)

    # Do we need to find all the images that are in the same
    # sequence as the sampled images?
    if options.max_sequence_length is not None:

        print('Preparing sequence friends')

        seq_id_to_images_rendered = defaultdict(int)

        # Don't add images that are already in the list
        image_ids_already_sampled = set()
        for im in images:
            image_ids_already_sampled.add(im['id'])
            if 'seq_id' in im:
                seq_id_to_images_rendered[im['seq_id']] += 1

        images_to_add = []

        # For every image we've decided to render, find all her
        # sequence friends
        for im in images:

            if 'seq_id' not in im:
                continue

            seq_id = im['seq_id']

            # This is *all* the images in this sequence, regardless of
            # whether we've already added them
            images_this_sequence = sequence_id_to_images[seq_id]

            for candidate_im in images_this_sequence:

                # Have we already hit the maximum number of images for this sequence?
                if (options.max_sequence_length > 0) and \
                   (seq_id_to_images_rendered[seq_id] >= options.max_sequence_length):
                    # Don't assert this, we may have more than max_sequence_length images
                    # if we just happened to sample that way, we just won't add more.
                    # assert seq_id_to_images_rendered[seq_id] == options.max_sequence_length
                    if options.verbose:
                        print('Already rendered {} images from sequence {}'.format(
                            options.max_sequence_length,seq_id))
                    break

                assert candidate_im['seq_id'] == seq_id

                # Add this image if necessary
                if candidate_im['id'] not in image_ids_already_sampled:

                    images_to_add.append(candidate_im)
                    image_ids_already_sampled.add(candidate_im['id'])
                    seq_id_to_images_rendered[seq_id] += 1

            # ...for each sequence friend of the current image

        # ...for each image in the sampled set

        print('Adding {} new images ({} initially) as sequence friends'.format(
            len(images_to_add),len(images)))
        images.extend(images_to_add)

        # Double-check that we didn't create duplicates
        all_image_ids = [im['id'] for im in images]
        assert len(all_image_ids) == len(set(all_image_ids))

    # ...if we need to add sequence friends for every sampled image

    print('Preparing rendering list')

    for i_image,img in tqdm(enumerate(images),total=len(images)):

        img_id = img['id']
        assert img_id is not None

        img_relative_path = img['file_name']

        if image_base_dir.startswith('http'):
            img_path = image_base_dir + img_relative_path
        else:
            img_path = os.path.join(image_base_dir,img_relative_path).replace('\\','/')

        if img_id in image_id_to_annotations:
            annotations_this_image = image_id_to_annotations[img_id]
        else:
            annotations_this_image = []

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
        for i_ann,ann in enumerate(annotations_this_image):

            if options.extra_annotation_fields_to_print is not None:
                field_names = list(ann.index)
                for field_name in field_names:
                    if field_name in options.extra_annotation_fields_to_print:
                        field_value = ann[field_name]
                        if (field_value is not None) and (not _isnan(field_value)):
                            extra_annotation_field_string += ' ({}:{})'.format(
                                field_name,field_value)

            if options.confidence_threshold is not None:
                assert options.confidence_field_name in ann, \
                    'Error: confidence thresholding requested, ' + \
                        'but at least one annotation does not have the {} field'.format(
                            options.confidence_field_name)
                if ann[options.confidence_field_name] < options.confidence_threshold:
                    continue

            if 'sequence_level_annotation' in ann:
                b_sequence_level_annotation = ann['sequence_level_annotation']
                if b_sequence_level_annotation:
                    annotation_level = 'sequence'
                else:
                    annotation_level = 'image'
                if annotation_level_for_image == '':
                    annotation_level_for_image = annotation_level
                elif annotation_level_for_image != annotation_level:
                    annotation_level_for_image = 'mixed'

            category_id = ann['category_id']
            category_name = label_map[category_id]
            if options.add_search_links:
                category_name = category_name.replace('"','')
                category_name = '<a href="https://www.google.com/search?tbm=isch&q={}">{}</a>'.format(
                    category_name,category_name)

            image_categories.add(category_name)

            assert not ('bbox' in ann and 'bbox_relative' in ann), \
                "An annotation can't have both an absolute and a relative bounding box"

            box_field = 'bbox'
            if 'bbox_relative' in ann:
                box_field = 'bbox_relative'
                assert (boxes_are_normalized is None) or (boxes_are_normalized), \
                    "An image can't have both absolute and relative bounding boxes"
                boxes_are_normalized = True
            elif 'bbox' in ann:
                assert (boxes_are_normalized is None) or (not boxes_are_normalized), \
                    "An image can't have both absolute and relative bounding boxes"
                boxes_are_normalized = False

            if box_field in ann:
                bbox = ann[box_field]
                if isinstance(bbox,float):
                    assert math.isnan(bbox), "I shouldn't see a bbox that's neither a box nor NaN"
                    continue
                bboxes.append(bbox)
                box_classes.append(ann['category_id'])

                box_score_string = ''
                if options.confidence_field_name is not None and \
                   options.confidence_field_name in ann:
                       score = ann[options.confidence_field_name]
                       box_score_string = '({}%)'.format(round(100 * score))
                box_score_strings.append(box_score_string)

        # ...for each of this image's annotations

        image_classes = ', '.join(image_categories)

        img_id_string = str(img_id).lower()
        file_name = '{}_gt.jpg'.format(os.path.splitext(img_id_string)[0])

        # Replace characters that muck up image links, including flattening file
        # separators.
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
        # fail to render.  For applications where this script is being used to debug a database
        # (the common case), this is useful behavior, because we still see a record of this
        # item in the HTML output.  For other applications, it's annoying to have broken images.
        image_dict = \
        {
            'filename': '{}/{}'.format('rendered_images', file_name),
            'title': '{}<br/>{}, num boxes: {},{} class labels: {}{}{}{}{}'.format(
                filename_text, img_id, len(bboxes), frame_string, image_classes,
                label_level_string, flag_string, extra_field_string, extra_annotation_field_string),
            'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;' + \
                'text-align:left;margin-top:20;margin-bottom:5',
            'image_categories': image_categories
        }

        # Make it clear which images start a new sequence
        if (options.max_sequence_length is not None) and \
           ('frame_num' in img) and ('seq_id' in img) and \
           (img['frame_num'] is not None) and (img['frame_num'] in (0,-1)):
            sequence_header = '<span style="font-weight:bold;font-size:120%">Sequence {}</span><br/><br/>'.format(
                img['seq_id'])
            image_dict['title'] = sequence_header + image_dict['title']

        if options.include_image_links:
            image_dict['linkTarget'] = img_path

        for field_name in ('seq_id','frame_num'):
            if field_name in img:
                image_dict[field_name] = img[field_name]

        images_html.append(image_dict)

    # ...for each sampled image

    def _render_image_info(rendering_info):
        """
        Render one image.

        Args:
            rendering_info (dict): a dict with fields:
                - img_path: the absolute path (or URL) to the input image
                - bboxes: a list of bbox dicts to render (or [])
                - bbox_classes: a list of category IDs the same length as [bboxes]
                - output_file_name: the relative output path to which we should render
                - tags (optional): a list of additional text tags to include with each box

        Returns:
            bool: True if rendering was successful, else False
        """

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

        rendering_label_map = None
        if options.custom_category_mapping is not None:
            rendering_label_map = options.custom_category_mapping

        vis_utils.render_db_bounding_boxes(boxes=bboxes,
                                           classes=bbox_classes,
                                           image=image,
                                           original_size=original_size,
                                           label_map=rendering_label_map,
                                           thickness=options.box_thickness,
                                           expansion=options.box_expansion,
                                           tags=bbox_tags,
                                           boxes_are_normalized=boxes_are_normalized,
                                           colormap=options.colormap)

        if options.quality is None:
            image.save(output_full_path)
        else:
            image.save(output_full_path,quality=options.quality)

        return True

    # ...def _render_image_info(...)

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
            rendering_success = list(tqdm(pool.imap(_render_image_info, rendering_info),
                                     total=len(rendering_info)))
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                print("Pool closed and joined for DB visualization")

    else:

        rendering_success = []
        for file_info in tqdm(rendering_info):
            rendering_success.append(_render_image_info(file_info))

    elapsed = time.time() - start_time

    print('Rendered {} images in {} ({} successful)'.format(
        len(rendering_info),humanfriendly.format_timespan(elapsed),sum(rendering_success)))

    # If we added every sampled image's sequence friends, we sort by sequence ID,
    # then frame number
    if options.max_sequence_length is not None:

        # Add fields we need for sorting by sequence
        for im in images_html:
            if 'seq_id' not in im:
                im['seq_id'] = 'no_sequence_id'
            if 'frame_num' not in im:
                im['frame_num'] = -1

        # Sort the list of dicts "images_html" by the
        # field "seq_id", then "frame_num", then "filename"
        images_html = sorted(images_html,
                             key=lambda x: (x['seq_id'], x['frame_num'], x['filename']))

    else:

        if options.sort_by_filename:
            images_html = sorted(images_html, key=lambda x: x['filename'])
        else:
            random.shuffle(images_html)

    # ...if we need to sort using sequence information

    html_output_file = None

    if options.create_category_pages:

        all_categories = set()
        for im in images_html:
            categories_this_image = im['image_categories']
            for category_name in categories_this_image:
                all_categories.add(category_name)

        all_categories = sorted(list(all_categories))

        # Create a special category for images with no annotations
        no_category_token = 'no_categories'
        all_categories.append(no_category_token)

        category_name_to_relative_filename = {}
        category_name_to_count = {}

        for category_name in all_categories:

            category_name_clean = clean_filename(category_name,
                                                 force_lower=True,
                                                 replace_whitespace='_')
            category_filename_relative = category_name_clean + '.html'
            category_name_to_relative_filename[category_name] = category_filename_relative
            category_filename_abs = os.path.join(output_dir,category_filename_relative)

            # Find images that should go in this category page
            images_this_category = []

            for im in images_html:

                include_this_image = False
                if category_name == no_category_token:
                    include_this_image = (len(im['image_categories']) == 0)
                else:
                    include_this_image = category_name in im['image_categories']
                if include_this_image:
                    images_this_category.append(im)

            # ...for each image

            category_name_to_count[category_name] = len(images_this_category)

            title_string = '<h1>Sample annotations for category {}</h1>'.format(
                category_name)

            html_options = deepcopy(options.html_options)
            html_options['headerHtml'] = title_string

            write_html_image_list(
                filename=category_filename_abs,
                images=images_this_category,
                options=html_options)

        # ...for each category

        html_output_file = os.path.join(output_dir, 'index.html')

        if isinstance(db_path,str):
            title_string = '<h1>Sample annotations from {}</h1>'.format(db_path)
        else:
            title_string = '<h1>Sample annotations</h1>'

        with open(html_output_file,'w') as f:
            f.write('<html><head>{}</head><body>\n'.format(title_string))
            for category_name in category_name_to_relative_filename:
                s = '<p style="padding:0px;margin:0px;margin-left:15px;text-align:left;'
                s += 'font-family:''segoe ui'',calibri,arial;font-size:100%;text-decoration:none;font-weight:normal;">'
                f.write(s)
                f.write('<a href="{}">{}</a> ({} images)</p>\n'.format(
                    category_name_to_relative_filename[category_name],
                    category_name,
                    category_name_to_count[category_name]
                ))
            f.write('</body></html>\n')

    else:

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
