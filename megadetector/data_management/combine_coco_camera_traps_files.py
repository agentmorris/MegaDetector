"""

combine_coco_camera_traps_files.py

Merges two or more .json files in COCO Camera Traps format, optionally
writing the results to another .json file.

- Concatenates image lists, erroring if images are not unique.
- Errors on unrecognized fields.
- Checks compatibility in info structs, within reason.

"""

#%% Constants and imports

import os
import sys
import json
import argparse

from copy import deepcopy

from megadetector.utils.ct_utils import write_json
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key


#%% Merge functions

def combine_cct_files(input_files,
                      output_file=None,
                      require_uniqueness=True,
                      filename_prefixes=None):
    """
    Merges the list of COCO Camera Traps files [input_files] into a single
    dictionary, optionally writing the result to [output_file].

    Args:
        input_files (list): paths to CCT .json files
        output_file (str, optional): path to write merged .json file
        require_uniqueness (bool, optional): whether to require that the images in
            each input_dict be unique
        filename_prefixes (dict, optional): dict mapping input filenames to strings
            that should be prepended to image filenames from that source

    Returns:
        dict: the merged COCO-formatted .json dict
    """

    # Argument validation
    for fn in input_files:
        assert os.path.isfile(fn), 'Could not find file {}'.format(fn)

    if filename_prefixes is not None:
        assert isinstance(filename_prefixes,dict), 'filename_prefixes must be a dict'
        assert len(filename_prefixes) == len(input_files), 'Prefix dict mismatch'

    input_dicts = []
    print('Loading input files')
    for fn in input_files:
        with open(fn, 'r', encoding='utf-8') as f:
            d = json.load(f)
            if filename_prefixes is not None:
                assert fn in filename_prefixes, \
                    'No prefix mapping for {}'.format(fn)
                d['filename_prefix'] = filename_prefixes[fn]
            input_dicts.append(d)

    print('Merging results')
    merged_dict = combine_cct_dictionaries(
        input_dicts,
        require_uniqueness=require_uniqueness)

    print('Writing output')
    if output_file is not None:
        write_json(output_file, merged_dict)

    return merged_dict

# ...def combine_cct_files(...)


def combine_cct_dictionaries(input_dicts,
                             require_uniqueness=True):
    """
    Merges the list of COCO Camera Traps dictionaries [input_dicts].  See module header
    comment for details on merge rules.

    Args:
        input_dicts (list of dict): list of CCT dicts
        require_uniqueness (bool, optional): whether to require that the images in
            each input_dict be unique

    Returns:
        dict: the merged COCO-formatted .json dict
    """

    # We are going to modify some of the inputs, so make copies first
    input_dicts = [deepcopy(d) for d in input_dicts]

    filename_to_image = {}
    all_annotations = []
    info = None

    category_name_to_category = {}

    known_fields = ['info', 'categories', 'annotations','images','filename_prefix']

    image_counts = []
    annotation_counts = []
    category_counts = []

    n_merged_categories = 0

    # i_input_dict = 0; input_dict = input_dicts[i_input_dict]
    for i_input_dict,input_dict in enumerate(input_dicts):

        image_counts.append(len(input_dict['images']))
        annotation_counts.append(len(input_dict['annotations']))
        category_counts.append(len(input_dict['categories']))

        filename_prefix = ''
        if ('filename_prefix' in input_dict.keys()):
            filename_prefix = input_dict['filename_prefix']

        for k in input_dict.keys():
            if k not in known_fields:
                raise ValueError(f'Unrecognized CCT field: {k}')

        # We will prepend an index to every ID to guarantee uniqueness
        index_string = 'ds' + str(i_input_dict).zfill(3) + '_'

        old_category_id_to_new_category_id = {}
        old_image_id_to_new_image_id = {}

        # Map detection categories from the original data set into the merged data set
        for original_category in input_dict['categories']:

            original_category_id = original_category['id']
            category_name = original_category['name']

            # If we've already created a new category for this name, use that ID
            if category_name in category_name_to_category:
                new_category_id = category_name_to_category[category_name]['id']
                n_merged_categories += 1
            else:
                new_category_id = len(category_name_to_category)
                new_category = deepcopy(original_category)
                new_category['id'] = new_category_id
                category_name_to_category[category_name] = new_category

            if original_category_id in old_category_id_to_new_category_id:
                assert old_category_id_to_new_category_id[original_category_id] == \
                    new_category_id, 'Category ID mismatch'
            else:
                old_category_id_to_new_category_id[original_category_id] = \
                    new_category_id

        # ...for each category


        # Merge original image list into the merged data set
        for im in input_dict['images']:

            if 'seq_id' in im:
                im['seq_id'] = index_string + str(im['seq_id'])
            if 'location' in im:
                im['location'] = index_string + im['location']

            im_file = filename_prefix + im['file_name']
            im['file_name'] = im_file

            # Create a unique ID
            im['id'] = index_string + str(im['id'])

            if im_file in filename_to_image:
                assert not require_uniqueness, f'Duplicate image: {im_file}'
                # Keep the first record we saw for this filename, and remember to point
                # this image's annotations at the record we're keeping
                print('Redundant image {}'.format(im_file))
                old_image_id_to_new_image_id[im['id']] = filename_to_image[im_file]['id']
            else:
                filename_to_image[im_file] = im

        # ...for each image


        # Same for annotations
        for ann in input_dict['annotations']:

            ann['image_id'] = index_string + str(ann['image_id'])
            if ann['image_id'] in old_image_id_to_new_image_id:
                ann['image_id'] = old_image_id_to_new_image_id[ann['image_id']]
            ann['id'] = index_string + str(ann['id'])
            assert ann['category_id'] in old_category_id_to_new_category_id
            ann['category_id'] = old_category_id_to_new_category_id[ann['category_id']]

        # ...for each annotation

        all_annotations.extend(input_dict['annotations'])

        # Merge info dicts, don't check completion time fields
        if info is None:
            info = deepcopy(input_dict['info'])
            info['original_info'] = [input_dict['info']]
        else:
            info['original_info'].append(input_dict['info'])

    # ...for each dictionary

    # Convert merged image dictionaries to a sorted list
    sorted_images = sorted(filename_to_image.values(), key=lambda im: im['file_name'])

    if require_uniqueness:
        assert len(sorted_images) == sum(image_counts), 'Image count mismatch'
    assert len(all_annotations) == sum(annotation_counts), 'Annotation count mismatch'

    # Every annotation should refer to an image that's still in the merged file.  We also
    # remove redundant annotations here; when the same image appears in multiple input
    # files, that image's annotations get merged onto a single image record, so we may
    # have annotations that are identical other than the "id" field.
    all_image_ids = set(im['id'] for im in sorted_images)
    assert len(all_image_ids) == len(sorted_images), 'Duplicate image IDs in merged output'

    deduplicated_annotations = []
    annotation_keys = set()
    n_redundant_annotations = 0

    for ann in all_annotations:

        assert ann['image_id'] in all_image_ids, \
            'Annotation {} refers to non-existent image {}'.format(
                ann['id'],ann['image_id'])

        # Compare every field except the annotation ID; sort_keys makes this
        # independent of field order, and default=str keeps us from choking on
        # anything unusual that made it into a custom field.
        annotation_key = json.dumps({k:v for k,v in ann.items() if k != 'id'},
                                    sort_keys=True, default=str)

        if annotation_key in annotation_keys:
            n_redundant_annotations += 1
        else:
            annotation_keys.add(annotation_key)
            deduplicated_annotations.append(ann)

    # ...for each annotation

    all_annotations = deduplicated_annotations

    all_categories = list(category_name_to_category.values())
    all_categories = sort_list_of_dicts_by_key(all_categories,'id')

    image_count_string = ','.join([str(n) for n in image_counts])
    annotation_count_string = ','.join([str(n) for n in annotation_counts])
    category_count_string = ', '.join([str(n) for n in category_counts])

    print('Merged file has {} images (by project: {})'.format(
        len(sorted_images),image_count_string))
    print('Merged file has {} annotations (by project: {})'.format(
        len(all_annotations),annotation_count_string))
    if n_redundant_annotations > 0:
        print('Removed {} redundant annotations'.format(n_redundant_annotations))
    print('Merged {} common categories ({})'.format(
        n_merged_categories,category_count_string))

    merged_dict = {'info': info,
                   'categories': all_categories,
                   'images': sorted_images,
                   'annotations': all_annotations}

    return merged_dict

# ...def combine_cct_dictionaries(...)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_paths', nargs='+',
        help='List of input .json files')
    parser.add_argument(
        'output_path',
        help='Output .json file')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    combine_cct_files(args.input_paths, args.output_path)

if __name__ == '__main__':
    main()
