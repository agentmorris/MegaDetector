"""

wi_download_csv_to_coco.py

Converts a .csv file (or a folder of .csv files) from a Wildlife Insights project export to a
COCO camera traps .json file.

Currently assumes that common names are unique identifiers, which is convenient but unreliable.

"""

#%% Imports and constants

import os
import re

from tqdm import tqdm
from collections import defaultdict

from megadetector.utils.ct_utils import write_json
from megadetector.utils.ct_utils import is_empty
from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.utils.string_utils import is_int
from megadetector.utils.path_utils import find_images
from megadetector.utils.wi_platform_utils import read_images_from_download_bundle
from megadetector.utils.wi_platform_utils import read_sequences_from_download_bundle
from megadetector.utils.wi_platform_utils import url_to_relative_path

wi_extra_annotation_columns = \
    ('identified_by',
     'wi_taxon_id',
     'uncertainty',
     'number_of_objects',
     'group_size',
     'age',
     'sex',
     'animal_recognizable',
     'individual_id',
     'individual_animal_notes',
     'behavior',
     'highlighted',
     'markings')

# Omitted:
#
# is_blank
# filename
# cv_confidence
# license
# bounding_boxes

# Handled as part of the category:
#
# class
# order
# family
# genus
# species

# Handled as part of the image:
#
# timestamp
# image_id
# project_id
# deployment_id
# location

wi_extra_image_columns = ('project_id','deployment_id')

def _make_location_id(project_id,deployment_id):
    return 'project_' + str(project_id) + '_deployment_' + deployment_id

# The default category mappings lose some information about vehicles,
# but we are typically using this to perform accuracy comparisons for animals,
# and having lots of categories that co-occur with other categories (which
# vehicles do) makes analysis messier.
default_category_remappings = {
    # "blank" is handled specially below
    # 'blank':'empty',
    'homo species':'human',
    'no cv result':'unknown',
    'misfire':'blank',
    '.*human.*':'human',
    '.*vehicle.*':'vehicle',
    'truck':'vehicle',
    'atv':'vehicle'
}


#%% Main function

def wi_download_csv_to_coco(csv_file_in,
                            coco_file_out=None,
                            image_folder=None,
                            exclude_missing_images=False,
                            image_flattening='deployment',
                            verbose=True,
                            category_remappings=default_category_remappings,
                            blank_disagreement_handling='trust_label',
                            include_blanks=True):
    """
    Converts a .csv file (or folder of .csv files) from a Wildlife Insights project export
    to a COCO Camera Traps .json file.

    TODO: currently relies on uniqueness of common names, which is not guaranteed.  Prints
    warnings for non-unique common names.

    Args:
        csv_file_in (str): a downloaded .csv file we should convert to COCO, or a folder
            containing images...csv files.
        coco_file_out (str, optional): the .json file we should write; if [coco_file_out] is None,
            returns data, but doesn't write it
        image_folder (str, optional): the folder where images live, only relevant if
            [exclude_missing_images] is True
        exclude_missing_images (bool, optional): whether to exclude images not present
            in disk; if this is True, [image_folder] must be a valid folder.  This has no
            impact on blank images if "include_blanks" is False.
        image_flattening (str, optional): if 'none', relative paths will be stored
            as the entire URL for each image, other than gs://.  Can be 'guid' (just
            store [GUID].JPG) or 'deployment' (store as [deployment]/[GUID].JPG).
        verbose (bool, optional): enable additional debug console output
        category_remappings (dict, optional): str --> str dict that maps WI category
            names to output category names.  Regular expressions allowed in keys.
        blank_disagreement_handling (str, optional): what to do when the "common_name"
            field disagrees with the "is_blank" field; can be "trust_label" (default),
            "trust_is_blank", or "error
        include_blanks (bool, optional): whether to include blank images in the COCO
            file

    Returns:
        dict: COCO-formatted data, identical to what's written to [coco_file_out]
    """

    ##%% Validate inputs

    assert os.path.isfile(csv_file_in) or os.path.isdir(csv_file_in), \
        '{} does not exist'.format(csv_file_in)

    assert blank_disagreement_handling in ('trust_label','trust_is_blank','error'), \
        'Unknown blank disagreement handling value: {}'.format(
            blank_disagreement_handling)


    ##%% Read input files

    # read_images_from_download_bundle supports a folder or a single .csv file
    image_id_to_image_records = read_images_from_download_bundle(csv_file_in)

    assert image_id_to_image_records is not None, \
        'Failed to read images from {}'.format(csv_file_in)

    print('Read image records for {} unique image IDs'.format(
        len(image_id_to_image_records)))

    sequence_id_to_sequence_records = read_sequences_from_download_bundle(csv_file_in)

    sequence_id_to_image_ids = None

    # Is this a sequence-based project?
    if sequence_id_to_sequence_records is not None:

        print('Read sequence records for {} sequence IDs'.format(
            len(sequence_id_to_sequence_records)))

        # Group images into sequences

        sequence_id_to_image_ids = defaultdict(set)

        # image_id = next(iter(image_id_to_image_records))
        for image_id in image_id_to_image_records:

            records_this_image = image_id_to_image_records[image_id]

            for r in records_this_image:

                assert image_id == r['image_id']
                if 'sequence_id' not in r:
                    print('Warning: image {} does not have a sequence ID'.format(r['image_id']))
                    continue
                sequence_id_to_image_ids[r['sequence_id']].add(image_id)

            # ...for each record associated with this image ID

        # ...for each image ID

        # Create frame numbers and frame ordering

        # sequence_id = next(iter(sequence_id_to_image_ids))
        for sequence_id in sequence_id_to_image_ids:

            image_ids_this_sequence = sequence_id_to_image_ids[sequence_id]

            records_this_sequence = []

            for image_id in image_ids_this_sequence:

                records_this_image = image_id_to_image_records[image_id]
                # Choose a representative record for sorting
                r = records_this_image[0]
                # Timestamps are formatted as "2019-09-09 13:45:00"
                assert isinstance(r['timestamp'],str) and len(r['timestamp']) == 19
                records_this_sequence.append(r)

            sorted_records_this_sequence = \
                sort_list_of_dicts_by_key(records_this_sequence,'timestamp')

            # i_record = 0; r = sorted_records_this_sequence[i_record]
            for i_record,r in enumerate(sorted_records_this_sequence):

                r['frame_num'] = i_record
                r['seq_num_frames'] = len(sorted_records_this_sequence)
                image_id = r['image_id']

                # If there are multiple records for this image (typically indicating multiple
                # species), propagate that information to the other records
                records_this_image_id = image_id_to_image_records[image_id]

                # target_r = records_this_image_id[0]
                for target_r in records_this_image_id:

                    if r == target_r:
                        continue

                    assert r['timestamp'] == target_r['timestamp']
                    target_r['frame_num'] = i_record
                    target_r['seq_num_frames'] = len(sorted_records_this_sequence)

            # ...for each record in this sequence

        # ...for each sequence ID

    # ...if this is a sequence-based project


    #%% Create COCO dictionaries

    category_name_to_category = {}
    empty_category = {'name':'empty','id':0,'count':0,'taxonomy_string':''}
    category_name_to_category['empty'] = empty_category

    image_id_to_image = {}
    image_id_to_annotations = defaultdict(list)

    print('Converting records to COCO...')

    n_blanks_excluded = 0

    # image_id = next(iter(image_id_to_image_records))
    for image_id in tqdm(image_id_to_image_records.keys(),
                         total=len(image_id_to_image_records)):

        image_records_this_id = image_id_to_image_records[image_id]

        reference_record = image_records_this_id[0]

        url = reference_record['location']
        assert url.startswith('gs://')

        file_name = url_to_relative_path(url,image_flattening=image_flattening)

        location_id = _make_location_id(
            reference_record['project_id'],
            reference_record['deployment_id'])

        nonblank_annotation_found = False

        im = {}
        im['id'] = image_id
        im['file_name'] = file_name
        im['location'] = location_id
        im['datetime'] = reference_record['timestamp']

        sequence_records_this_sequence = None

        # Should we iterate over image records or sequence records to determine
        # labels for this image?
        label_records = image_records_this_id

        if 'sequence_id' in reference_record:

            assert sequence_id_to_image_ids is not None
            assert 'seq_num_frames' in reference_record, 'sequence processing error'
            assert 'frame_num' in reference_record, 'sequence processing error'

            # Not a typo; WI uses "sequence_id", COCO Camera Traps uses "seq_id"
            im['seq_id'] = reference_record['sequence_id']
            im['seq_num_frames'] = reference_record['seq_num_frames']
            im['frame_num'] = reference_record['frame_num']

            sequence_records_this_sequence = \
                sequence_id_to_sequence_records[reference_record['sequence_id']]
            label_records = sequence_records_this_sequence

            # Image-level and sequence-level taxa should be the same
            #
            # I don't know why labels are reported at both levels.
            taxon_ids_this_sequence = set([r['wi_taxon_id'] for r in sequence_records_this_sequence])
            taxon_ids_each_image = set([r['wi_taxon_id'] for r in image_records_this_id])

            assert taxon_ids_each_image == taxon_ids_this_sequence, \
                'Sequence label inconsistency'

        im['wi_image_info'] = {}
        for s in wi_extra_image_columns:
            assert s in reference_record, \
                'Required column {} missing from image {}'.format(s,reference_record['image_id'])
            im['wi_image_info'][s] = str(reference_record[s])

        categories_this_image = set()

        # Iterate over either image records or label records to determine the labels
        # we should store for this image.
        #
        # record = label_records[0]
        for record in label_records:

            # If there are multiple records for this image (typically because multiple species
            # were recorded), make sure the metadata is consistent across records
            if record != reference_record:

                # "Timestamp" is only present for image records; sequence records use
                # "start_time" and "end_time"
                # assert record['timestamp'] == reference_record['timestamp']
                assert record['project_id'] == reference_record['project_id']
                assert record['deployment_id'] == reference_record['deployment_id']

            count = None

            # This is a bit of future-proofing... it seems odd to me that "count"
            # becomes "number_of_objects" in image-based project downloads.
            if 'count' in record:
                raise ValueError(
                    'Note to self: you suspected a field called "count" might occur in some scenarios')

            # Image-based projects use "number_of_objects"
            if 'number_of_objects' in record:
                assert 'group_size' not in record
                count = record['number_of_objects']

            # Sequence-based projects use "group_size"
            if 'group_size' in record:
                assert 'number_of_objects' not in record
                count = record['group_size']

            if is_empty(count):
                count = None
            else:
                assert is_int(count), \
                    'Illegal group size value: {}'.format(count)
                count = int(count)

            category_name = record['common_name'].strip().lower()

            if len(category_name) == '':

                if len(record['genus']) > 0 and len(record['species']) > 0:
                    category_name = record['genus'] + ' ' + record['species']
                elif len(record['genus']) > 0:
                    category_name = record['genus']
                elif len(record['family']) > 0:
                    category_name = record['family']
                elif len(record['order']) > 0:
                    category_name = record['order']
                elif len(record['class']) > 0:
                    category_name = record['class']
                else:
                    print('Warning: no common name or binomial name available for {}'.format(
                        record['wi_taxon_id']))
                    category_name = record['wi_taxon_id']
                category_name = category_name.strip().lower()

            # ...handling empty category names

            taxonomy_tokens = []
            for level in ('class','order','family','genus','species'):
                taxonomy_tokens.append(record[level])
            taxonomy_string = ';'.join(taxonomy_tokens)
            taxonomy_string = taxonomy_string.lower().strip()

            # Should this category name get remapped?
            if (category_remappings is not None):
                # Check for exact matches
                if category_name in category_remappings:
                    category_name = category_remappings[category_name]
                # Check for regex matches
                else:
                    for k in category_remappings.keys():
                        if re.search(k,category_name):
                            category_name = category_remappings[k]
                            break

            # This is used for logic below, so we handle it outside of category_remappings
            if category_name == 'blank':
                category_name = 'empty'

            assert isinstance(record['is_blank'],int) and \
                record['is_blank'] in (0,1)

            # Resolve disagreements between different ways that blank-ness
            # can be represented
            category_says_blank = category_name == 'empty'
            is_blank_says_blank = record['is_blank'] == 1

            if (category_says_blank) and (is_blank_says_blank):
                category_name = 'empty'
            elif (category_says_blank) or (is_blank_says_blank):
                if blank_disagreement_handling == 'error':
                    raise ValueError('Blank disagreement for {} ({})'.format(
                        image_id, file_name))
                elif blank_disagreement_handling == 'trust_category':
                    print('Warning: category says {}, is_blank says {}, using category'.format(
                        category_name,record['is_blank']))
                elif blank_disagreement_handling == 'trust_is_blank':
                    print('Warning: category says {}, is_blank says {}, using is_blank'.format(
                        category_name,record['is_blank']))
                    if is_blank_says_blank:
                        category_name = 'empty'
                    else:
                        # This is a quirky case, we're supposed to trust is_blank, but
                        # and it says it's not blank, but the category says it is, so we
                        # have no other category we can use
                        assert category_name == 'empty'
                        category_name = 'unknown'

            assert category_name != 'blank'

            # Don't create annotations for the same category twice for the same image
            if category_name in categories_this_image:
                continue
            categories_this_image.add(category_name)

            if category_name in category_name_to_category:
                category = category_name_to_category[category_name]
                category_id = category['id']
                category['count'] = category['count'] + 1
                assert category['name'] == category_name
                if (category_name != 'empty') and \
                   (taxonomy_string != category['taxonomy_string']):
                    print('Warning: category {} has multiple taxonomy strings:\n{}\n{}\n'.format(
                        category_name,
                        taxonomy_string,
                        category['taxonomy_string']))
            else:
                category_id = len(category_name_to_category)
                category = {}
                category_name_to_category[category_name] = category
                category['name'] = category_name
                category['id'] = category_id
                category['count'] = 1
                category['taxonomy_string'] = taxonomy_string

            if category_name != 'empty':
                nonblank_annotation_found = True

            ann = {}
            ann['image_id'] = image_id
            annotations_this_image = image_id_to_annotations[image_id]
            annotation_number = len(annotations_this_image)
            ann['id'] = image_id + '_' + str(annotation_number).zfill(2)
            ann['category_id'] = category_id

            if sequence_records_this_sequence is not None:
                ann['sequence_level_annotation'] = True
            else:
                ann['sequence_level_annotation'] = False

            if count is not None:
                ann['count'] = count

            annotations_this_image.append(ann)

            extra_info = {}
            for s in wi_extra_annotation_columns:
                if s in record:
                    v = record[s]
                    # Don't store empty fields
                    if isinstance(v,str):
                        if (len(v) > 0):
                            extra_info[s] = v
                    # Treat bools as store_true, there are tons of uninformative "False"
                    # fields (e.g. "highlighted").
                    elif isinstance(v,bool):
                        if v:
                            extra_info[s] = v

            if len(extra_info) > 0:
                ann['wi_extra_info'] = extra_info

        # ...for each label record (image or sequence) associated with this image

        if include_blanks or nonblank_annotation_found:
            image_id_to_image[image_id] = im
        else:
            n_blanks_excluded += 1

    # ...for each image


    ##%% Write COCO output

    images = list(image_id_to_image.values())
    categories = list(category_name_to_category.values())

    print('Created COCO records for {} image IDs ({} blanks excluded)'.format(
        len(image_id_to_image),n_blanks_excluded))

    annotations = []

    # image_id_to_annotations contains image IDs we didn't end up using,
    # so we loop over [images] to find the image IDs for which we want to
    # store annotations
    for im in images:
        image_id = im['id']
        annotations_this_image = image_id_to_annotations[image_id]
        for ann in annotations_this_image:
            annotations.append(ann)

    print('Created COCO {} annotation records ({} categories)'.format(
        len(annotations),len(categories)))

    info = {'version':'1.00','description':'converted from WI export'}
    info['source_file'] = csv_file_in
    coco_data = {}
    coco_data['info'] = info
    coco_data['images'] = images
    coco_data['annotations'] = annotations
    coco_data['categories'] = categories

    print_category_counts = False

    if print_category_counts:

        print('Categories and counts:\n')

        category_name_to_count = {c['name']:c['count'] for c in categories}
        category_name_to_count = \
            sort_dictionary_by_value(category_name_to_count,reverse=True)

        for i_category,category_name in enumerate(category_name_to_count):
            category_name_string = category_name
            if (category_name == 'empty') and (not include_blanks):
                category_name_string += (' (excluded)')
            print('{}: {}'.format(category_name_string,
                                category_name_to_count[category_name]))

    ##%% Exclude missing images if requested

    if exclude_missing_images:

        assert os.path.isdir(image_folder), \
            'Must specify a valid image folder if you specify validate_images=True'

        print('Enumerating images in {}'.format(image_folder))
        all_images = find_images(image_folder, return_relative_paths=True, recursive=True)
        all_images_set = set(all_images)

        missing_images = []

        category_name_to_missing_image_count = defaultdict(int)

        category_id_to_name = {c['id']:c['name'] for c in categories}

        # im = images[0]
        for im in tqdm(images):

            file_name_relative = im['file_name']
            if file_name_relative not in all_images_set:

                annotations_this_image = image_id_to_annotations[im['id']]
                categories_this_image = []
                for ann in annotations_this_image:
                    category_id = ann['category_id']
                    category_name = category_id_to_name[category_id]
                    category_name_to_missing_image_count[category_name] += 1

                missing_images.append(im)

        print('Missing {} of {} images'.format(
            len(missing_images),
            len(images)))

        if len(category_name_to_missing_image_count) > 0:

            print('\nCategories with missing images:\n')

            category_name_to_missing_image_count = \
                sort_dictionary_by_value(category_name_to_missing_image_count,
                                         reverse=True)

            for category_name in category_name_to_missing_image_count:
                expected_count_string = ''
                if category_name in category_name_to_count:
                    expected_count_string = ' (of {} in metadata)'.format(
                        category_name_to_count[category_name])
                print('{}: {}{}'.format(category_name,
                                        category_name_to_missing_image_count[category_name],
                                        expected_count_string))

        # ...if we're missing any images

        # TODO: clean up categories that are no longer used
        missing_filenames = set([im['file_name'] for im in missing_images])
        missing_image_ids = set([im['id'] for im in missing_images])

        images = [im for im in images if im['file_name'] not in missing_filenames]
        annotations = [ann for ann in annotations if ann['image_id'] not in missing_image_ids]
        coco_data['images'] = images
        coco_data['annotations'] = annotations

    # ...if we are supposed to exclude missing images


    ##%% Write output json

    if coco_file_out is not None:
        print('Writing COCO data to {}'.format(coco_file_out))
        write_json(coco_file_out,coco_data)


    ##%% Validate output

    from megadetector.data_management.databases.integrity_check_json_db import \
        IntegrityCheckOptions,integrity_check_json_db

    print('Validating COCO file {}'.format(coco_file_out))

    options = IntegrityCheckOptions()
    options.baseDir = image_folder
    options.bCheckImageExistence = False
    options.verbose = verbose

    _ = integrity_check_json_db(coco_file_out,options)

    ##%%

    return coco_data

# ...def wi_download_csv_to_coco(...)


#%% Interactive driver

if False:

    #%%

    image_folder = '/blah/images/2000000'
    csv_file_in = '/csv_downloads/wildlife-insights_046ddddd-d870-dddd-a91d-a50c1a28fe29_project-2001650_data'
    coco_file_out = None
    gs_prefix = 'gs://000000000000_2000000_3658_project_name__main/deployment/'

    validate_images = False
    verbose = True
    category_remappings = default_category_remappings


#%% Command-line driver

# TODO
