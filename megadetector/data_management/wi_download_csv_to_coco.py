"""

wi_download_csv_to_coco.py

Converts a .csv file (or a folder of .csv files) from a Wildlife Insights project export to a
COCO camera traps .json file.

Currently assumes that common names are unique identifiers, which is convenient but unreliable.

"""

#%% Imports and constants

import os

from tqdm import tqdm
from collections import defaultdict

from megadetector.utils.ct_utils import is_empty
from megadetector.utils.ct_utils import write_json
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.path_utils import find_images
from megadetector.utils.wi_platform_utils import read_images_from_download_bundle
from megadetector.utils.wi_platform_utils import url_to_relative_path

wi_extra_annotation_columns = \
    ('identified_by',
     'wi_taxon_id',
     'class',
     'order',
     'family',
     'genus',
     'species',
     'uncertainty',
     'number_of_objects',
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

default_category_remappings = {
    # "blank" is handled specially below
    # 'blank':'empty',
    'homo species':'human',
    'human-camera trapper':'human',
    'no cv result':'unknown',
    'misfire':'blank'
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

    Args:
        csv_file_in (str): a downloaded .csv file we should convert to COCO, or a folder
            containing images...csv files.
        coco_file_out (str, optional): the .json file we should write; if [coco_file_out] is None,
            returns data, but doesn't write it
        image_folder (str, optional): the folder where images live, only relevant if
            [exclude_missing_images] is True
        exclude_missing_images (bool, optional): whether to exclude images not present
            in disk; if this is True, [image_folder] must be a valid folder
        image_flattening (str, optional): if 'none', relative paths will be stored
            as the entire URL for each image, other than gs://.  Can be 'guid' (just
            store [GUID].JPG) or 'deployment' (store as [deployment]/[GUID].JPG).
        verbose (bool, optional): enable additional debug console output
        category_remappings (dict, optional): str --> str dict that maps any number of
            WI category names to output category names; for example defaults to mapping
            "Homo Species" to "Human", but leaves 99.99% of categories unchanged.
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


    ##%% Create COCO dictionaries

    category_name_to_id = {}
    category_name_to_id['empty'] = 0
    category_name_to_count = defaultdict(int)

    image_id_to_image = {}
    image_id_to_annotations = defaultdict(list)

    print('Converting records to COCO...')

    # image_id = next(iter(image_id_to_image_records))
    for image_id in tqdm(image_id_to_image_records.keys(),
                         total=len(image_id_to_image_records)):

        image_records_this_id = image_id_to_image_records[image_id]

        # Remove None and NaN
        for record in image_records_this_id:
            for k in record:
                if is_empty(record[k]):
                    record[k] = ''

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

        im['wi_image_info'] = {}
        for s in wi_extra_image_columns:
            im['wi_image_info'][s] = str(record[s])

        categories_this_image = set()

        # record = image_records_this_id[0]
        for record in image_records_this_id:

            # If there are multiple records for this image, make sure the metadata
            # is consistent
            if record != reference_record:

                assert record['timestamp'] == reference_record['timestamp']
                assert record['project_id'] == reference_record['project_id']
                assert record['deployment_id'] == reference_record['deployment_id']

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

            if (category_remappings is not None) and (category_name in category_remappings):
                category_name = category_remappings[category_name]

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

            if category_name in category_name_to_id:
                category_id = category_name_to_id[category_name]
            else:
                category_id = len(category_name_to_id)
                category_name_to_id[category_name] = category_id

            if category_name != 'empty':
                nonblank_annotation_found = True

            ann = {}
            ann['image_id'] = image_id
            annotations_this_image = image_id_to_annotations[image_id]
            annotation_number = len(annotations_this_image)
            ann['id'] = image_id + '_' + str(annotation_number).zfill(2)
            ann['category_id'] = category_id
            annotations_this_image.append(ann)

            category_name_to_count[category_name] += 1

            extra_info = {}
            for s in wi_extra_annotation_columns:
                v = record[s]
                if isinstance(v,str):
                    if (len(v) > 0):
                        extra_info[s] = v
                elif isinstance(v,bool):
                    if v:
                        extra_info[s] = v

            if len(extra_info) > 0:
                ann['wi_extra_info'] = extra_info

        # ...for each record associated with this image

        if include_blanks or nonblank_annotation_found:
            image_id_to_image[image_id] = im

    # ...for each image

    images = list(image_id_to_image.values())
    categories = []
    for category_name in category_name_to_id:
        category_id = category_name_to_id[category_name]
        categories.append({'id':category_id,'name':category_name})
    annotations = []

    # image_id_to_annotations contains image IDs we didn't end up using,
    # so we loop over [images] to find the image IDs for which we want to
    # store annotations
    for im in images:
        image_id = im['id']
        annotations_this_image = image_id_to_annotations[image_id]
        for ann in annotations_this_image:
            annotations.append(ann)

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

        category_id_to_name = invert_dictionary(category_name_to_id)

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
