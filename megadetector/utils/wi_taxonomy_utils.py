"""

wi_taxonomy_utils.py

Functions related to working with the SpeciesNet / Wildlife Insights taxonomy.

"""

#%% Imports and constants

import os
import json

import pandas as pd

from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from megadetector.utils.path_utils import \
    insert_before_extension, find_images

from megadetector.utils.ct_utils import (
    split_list_into_n_chunks,
    round_floats_in_nested_dict,
    is_list_sorted,
    invert_dictionary,
    sort_list_of_dicts_by_key,
    sort_dictionary_by_value,
)

from megadetector.postprocessing.validate_batch_results import \
    validate_batch_results, ValidateBatchResultsOptions

from megadetector.detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP

md_category_id_to_name = DEFAULT_DETECTOR_LABEL_MAP
md_category_name_to_id = invert_dictionary(md_category_id_to_name)

blank_prediction_string = \
    'f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank'
no_cv_result_prediction_string = \
    'f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result'
animal_prediction_string = \
    '1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal'
human_prediction_string = \
    '990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human'
vehicle_prediction_string = \
    'e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle'

non_taxonomic_prediction_strings = [blank_prediction_string,
                                    no_cv_result_prediction_string,
                                    animal_prediction_string,
                                    vehicle_prediction_string]

non_taxonomic_prediction_short_strings = [';'.join(s.split(';')[1:-1]) for s in \
                                          non_taxonomic_prediction_strings]

# Ignore some files when generating instances.json from a folder
default_tokens_to_ignore = ['$RECYCLE.BIN']


#%% Miscellaneous taxonomy support functions

def is_valid_prediction_string(s):
    """
    Determine whether [s] is a valid WI prediction string.  Prediction strings look like:

    '90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent'

    Args:
        s (str): the string to be tested for validity

    Returns:
        bool: True if this looks more or less like a WI prediction string
    """

    # Note to self... don't get tempted to remove spaces here; spaces are used
    # to indicate subspecies.
    return isinstance(s,str) and (len(s.split(';')) == 7) and (s == s.lower())


def is_valid_taxonomy_string(s):
    """
    Determine whether [s] is a valid 5-token WI taxonomy string.  Taxonomy strings
    look like:

    'mammalia;rodentia;;;;rodent'
    'mammalia;chordata;canidae;canis;lupus dingo'

    Args:
        s (str): the string to be tested for validity

    Returns:
        bool: True if this looks more or less like a WI taxonomy string
    """
    return isinstance(s,str) and (len(s.split(';')) == 5) and (s == s.lower())


def clean_taxonomy_string(s):
    """
    If [s] is a seven-token prediction string, trim the GUID and common name to produce
    a "clean" taxonomy string.  Else if [s] is a five-token string, return it.  Else error.

    Args:
        s (str): the seven- or five-token taxonomy/prediction string to clean

    Returns:
        str: the five-token taxonomy string
    """

    if is_valid_taxonomy_string(s):
        return s
    elif is_valid_prediction_string(s):
        tokens = s.split(';')
        assert len(tokens) == 7
        return ';'.join(tokens[1:-1])
    else:
        raise ValueError('Invalid taxonomy string')


taxonomy_level_names = \
    ['non-taxonomic','kingdom','phylum','class','order','family','genus','species','subspecies']


def taxonomy_level_to_string(k):
    """
    Maps taxonomy level indices (0 for kindgom, 1 for phylum, etc.) to strings.

    Args:
        k (int): taxonomy level index

    Returns:
        str: taxonomy level string
    """

    assert k >= 0 and k < len(taxonomy_level_names), \
        'Illegal taxonomy level index {}'.format(k)

    return taxonomy_level_names[k]


def taxonomy_level_string_to_index(s):
    """
    Maps strings ('kingdom', 'species', etc.) to level indices.

    Args:
        s (str): taxonomy level string

    Returns:
        int: taxonomy level index
    """

    assert s in taxonomy_level_names, 'Unrecognized taxonomy level string {}'.format(s)
    return taxonomy_level_names.index(s)


def taxonomy_level_index(s):
    """
    Returns the taxonomy level up to which [s] is defined (0 for non-taxnomic, 1 for kingdom,
    2 for phylum, etc.  Empty strings and non-taxonomic strings are treated as level 0.  1 and 2
    will never be returned; "animal" doesn't look like other taxonomic strings, so here we treat
    it as non-taxonomic.

    Args:
        s (str): 5-token or 7-token taxonomy string

    Returns:
        int: taxonomy level
    """

    if s in non_taxonomic_prediction_strings or s in non_taxonomic_prediction_short_strings:
        return 0

    tokens = s.split(';')
    assert len(tokens) in (5,7)

    if len(tokens) == 7:
        tokens = tokens[1:-1]

    # Anything without a class is considered non-taxonomic
    if len(tokens[0]) == 0:
        return 0

    # WI taxonomy strings start at class, so we'll never return 1 (kingdom) or 2 (phylum)
    elif len(tokens[1]) == 0:
        return 3
    elif len(tokens[2]) == 0:
        return 4
    elif len(tokens[3]) == 0:
        return 5
    elif len(tokens[4]) == 0:
        return 6
    # Subspecies are delimited with a space
    elif ' ' not in tokens[4]:
        return 7
    else:
        return 8


def is_taxonomic_prediction_string(s):
    """
    Determines whether [s] is a classification string that has taxonomic properties; this
    does not include, e.g., blanks/vehicles/no cv result.  It also excludes "animal".

    Args:
        s (str): a five- or seven-token taxonomic string

    Returns:
        bool: whether [s] is a taxonomic category
    """

    return (taxonomy_level_index(s) > 0)



def get_kingdom(prediction_string):
    """
    Return the kingdom field from a WI prediction string

    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format

    Returns:
        str: the kingdom field from the input string
    """
    tokens = prediction_string.split(';')
    assert is_valid_prediction_string(prediction_string)
    return tokens[1]


def is_human_classification(prediction_string):
    """
    Determines whether the input string represents a human classification, which includes a variety
    of common names (hiker, person, etc.)

    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format

    Returns:
        bool: whether this string corresponds to a human category
    """
    return prediction_string == human_prediction_string or 'homo;sapiens' in prediction_string


def is_vehicle_classification(prediction_string):
    """
    Determines whether the input string represents a vehicle classification.

    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format

    Returns:
        bool: whether this string corresponds to the vehicle category
    """
    return prediction_string == vehicle_prediction_string


def is_animal_classification(prediction_string):
    """
    Determines whether the input string represents an animal classification, which excludes, e.g.,
    humans, blanks, vehicles, unknowns

    Args:
        prediction_string (str): a string in the semicolon-delimited prediction string format

    Returns:
        bool: whether this string corresponds to an animal category
    """

    if prediction_string == animal_prediction_string:
        return True
    if prediction_string == human_prediction_string or 'homo;sapiens' in prediction_string:
        return False
    if prediction_string == blank_prediction_string:
        return False
    if prediction_string == no_cv_result_prediction_string:
        return False
    if len(get_kingdom(prediction_string)) == 0:
        return False
    return True


def taxonomy_info_to_taxonomy_string(taxonomy_info, include_taxon_id_and_common_name=False):
    """
    Convert a taxonomy record in dict format to a five- or seven-token semicolon-delimited string

    Args:
        taxonomy_info (dict): dict in the format stored in, e.g., taxonomy_string_to_taxonomy_info
        include_taxon_id_and_common_name (bool, optional): by default, this function returns a
            five-token string of latin names; if this argument is True, it includes the leading
            (GUID) and trailing (common name) tokens

    Returns:
        str: string in the format used as keys in, e.g., taxonomy_string_to_taxonomy_info
    """
    s = taxonomy_info['class'] + ';' + \
        taxonomy_info['order'] + ';'  + \
        taxonomy_info['family'] + ';'  + \
        taxonomy_info['genus'] + ';'  + \
        taxonomy_info['species']

    if include_taxon_id_and_common_name:
        s = taxonomy_info['taxon_id'] + ';' + s + ';' + taxonomy_info['common_name']

    return s


#%% Functions used to manipulate results files

def generate_whole_image_detections_for_classifications(classifications_json_file,
                                                        detections_json_file,
                                                        ensemble_json_file=None,
                                                        ignore_blank_classifications=True,
                                                        verbose=True):
    """
    Given a set of classification results in SpeciesNet format that were likely run on
    already-cropped images, generate a file of [fake] detections in SpeciesNet format in which each
    image is covered in a single whole-image detection.

    Args:
        classifications_json_file (str): SpeciesNet-formatted file containing classifications
        detections_json_file (str): SpeciesNet-formatted file to write with detections
        ensemble_json_file (str, optional): SpeciesNet-formatted file to write with detections
            and classfications
        ignore_blank_classifications (bool, optional): use non-top classifications when
            the top classification is "blank" or "no CV result"
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: the contents of [detections_json_file]
    """

    with open(classifications_json_file,'r') as f:
        classification_results = json.load(f)
    predictions = classification_results['predictions']

    output_predictions = []
    ensemble_predictions = []

    # i_prediction = 0; prediction = predictions[i_prediction]
    for i_prediction,prediction in enumerate(predictions):

        output_prediction = {}
        output_prediction['filepath'] = prediction['filepath']
        i_score = 0

        if ignore_blank_classifications:

            while (prediction['classifications']['classes'][i_score] in \
                   (blank_prediction_string,no_cv_result_prediction_string)):

                i_score += 1
                if (i_score >= len(prediction['classifications']['classes'])):

                    if verbose:

                        print('Ignoring blank classifications, but ' + \
                              'image {} has no non-blank values'.format(
                                i_prediction))

                    # Just use the first one
                    i_score = 0
                    break

                # ...if we passed the last prediction

            # ...iterate over classes within this prediction

        # ...if we're supposed to ignore blank classifications

        top_classification = prediction['classifications']['classes'][i_score]
        top_classification_score = prediction['classifications']['scores'][i_score]
        if is_animal_classification(top_classification):
            category_name = 'animal'
        elif is_human_classification(top_classification):
            category_name = 'human'
        else:
            category_name = 'vehicle'

        if category_name == 'human':
            md_category_name = 'person'
        else:
            md_category_name = category_name

        output_detection = {}
        output_detection['label'] = category_name
        output_detection['category'] = md_category_name_to_id[md_category_name]
        output_detection['conf'] = 1.0
        output_detection['bbox'] = [0.0, 0.0, 1.0, 1.0]
        output_prediction['detections'] = [output_detection]
        output_predictions.append(output_prediction)

        ensemble_prediction = {}
        ensemble_prediction['filepath'] = prediction['filepath']
        ensemble_prediction['detections'] = [output_detection]
        ensemble_prediction['prediction'] = top_classification
        ensemble_prediction['prediction_score'] = top_classification_score
        ensemble_prediction['prediction_source'] = 'fake_ensemble_file_utility'
        ensemble_prediction['classifications'] = prediction['classifications']
        ensemble_predictions.append(ensemble_prediction)

    # ...for each image

    ## Write output

    if ensemble_json_file is not None:

        ensemble_output_data = {'predictions':ensemble_predictions}
        with open(ensemble_json_file,'w') as f:
            json.dump(ensemble_output_data,f,indent=1)
        _ = validate_predictions_file(ensemble_json_file)

    output_data = {'predictions':output_predictions}
    with open(detections_json_file,'w') as f:
        json.dump(output_data,f,indent=1)
    return validate_predictions_file(detections_json_file)

# ...def generate_whole_image_detections_for_classifications(...)


def generate_md_results_from_predictions_json(predictions_json_file,
                                              md_results_file=None,
                                              base_folder=None,
                                              max_decimals=5,
                                              convert_human_to_person=True,
                                              convert_homo_species_to_human=True,
                                              verbose=False):
    """
    Generate an MD-formatted .json file from a predictions.json file, generated by the
    SpeciesNet ensemble.  Typically, MD results files use relative paths, and predictions.json
    files use absolute paths, so this function optionally removes the leading string
    [base_folder] from all file names.

    Currently just applies the top classification category to every detection.  If the top
    classification is "blank", writes an empty detection list.

    Uses the classification from the "prediction" field if it's available, otherwise
    uses the "classifications" field.

    When using the "prediction" field, records the top class in the "classifications" field to
    a field in each image called "top_classification_common_name".  This is often different
    from the value of  the "prediction" field.

    speciesnet_to_md.py is a command-line driver for this function.

    Args:
        predictions_json_file (str): path to a predictions.json file, or a dict
        md_results_file (str, optional): path to which we should write an MD-formatted .json file
        base_folder (str, optional): leading string to remove from each path in the
            predictions.json file
        max_decimals (int, optional): number of decimal places to which we should round
            all values
        convert_human_to_person (bool, optional): WI predictions.json files sometimes use the
            detection category "human"; MD files usually use "person".  If True, switches "human"
            to "person".
        convert_homo_species_to_human (bool, optional): the ensemble often rolls human predictions
            up to "homo species", which isn't wrong, but looks odd.  This forces these back to
            "homo sapiens".
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: results in MD format
    """

    # Read predictions file
    if isinstance(predictions_json_file,str):
        with open(predictions_json_file,'r') as f:
            predictions = json.load(f)
    else:
        assert isinstance(predictions_json_file,dict)
        predictions = predictions_json_file

    # Round floating-point values (confidence scores, coordinates) to a
    # reasonable number of decimal places
    if (max_decimals is not None) and (max_decimals > 0):
        round_floats_in_nested_dict(predictions, decimal_places=max_decimals)

    predictions = predictions['predictions']
    assert isinstance(predictions,list)

    # Convert backslashes to forward slashes in both filenames and the base folder string
    for im in predictions:
        im['filepath'] = im['filepath'].replace('\\','/')
    if base_folder is not None:
        base_folder = base_folder.replace('\\','/')

    detection_category_id_to_name = {}
    classification_category_name_to_id = {}

    # Keep track of detections that don't have an assigned detection category; these
    # are fake detections we create for non-blank images with non-empty detection lists.
    # We need to go back later and give them a legitimate detection category ID.
    all_unknown_detections = []

    # Create the output images list
    images_out = []

    base_folder_replacements = 0

    # im_in = predictions[0]
    for im_in in predictions:

        im_out = {}

        fn = im_in['filepath']
        if base_folder is not None:
            if fn.startswith(base_folder):
                base_folder_replacements += 1
                fn = fn.replace(base_folder,'',1)

        im_out['file'] = fn

        if 'failures' in im_in:

            im_out['failure'] = str(im_in['failures'])
            im_out['detections'] = None

        else:

            im_out['detections'] = []

            if 'detections' in im_in:

                if len(im_in['detections']) == 0:
                    im_out['detections'] = []
                else:
                    # det_in = im_in['detections'][0]
                    for det_in in im_in['detections']:
                        det_out = {}
                        if det_in['category'] in detection_category_id_to_name:
                            assert detection_category_id_to_name[det_in['category']] == det_in['label']
                        else:
                            detection_category_id_to_name[det_in['category']] = det_in['label']
                        det_out = {}
                        for s in ['category','conf','bbox']:
                            det_out[s] = det_in[s]
                        im_out['detections'].append(det_out)

            # ...if detections are present

            class_to_assign = None
            class_confidence = None
            top_classification_common_name = None

            if 'classifications' in im_in:

                classifications = im_in['classifications']
                assert len(classifications['scores']) == len(classifications['classes'])
                assert is_list_sorted(classifications['scores'],reverse=True)
                class_to_assign = classifications['classes'][0]
                class_confidence = classifications['scores'][0]

                tokens = class_to_assign.split(';')
                assert len(tokens) == 7
                top_classification_common_name = tokens[-1]
                if len(top_classification_common_name) == 0:
                    top_classification_common_name = 'undefined'

            if 'prediction' in im_in:

                class_to_assign = None
                im_out['top_classification_common_name'] = top_classification_common_name
                class_to_assign = im_in['prediction']
                if convert_homo_species_to_human and class_to_assign.endswith('homo species'):
                    class_to_assign = human_prediction_string
                class_confidence = im_in['prediction_score']

            if class_to_assign is not None:

                if class_to_assign == blank_prediction_string:

                    # This is a scenario that's not captured well by the MD format: a blank prediction
                    # with detections present.  But, for now, don't do anything special here, just making
                    # a note of this.
                    if len(im_out['detections']) > 0:
                        pass

                else:

                    assert not class_to_assign.endswith('blank')

                    # This is a scenario that's not captured well by the MD format: no detections present,
                    # but a non-blank prediction.  For now, create a fake detection to handle this prediction.
                    if len(im_out['detections']) == 0:

                        if verbose:
                            print('Warning: creating fake detection for non-blank whole-image classification' + \
                                  ' in {}'.format(im_in['file']))
                        det_out = {}
                        all_unknown_detections.append(det_out)

                        # We will change this to a string-int later
                        det_out['category'] = 'unknown'
                        det_out['conf'] = class_confidence
                        det_out['bbox'] = [0,0,1,1]
                        im_out['detections'].append(det_out)

                # ...if this is/isn't a blank classification

                # Attach that classification to each detection

                # Create a new category ID if necessary
                if class_to_assign in classification_category_name_to_id:
                    classification_category_id = classification_category_name_to_id[class_to_assign]
                else:
                    classification_category_id = str(len(classification_category_name_to_id))
                    classification_category_name_to_id[class_to_assign] = classification_category_id

                for det in im_out['detections']:
                    det['classifications'] = []
                    det['classifications'].append([classification_category_id,class_confidence])

            # ...if we have some type of classification for this image

        # ...if this is/isn't a failure

        images_out.append(im_out)

    # ...for each image

    if base_folder is not None:
        if base_folder_replacements == 0:
            print('Warning: you supplied {} as the base folder, but I made zero replacements'.format(
                base_folder))

    # Fix the 'unknown' category
    if len(all_unknown_detections) > 0:

        max_detection_category_id = max([int(x) for x in detection_category_id_to_name.keys()])
        unknown_category_id = str(max_detection_category_id + 1)
        detection_category_id_to_name[unknown_category_id] = 'unknown'

        for det in all_unknown_detections:
            assert det['category'] == 'unknown'
            det['category'] = unknown_category_id


    # Sort by filename

    images_out = sort_list_of_dicts_by_key(images_out,'file')

    # Prepare friendly classification names

    classification_category_descriptions = \
        invert_dictionary(classification_category_name_to_id)
    classification_categories_out = {}
    for category_id in classification_category_descriptions.keys():
        category_name = classification_category_descriptions[category_id].split(';')[-1]
        classification_categories_out[category_id] = category_name

    # Prepare the output dict

    detection_categories_out = detection_category_id_to_name
    info = {}
    info['format_version'] = 1.4
    info['detector'] = 'converted_from_predictions_json'

    if convert_human_to_person:
        for k in detection_categories_out.keys():
            if detection_categories_out[k] == 'human':
                detection_categories_out[k] = 'person'

    output_dict = {}
    output_dict['info'] = info
    output_dict['detection_categories'] = detection_categories_out
    output_dict['classification_categories'] = classification_categories_out
    output_dict['classification_category_descriptions'] = classification_category_descriptions
    output_dict['images'] = images_out

    if md_results_file is not None:
        with open(md_results_file,'w') as f:
            json.dump(output_dict,f,indent=1)

        validation_options = ValidateBatchResultsOptions()
        validation_options.raise_errors = True
        _ = validate_batch_results(md_results_file, options=validation_options)

    return output_dict

# ...def generate_md_results_from_predictions_json(...)


def generate_predictions_json_from_md_results(md_results_file,
                                              predictions_json_file,
                                              base_folder=None):
    """
    Generate a predictions.json file from the MD-formatted .json file [md_results_file].  Typically,
    MD results files use relative paths, and predictions.json files use absolute paths, so
    this function optionally prepends [base_folder].  Does not handle classification results in
    MD format, since this is intended to prepare data for passing through the WI classifier.

    md_to_wi.py is a command-line driver for this function.

    Args:
        md_results_file (str): path to an MD-formatted .json file
        predictions_json_file (str): path to which we should write a predictions.json file
        base_folder (str, optional): folder name to prepend to each path in md_results_file,
            to convert relative paths to absolute paths.
    """

    # Validate the input file
    validation_options = ValidateBatchResultsOptions()
    validation_options.raise_errors = True
    validation_options.return_data = True
    md_results = validate_batch_results(md_results_file, options=validation_options)
    category_id_to_name = md_results['detection_categories']

    output_dict = {}
    output_dict['predictions'] = []

    # im = md_results['images'][0]
    for im in md_results['images']:

        prediction = {}
        fn = im['file']
        if base_folder is not None:
            fn = os.path.join(base_folder,fn)
        fn = fn.replace('\\','/')
        prediction['filepath'] = fn
        if 'failure' in im and im['failure'] is not None:
            prediction['failures'] = ['DETECTOR']
        else:
            assert 'detections' in im and im['detections'] is not None
            detections = []
            for det in im['detections']:
                output_det = deepcopy(det)
                output_det['label'] = category_id_to_name[det['category']]
                detections.append(output_det)

            # detections *must* be sorted in descending order by confidence
            detections = sort_list_of_dicts_by_key(detections,'conf', reverse=True)
            prediction['detections'] = detections

        assert len(prediction.keys()) >= 2
        output_dict['predictions'].append(prediction)

    # ...for each image

    output_dir = os.path.dirname(predictions_json_file)
    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
    with open(predictions_json_file,'w') as f:
        json.dump(output_dict,f,indent=1)

# ...def generate_predictions_json_from_md_results(...)


def generate_instances_json_from_folder(folder,
                                        country=None,
                                        admin1_region=None,
                                        lat=None,
                                        lon=None,
                                        output_file=None,
                                        filename_replacements=None,
                                        tokens_to_ignore=default_tokens_to_ignore):
    """
    Generate an instances.json record that contains all images in [folder], optionally
    including location information, in a format suitable for run_model.py.  Optionally writes
    the results to [output_file].

    Args:
        folder (str): the folder to recursively search for images
        country (str, optional): a three-letter country code
        admin1_region (str, optional): an administrative region code, typically a two-letter
            US state code
        lat (float, optional): latitude to associate with all images
        lon (float, optional): longitude to associate with all images
        output_file (str, optional): .json file to which we should write instance records
        filename_replacements (dict, optional): str --> str dict indicating filename substrings
            that should be replaced with other strings.  Replacement occurs *after* converting
            backslashes to forward slashes.
        tokens_to_ignore (list, optional): ignore any images with these tokens in their
            names, typically used to avoid $RECYCLE.BIN.  Can be None.

    Returns:
        dict: dict with at least the field "instances"
    """

    assert os.path.isdir(folder)

    print('Enumerating images in {}'.format(folder))
    image_files_abs = find_images(folder,recursive=True,return_relative_paths=False)

    if tokens_to_ignore is not None:
        n_images_before_ignore_tokens = len(image_files_abs)
        for token in tokens_to_ignore:
            image_files_abs = [fn for fn in image_files_abs if token not in fn]
        print('After ignoring {} tokens, kept {} of {} images'.format(
            len(tokens_to_ignore),len(image_files_abs),n_images_before_ignore_tokens))

    instances = []

    # image_fn_abs = image_files_abs[0]
    for image_fn_abs in image_files_abs:
        instance = {}
        instance['filepath'] = image_fn_abs.replace('\\','/')
        if filename_replacements is not None:
            for s in filename_replacements:
                instance['filepath'] = instance['filepath'].replace(s,filename_replacements[s])
        if country is not None:
            instance['country'] = country
        if admin1_region is not None:
            instance['admin1_region'] = admin1_region
        if lat is not None:
            assert lon is not None, 'Latitude provided without longitude'
            instance['latitude'] = lat
        if lon is not None:
            assert lat is not None, 'Longitude provided without latitude'
            instance['longitude'] = lon
        instances.append(instance)

    to_return = {'instances':instances}

    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if len(output_dir) > 0:
            os.makedirs(output_dir,exist_ok=True)
        with open(output_file,'w') as f:
            json.dump(to_return,f,indent=1)

    return to_return

# ...def generate_instances_json_from_folder(...)


def split_instances_into_n_batches(instances_json,n_batches,output_files=None):
    """
    Given an instances.json file, split it into batches of equal size.

    Args:
        instances_json (str): input .json file in
        n_batches (int): number of new files to generate
        output_files (list, optional): output .json files for each
            batch.  If supplied, should have length [n_batches].  If not
            supplied, filenames will be generated based on [instances_json].

    Returns:
        list: list of output files that were written; identical to [output_files]
        if it was supplied as input.
    """

    with open(instances_json,'r') as f:
        instances = json.load(f)
    assert isinstance(instances,dict) and 'instances' in instances
    instances = instances['instances']

    if output_files is not None:
        assert len(output_files) == n_batches, \
            'Expected {} output files, received {}'.format(
                n_batches,len(output_files))
    else:
        output_files = []
        for i_batch in range(0,n_batches):
            batch_string = 'batch_{}'.format(str(i_batch).zfill(3))
            output_files.append(insert_before_extension(instances_json,batch_string))

    batches = split_list_into_n_chunks(instances, n_batches)

    for i_batch,batch in enumerate(batches):
        batch_dict = {'instances':batch}
        with open(output_files[i_batch],'w') as f:
            json.dump(batch_dict,f,indent=1)

    print('Wrote {} batches to file'.format(n_batches))

    return output_files

# ...def split_instances_into_n_batches(...)


def merge_prediction_json_files(input_prediction_files,output_prediction_file):
    """
    Merge all predictions.json files in [files] into a single .json file.

    Args:
        input_prediction_files (list): list of predictions.json files to merge
        output_prediction_file (str): output .json file
    """

    predictions = []
    image_filenames_processed = set()

    # input_json_fn = input_prediction_files[0]
    for input_json_fn in tqdm(input_prediction_files):

        assert os.path.isfile(input_json_fn), \
            'Could not find prediction file {}'.format(input_json_fn)
        with open(input_json_fn,'r') as f:
            results_this_file = json.load(f)
        assert isinstance(results_this_file,dict)
        predictions_this_file = results_this_file['predictions']
        for prediction in predictions_this_file:
            image_fn = prediction['filepath']
            assert image_fn not in image_filenames_processed
        predictions.extend(predictions_this_file)

    output_dict = {'predictions':predictions}

    output_dir = os.path.dirname(output_prediction_file)
    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
    with open(output_prediction_file,'w') as f:
        json.dump(output_dict,f,indent=1)

# ...def merge_prediction_json_files(...)


def load_md_or_speciesnet_file(fn,verbose=True):
    """
    Load a .json file that may be in MD or SpeciesNet format.  Typically used so
    SpeciesNet files can be supplied to functions originally written to support MD
    format.

    Args:
        fn (str): a .json file in predictions.json (MD or SpeciesNet) format
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: the contents of [fn], in MD format.
    """

    with open(fn,'r') as f:
        detector_output = json.load(f)

    # If this is a SpeicesNet file, convert to MD format
    if 'predictions' in detector_output:

        if verbose:
            print('This appears to be a SpeciesNet output file, converting to MD format')
        detector_output = generate_md_results_from_predictions_json(predictions_json_file=fn,
                                                                    md_results_file=None,
                                                                    base_folder=None)

    # ...if this is a SpeciesNet file

    assert 'images' in detector_output, \
        'Detector output file should be a json file with an "images" field.'

    return detector_output

# ...def load_md_or_speciesnet_file(...)


def validate_predictions_file(fn,instances=None,verbose=True):
    """
    Validate the predictions.json file [fn].

    Args:
        fn (str): a .json file in predictions.json (SpeciesNet) format
        instances (str or list, optional): a folder, instances.json file,
            or dict loaded from an instances.json file.  If supplied, this
            function will verify that [fn] contains the same number of
            images as [instances].
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: the contents of [fn]
    """

    with open(fn,'r') as f:
        d = json.load(f)
    predictions = d['predictions']

    failures = []

    for im in predictions:
        if 'failures' in im:
            failures.append(im)

    if verbose:
        print('Read predictions for {} images, with {} failure(s)'.format(
            len(d['predictions']),len(failures)))

    if instances is not None:

        if isinstance(instances,str):
            if os.path.isdir(instances):
                instances = generate_instances_json_from_folder(folder=instances)
            elif os.path.isfile(instances):
                with open(instances,'r') as f:
                    instances = json.load(f)
            else:
                raise ValueError('Could not find instances file/folder {}'.format(
                    instances))
        assert isinstance(instances,dict)
        assert 'instances' in instances
        instances = instances['instances']
        if verbose:
            print('Expected results for {} files'.format(len(instances)))
        assert len(instances) == len(predictions), \
            '{} instances expected, {} found'.format(
                len(instances),len(predictions))

        expected_files = set([instance['filepath'] for instance in instances])
        found_files = set([prediction['filepath'] for prediction in predictions])
        assert expected_files == found_files

    # ...if a list of instances was supplied

    return d

# ...def validate_predictions_file(...)


#%% Functions related to geofencing

def find_geofence_adjustments(ensemble_json_file,use_latin_names=False):
    """
    Count the number of instances of each unique change made by the geofence.

    Args:
        ensemble_json_file (str): SpeciesNet-formatted .json file produced
            by the full ensemble.
        use_latin_names (bool, optional): return a mapping using binomial names
            rather than common names.

    Returns:
        dict: maps strings that look like "puma,felidae family" to integers,
            where that entry would indicate the number of times that "puma" was
            predicted, but mapped to family level by the geofence.  Sorted in
            descending order by count.
    """

    # Load and validate ensemble results
    ensemble_results = validate_predictions_file(ensemble_json_file)

    assert isinstance(ensemble_results,dict)
    predictions = ensemble_results['predictions']

    # Maps comma-separated pairs of common names (or binomial names) to
    # the number of times that transition (first --> second) happened
    rollup_pair_to_count = defaultdict(int)

    # prediction = predictions[0]
    for prediction in tqdm(predictions):

        if 'failures' in prediction and \
            prediction['failures'] is not None and \
            len(prediction['failures']) > 0:
                continue

        assert 'prediction_source' in prediction, \
            'Prediction present without [prediction_source] field, are you sure this ' + \
            'is an ensemble output file?'

        if 'geofence' in prediction['prediction_source']:

            classification_taxonomy_string = \
                prediction['classifications']['classes'][0]
            prediction_taxonomy_string = prediction['prediction']
            assert is_valid_prediction_string(classification_taxonomy_string)
            assert is_valid_prediction_string(prediction_taxonomy_string)

            # Typical examples:
            # '86f5b978-4f30-40cc-bd08-be9e3fba27a0;mammalia;rodentia;sciuridae;sciurus;carolinensis;eastern gray squirrel'
            # 'e4d1e892-0e4b-475a-a8ac-b5c3502e0d55;mammalia;rodentia;sciuridae;;;sciuridae family'
            classification_common_name = classification_taxonomy_string.split(';')[-1]
            prediction_common_name = prediction_taxonomy_string.split(';')[-1]
            classification_binomial_name = classification_taxonomy_string.split(';')[-2]
            prediction_binomial_name = prediction_taxonomy_string.split(';')[-2]

            input_name = classification_binomial_name if use_latin_names else \
                classification_common_name
            output_name = prediction_binomial_name if use_latin_names else \
                prediction_common_name

            rollup_pair = input_name.strip() + ',' + output_name.strip()
            rollup_pair_to_count[rollup_pair] += 1

        # ...if we made a geofencing change

    # ...for each prediction

    rollup_pair_to_count = sort_dictionary_by_value(rollup_pair_to_count,reverse=True)

    return rollup_pair_to_count

# ...def find_geofence_adjustments(...)


def generate_geofence_adjustment_html_summary(rollup_pair_to_count,min_count=10):
    """
    Given a list of geofence rollups, likely generated by find_geofence_adjustments,
    generate an HTML summary of the changes made by geofencing.  The resulting HTML
    is wrapped in <div>, but not, for example, in <html> or <body>.

    Args:
        rollup_pair_to_count (dict): list of changes made by geofencing, see
            find_geofence_adjustments for details
        min_count (int, optional): minimum number of changes a pair needs in order
            to be included in the report.
    """

    geofence_footer = ''

    # Restrict to the list of taxa that were impacted by geofencing
    rollup_pair_to_count = \
        {key: value for key, value in rollup_pair_to_count.items() if value >= min_count}

    # rollup_pair_to_count is sorted in descending order by count
    assert is_list_sorted(list(rollup_pair_to_count.values()),reverse=True)

    if len(rollup_pair_to_count) > 0:

        geofence_footer = \
            '<h3>Geofence changes that occurred more than {} times</h3>\n'.format(min_count)
        geofence_footer += '<div class="contentdiv">\n'

        print('\nRollup changes with count > {}:'.format(min_count))
        for rollup_pair in rollup_pair_to_count.keys():
            count = rollup_pair_to_count[rollup_pair]
            rollup_pair_s = rollup_pair.replace(',',' --> ')
            print('{}: {}'.format(rollup_pair_s,count))
            rollup_pair_html = rollup_pair.replace(',',' &rarr; ')
            geofence_footer += '{} ({})<br/>\n'.format(rollup_pair_html,count)

        geofence_footer += '</div>\n'

    return geofence_footer

# ...def generate_geofence_adjustment_html_summary(...)


#%% TaxonomyHandler class

class TaxonomyHandler:
    """
    Handler for taxonomy mapping and geofencing operations.
    """

    def __init__(self, taxonomy_file, geofencing_file, country_code_file):
        """
        Initialize TaxonomyHandler with taxonomy information.

        Args:
            taxonomy_file (str): .csv file containing the SpeciesNet (or WI) taxonomy,
                as seven-token taxonomic specifiers.  Distributed with the SpeciesNet model.
            geofencing_file (str): .json file containing the SpeciesNet geofencing rules.
                Distributed with the SpeciesNet model.
            country_code_file: .csv file mapping country codes to names.  Should include columns
                called "name" and "alpha-3".  A compatible file is available at
                https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes
        """

        #: Maps a taxonomy string (e.g. mammalia;cetartiodactyla;cervidae;odocoileus;virginianus) to
        #: a dict with keys taxon_id, common_name, kingdom, phylum, class, order, family, genus, species
        self.taxonomy_string_to_taxonomy_info = None

        #: Maps a binomial name (one, two, or three ws-delimited tokens) to the same dict described above.
        self.binomial_name_to_taxonomy_info = None

        #: Maps a common name to the same dict described above
        self.common_name_to_taxonomy_info = None

        #: Dict mapping 5-token semicolon-delimited taxonomy strings to geofencing rules
        self.taxonomy_string_to_geofencing_rules = None

        #: Maps lower-case country names to upper-case country codes
        self.country_to_country_code = None

        #: Maps upper-case country codes to lower-case country names
        self.country_code_to_country = None

        self._load_taxonomy_info(taxonomy_file=taxonomy_file)
        self._initialize_geofencing(geofencing_file=geofencing_file,
                                    country_code_file=country_code_file)


    def _load_taxonomy_info(self, taxonomy_file):
        """
        Load WI/SpeciesNet taxonomy information from a .csv file.  Stores information in the
        instance dicts [taxonomy_string_to_taxonomy_info], [binomial_name_to_taxonomy_info],
        and [common_name_to_taxonomy_info].

        Args:
            taxonomy_file (str): .csv file containing the SpeciesNet (or WI) taxonomy,
                as seven-token taxonomic specifiers.  Distributed with the SpeciesNet model.
        """

        """
        Taxonomy keys are five-token taxonomy strings, e.g.:

        'mammalia;cetartiodactyla;cervidae;odocoileus;virginianus'

        Taxonomy values are seven-token strings w/Taxon IDs and common names, e.g.:

        '5c7ce479-8a45-40b3-ae21-7c97dfae22f5;mammalia;cetartiodactyla;cervidae;odocoileus;virginianus;white-tailed deer'
        """

        with open(taxonomy_file,'r') as f:
            taxonomy_lines = f.readlines()
        taxonomy_lines = [s.strip() for s in taxonomy_lines]

        self.taxonomy_string_to_taxonomy_info = {}
        self.binomial_name_to_taxonomy_info = {}
        self.common_name_to_taxonomy_info = {}

        five_token_string_to_seven_token_string = {}

        for line in taxonomy_lines:
            tokens = line.split(';')
            assert len(tokens) == 7, 'Illegal line {} in taxonomy file {}'.format(
                line,taxonomy_file)
            five_token_string = ';'.join(tokens[1:-1])
            assert len(five_token_string.split(';')) == 5
            five_token_string_to_seven_token_string[five_token_string] = line

        for taxonomy_string in five_token_string_to_seven_token_string.keys():

            taxonomy_string = taxonomy_string.lower()

            taxon_info = {}
            extended_string = five_token_string_to_seven_token_string[taxonomy_string]
            tokens = extended_string.split(';')
            assert len(tokens) == 7
            taxon_info['taxon_id'] = tokens[0]
            assert len(taxon_info['taxon_id']) == 36
            taxon_info['kingdom'] = 'animal'
            taxon_info['phylum'] = 'chordata'
            taxon_info['class'] = tokens[1]
            taxon_info['order'] = tokens[2]
            taxon_info['family'] = tokens[3]
            taxon_info['genus'] = tokens[4]
            taxon_info['species'] = tokens[5]
            taxon_info['common_name'] = tokens[6]

            if taxon_info['common_name'] != '':
                self.common_name_to_taxonomy_info[taxon_info['common_name']] = taxon_info

            self.taxonomy_string_to_taxonomy_info[taxonomy_string] = taxon_info

            binomial_name = None
            if len(tokens[4]) > 0 and len(tokens[5]) > 0:
                # strip(), but don't remove spaces from the species name;
                # subspecies are separated with a space, e.g. canis;lupus dingo
                binomial_name = tokens[4].strip() + ' ' + tokens[5].strip()
            elif len(tokens[4]) > 0:
                binomial_name = tokens[4].strip()
            elif len(tokens[3]) > 0:
                binomial_name = tokens[3].strip()
            elif len(tokens[2]) > 0:
                binomial_name = tokens[2].strip()
            elif len(tokens[1]) > 0:
                binomial_name = tokens[1].strip()
            if binomial_name is None:
                # print('Warning: no binomial name for {}'.format(taxonomy_string))
                pass
            else:
                self.binomial_name_to_taxonomy_info[binomial_name] = taxon_info

            taxon_info['binomial_name'] = binomial_name

        # ...for each taxonomy string in the file

        print('Created {} records in taxonomy_string_to_taxonomy_info'.format(len(self.taxonomy_string_to_taxonomy_info)))
        print('Created {} records in common_name_to_taxonomy_info'.format(len(self.common_name_to_taxonomy_info)))

    # ...def _load_taxonomy_info(...)


    def _initialize_geofencing(self, geofencing_file, country_code_file):
        """
        Load geofencing information from a .json file, and country code mappings from
        a .csv file.  Stores results in the instance tables [taxonomy_string_to_geofencing_rules],
        [country_to_country_code], and [country_code_to_country].

        Args:
            geofencing_file (str): .json file with geofencing rules
            country_code_file (str): .csv file with country code mappings, in columns
                called "name" and "alpha-3", e.g. from
                https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
        """

        # Read country code information
        country_code_df = pd.read_csv(country_code_file)
        self.country_to_country_code = {}
        self.country_code_to_country = {}
        for i_row,row in country_code_df.iterrows():
            self.country_to_country_code[row['name'].lower()] = row['alpha-3'].upper()
            self.country_code_to_country[row['alpha-3'].upper()] = row['name'].lower()

        # Read geofencing information
        with open(geofencing_file,'r',encoding='utf-8') as f:
            self.taxonomy_string_to_geofencing_rules = json.load(f)

        """
        Geofencing keys are five-token taxonomy strings, e.g.:

        'mammalia;cetartiodactyla;cervidae;odocoileus;virginianus'

        Geofencing values are tables mapping allow/block to country codes, optionally including region/state codes, e.g.:

        {'allow': {
          'ALA': [],
          'ARG': [],
          ...
          'SUR': [],
          'TTO': [],
          'USA': ['AL',
           'AR',
           'AZ',
           ...
        }
        """

        # Validate

        # species_string = next(iter(taxonomy_string_to_geofencing_rules.keys()))
        for species_string in self.taxonomy_string_to_geofencing_rules.keys():

            species_rules = self.taxonomy_string_to_geofencing_rules[species_string]

            if len(species_rules.keys()) > 1:
                print('Warning: taxon {} has both allow and block rules'.format(species_string))

            for rule_type in species_rules.keys():

                assert rule_type in ('allow','block')
                all_country_rules_this_species = species_rules[rule_type]

                for country_code in all_country_rules_this_species.keys():

                    assert country_code in self.country_code_to_country
                    region_rules = all_country_rules_this_species[country_code]
                    # Right now we only have regional rules for the USA; these may be part of
                    # allow or block rules.
                    if len(region_rules) > 0:
                        assert country_code == 'USA'

                # ...for each country code in this rule set

            # ...for each rule set for this species

        # ...for each species

    # ...def _initialize_geofencing(...)


    def _parse_region_code_list(self, codes):
        """
        Turn a list of country or state codes in string, delimited string, or list format
        into a list.  Also does basic validity checking.
        """

        if not isinstance(codes,list):

            assert isinstance(codes,str)

            codes = codes.strip()

            # This is just a single codes
            if ',' not in codes:
                codes = [codes]
            else:
                codes = codes.split(',')
            codes = [c.strip() for c in codes]

        assert isinstance(codes,list)

        codes = [c.upper().strip() for c in codes]

        for c in codes:
            assert len(c) in (2,3)

        return codes

    # ...def _parse_region_code_list(...)


    def generate_csv_rows_to_block_all_countries_except(self, species_string, block_except_list):
        """
        Generate rows in the format expected by geofence_fixes.csv, representing a list of
        allow and block rules to block all countries currently allowed for this species
        except [allow_countries], and add allow rules these countries.

        Args:
            species_string (str): five-token taxonomy string
            block_except_list (list): list of country codes not to block

        Returns:
            list of str: strings compatible with geofence_fixes.csv
        """

        assert is_valid_taxonomy_string(species_string), \
            '{} is not a valid taxonomy string'.format(species_string)

        assert self.taxonomy_string_to_geofencing_rules is not None, \
            'Initialize geofencing prior to species lookup'
        assert self.taxonomy_string_to_taxonomy_info is not None, \
            'Initialize taxonomy lookup prior to species lookup'

        geofencing_rules_this_species = \
            self.taxonomy_string_to_geofencing_rules[species_string]

        allowed_countries = []
        if 'allow' in geofencing_rules_this_species:
            allowed_countries.extend(geofencing_rules_this_species['allow'])

        blocked_countries = []
        if 'block' in geofencing_rules_this_species:
            blocked_countries.extend(geofencing_rules_this_species['block'])

        block_except_list = self._parse_region_code_list(block_except_list)

        countries_to_block = []
        countries_to_allow = []

        # country = allowed_countries[0]
        for country in allowed_countries:
            if country not in block_except_list and country not in blocked_countries:
                countries_to_block.append(country)

        for country in block_except_list:
            if country in blocked_countries:
                raise ValueError("I can't allow a country that has already been blocked")
            if country not in allowed_countries:
                countries_to_allow.append(country)

        rows = self.generate_csv_rows_for_species(species_string,
                                             allow_countries=countries_to_allow,
                                             block_countries=countries_to_block)

        return rows

    # ...def generate_csv_rows_to_block_all_countries_except(...)


    def generate_csv_rows_for_species(self, species_string,
                                      allow_countries=None,
                                      block_countries=None,
                                      allow_states=None,
                                      block_states=None):
        """
        Generate rows in the format expected by geofence_fixes.csv, representing a list of
        allow and/or block rules for the specified species and countries/states.  Does not check
        that the rules make sense; e.g. nothing will stop you in this function from both allowing
        and blocking a country.

        Args:
            species_string (str): five-token string in semicolon-delimited WI taxonomy format
            allow_countries (list or str, optional): three-letter country codes, list of
                country codes, or comma-separated list of country codes to allow
            block_countries (list or str, optional): three-letter country codes, list of
                country codes, or comma-separated list of country codes to block
            allow_states (list or str, optional): two-letter state codes, list of
                state codes, or comma-separated list of state codes to allow
            block_states (list or str, optional): two-letter state code, list of
                state codes, or comma-separated list of state codes to block

        Returns:
            list of str: lines ready to be pasted into geofence_fixes.csv
        """

        assert is_valid_taxonomy_string(species_string), \
            '{} is not a valid taxonomy string'.format(species_string)

        lines = []

        if allow_countries is not None:
            allow_countries = self._parse_region_code_list(allow_countries)
            for country in allow_countries:
                lines.append(species_string + ',allow,' + country + ',')

        if block_countries is not None:
            block_countries = self._parse_region_code_list(block_countries)
            for country in block_countries:
                lines.append(species_string + ',block,' + country + ',')

        if allow_states is not None:
            allow_states = self._parse_region_code_list(allow_states)
            for state in allow_states:
                lines.append(species_string + ',allow,USA,' + state)

        if block_states is not None:
            block_states = self._parse_region_code_list(block_states)
            for state in block_states:
                lines.append(species_string + ',block,USA,' + state)

        return lines

    # ...def generate_csv_rows_for_species(...)


    def species_string_to_canonical_species_string(self, species):
        """
        Convert a string that may be a 5-token species string, a binomial name,
        or a common name into a 5-token species string, using taxonomic lookup.

        Args:
            species (str): 5-token species string, binomial name, or common name

        Returns:
            str: common name

        Raises:
            ValueError: if [species] is not in our dictionary
        """

        species = species.lower().strip()

        # Turn "species" into a taxonomy string

        # If this is already a taxonomy string...
        if len(species.split(';')) == 5:
            taxonomy_string = species
        # If this is a common name...
        elif species in self.common_name_to_taxonomy_info:
            taxonomy_info = self.common_name_to_taxonomy_info[species]
            taxonomy_string = taxonomy_info_to_taxonomy_string(taxonomy_info)
        # If this is a binomial name...
        elif (species in self.binomial_name_to_taxonomy_info):
            taxonomy_info = self.binomial_name_to_taxonomy_info[species]
            taxonomy_string = taxonomy_info_to_taxonomy_string(taxonomy_info)
        else:
            raise ValueError('Could not find taxonomic information for {}'.format(species))

        return taxonomy_string

    # ...def species_string_to_canonical_species_string(...)


    def species_string_to_taxonomy_info(self,species):
        """
        Convert a string that may be a 5-token species string, a binomial name,
        or a common name into a taxonomic info dictionary, using taxonomic lookup.

        Args:
            species (str): 5-token species string, binomial name, or common name

        Returns:
            dict: taxonomy information

        Raises:
            ValueError: if [species] is not in our dictionary
        """

        species = species.lower().strip()
        canonical_string = self.species_string_to_canonical_species_string(species)
        return self.taxonomy_string_to_taxonomy_info[canonical_string]


    def species_allowed_in_country(self, species, country, state=None, return_status=False):
        """
        Determines whether [species] is allowed in [country], according to
        already-initialized geofencing rules.

        Args:
            species (str): can be a common name, a binomial name, or a species string
            country (str): country name or three-letter code
            state (str, optional): two-letter US state code
            return_status (bool, optional): by default, this function returns a bool;
                if you want to know *why* [species] is allowed/not allowed, settings
                return_status to True will return additional information.

        Returns:
            bool or str: typically returns True if [species] is allowed in [country], else
            False.  Returns a more detailed string if return_status is set.
        """

        assert self.taxonomy_string_to_geofencing_rules is not None, \
            'Initialize geofencing prior to species lookup'
        assert self.taxonomy_string_to_taxonomy_info is not None, \
            'Initialize taxonomy lookup prior to species lookup'

        taxonomy_string = self.species_string_to_canonical_species_string(species)

        # Normalize [state]

        if state is not None:
            state = state.upper()
            assert len(state) == 2

        # Turn "country" into a country code

        if len(country) == 3:
            assert country.upper() in self.country_code_to_country
            country = country.upper()
        else:
            assert country.lower() in self.country_to_country_code
            country = self.country_to_country_code[country.lower()]

        country_code = country.upper()

        # Species with no rules are allowed everywhere
        if taxonomy_string not in self.taxonomy_string_to_geofencing_rules:
            status = 'allow_by_default'
            if return_status:
                return status
            else:
                return True

        geofencing_rules_this_species = self.taxonomy_string_to_geofencing_rules[taxonomy_string]
        allowed_countries = []
        blocked_countries = []

        rule_types_this_species = list(geofencing_rules_this_species.keys())
        for rule_type in rule_types_this_species:
            assert rule_type in ('allow','block')

        if 'block' in rule_types_this_species:
            blocked_countries = list(geofencing_rules_this_species['block'])
        if 'allow' in rule_types_this_species:
            allowed_countries = list(geofencing_rules_this_species['allow'])

        status = None

        # The convention is that block rules win over allow rules
        if country_code in blocked_countries:
            if country_code in allowed_countries:
                status = 'blocked_over_allow'
            else:
                status = 'blocked'
        elif country_code in allowed_countries:
            status = 'allowed'
        elif len(allowed_countries) > 0:
            # The convention is that if allow rules exist, any country not on that list
            # is blocked.
            status = 'block_not_on_country_allow_list'
        else:
            # Only block rules exist for this species, and they don't include this country
            assert len(blocked_countries) > 0
            status = 'allow_not_on_block_list'

        # Now let's see whether we have to deal with any regional rules.
        #
        # Right now regional rules only exist for the US.
        if (country_code == 'USA') and ('USA' in geofencing_rules_this_species[rule_type]):

            if state is None:

                state_list = geofencing_rules_this_species[rule_type][country_code]
                if len(state_list) > 0:
                    assert status.startswith('allow')
                    status = 'allow_no_state'

            else:

                state_list = geofencing_rules_this_species[rule_type][country_code]

                if state in state_list:
                    # If the state is on the list, do what the list says
                    if rule_type == 'allow':
                        status = 'allow_on_state_allow_list'
                    else:
                        status = 'block_on_state_block_list'
                else:
                    # If the state is not on the list, do the opposite of what the list says
                    if rule_type == 'allow':
                        status = 'block_not_on_state_allow_list'
                    else:
                        status = 'allow_not_on_state_block_list'

        if return_status:
            return status
        else:
            if status.startswith('allow'):
                return True
            else:
                assert status.startswith('block')
                return False

    # ...def species_allowed_in_country(...)


    def export_geofence_data_to_csv(self, csv_fn=None, include_common_names=True):
        """
        Converts the geofence .json representation into an equivalent .csv representation,
        with one taxon per row and one region per column.  Empty values indicate non-allowed
        combinations, positive numbers indicate allowed combinations.  Negative values
        are reserved for specific non-allowed combinations.

        Args:
            csv_fn (str): output .csv file
            include_common_names (bool, optional): include a column for common names

        Returns:
            dataframe: the pandas representation of the csv output file
        """

        all_taxa = sorted(list(self.taxonomy_string_to_geofencing_rules.keys()))
        print('Preparing geofencing export for {} taxa'.format(len(all_taxa)))

        all_regions = set()

        # taxon = all_taxa[0]
        for taxon in all_taxa:

            taxon_rules = self.taxonomy_string_to_geofencing_rules[taxon]
            for rule_type in taxon_rules.keys():

                assert rule_type in ('allow','block')
                all_country_rules_this_species = taxon_rules[rule_type]

                for country_code in all_country_rules_this_species.keys():
                    all_regions.add(country_code)
                    assert country_code in self.country_code_to_country
                    assert len(country_code) == 3
                    region_rules = all_country_rules_this_species[country_code]
                    if len(region_rules) > 0:
                        assert country_code == 'USA'
                        for region_name in region_rules:
                            assert len(region_name) == 2
                            assert isinstance(region_name,str)
                            all_regions.add(country_code + ':' + region_name)

        all_regions = sorted(list(all_regions))

        print('Found {} regions'.format(len(all_regions)))

        n_allowed = 0
        df = pd.DataFrame(index=all_taxa,columns=all_regions)
        # df = df.fillna(np.nan)

        for taxon in tqdm(all_taxa):
            for region in all_regions:
                tokens = region.split(':')
                country_code = tokens[0]
                state_code = None
                if len(tokens) > 1:
                    state_code = tokens[1]
                allowed = self.species_allowed_in_country(species=taxon,
                                                     country=country_code,
                                                     state=state_code,
                                                     return_status=False)
                if allowed:
                    n_allowed += 1
                    df.loc[taxon,region] = 1

            # ...for each region

        # ...for each taxon

        print('Allowed {} of {} combinations'.format(n_allowed,len(all_taxa)*len(all_regions)))

        # Before saving, convert columns with numeric values to integers
        for col in df.columns:
            # Check whether each column has any non-NaN values that could be integers
            if df[col].notna().any() and pd.to_numeric(df[col], errors='coerce').notna().any():
                # Convert column to Int64 type (pandas nullable integer type)
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        if include_common_names:
            df.insert(loc=0,column='common_name',value='')
            for taxon in all_taxa:
                if taxon in self.taxonomy_string_to_taxonomy_info:
                    taxonomy_info = self.taxonomy_string_to_taxonomy_info[taxon]
                    common_name = taxonomy_info['common_name']
                    assert isinstance(common_name,str) and len(common_name) < 50
                    df.loc[taxon,'common_name'] = common_name

        if csv_fn is not None:
            df.to_csv(csv_fn,index=True,header=True)

        return df

    # ...def export_geofence_data_to_csv(...)

# ...class TaxonomyHandler
