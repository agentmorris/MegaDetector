"""

classification_postprocessing.py

Functions for postprocessing species classification results, particularly:

* Smoothing results within an image (an image with 700 cows and one deer is really just 701
  cows)
* Smoothing results within a sequence (a sequence that looks like deer/deer/deer/elk/deer/deer
  is really just a deer)

"""

#%% Constants and imports

import json
import copy
import pandas as pd

from collections import defaultdict
from tqdm import tqdm

from megadetector.utils.ct_utils import is_list_sorted
from megadetector.utils.ct_utils import is_empty
from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.ct_utils import sort_dictionary_by_key
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import write_json

from megadetector.utils.wi_taxonomy_utils import clean_taxonomy_string
from megadetector.utils.wi_taxonomy_utils import taxonomy_level_index
from megadetector.utils.wi_taxonomy_utils import taxonomy_level_string_to_index

from megadetector.utils.wi_taxonomy_utils import human_prediction_string
from megadetector.utils.wi_taxonomy_utils import animal_prediction_string
from megadetector.utils.wi_taxonomy_utils import is_taxonomic_prediction_string
from megadetector.utils.wi_taxonomy_utils import blank_prediction_string # noqa


#%% Options classes

class ClassificationSmoothingOptions:
    """
    Options used to parameterize smooth_classification_results_image_level()
    and smooth_classification_results_sequence_level()
    """

    def __init__(self):

        #: How many detections do we need in a dominant category to overwrite
        #: non-dominant classifications?  This is irrelevant if
        #: max_detections_nondominant_class <= 1.
        self.min_detections_to_overwrite_secondary = 4

        #: Even if we have a dominant class, if a non-dominant class has at least
        #: this many classifications in an image, leave them alone.
        #:
        #: If this is <= 1, we won't replace non-dominant, non-other classes
        #: with the dominant class, even if there are 900 cows and 1 deer.
        self.max_detections_nondominant_class = 1

        #: How many detections do we need in a dominant category to overwrite
        #: non-dominant classifications in the same family?  If this is <= 0,
        #: we'll skip this step.  This option doesn't mean anything if
        #: max_detections_nondominant_class_same_family <= 1.
        self.min_detections_to_overwrite_secondary_same_family = 2

        #: If we have this many classifications of a nondominant category,
        #: we won't do same-family overwrites.  <= 1 means "even if there are
        #: a million deer, if there are two million moose, call all the deer
        #: moose".  This option doesn't mean anything if
        #: min_detections_to_overwrite_secondary_same_family <= 0.
        self.max_detections_nondominant_class_same_family = -1

        #: If the dominant class has at least this many classifications, overwrite
        #: "other" classifications with the dominant class
        self.min_detections_to_overwrite_other = 2

        #: Names to treat as "other" categories; can't be None, but can be empty
        #:
        #: "Other" classifications will be changed to the dominant category, regardless
        #: of confidence, as long as there are at least min_detections_to_overwrite_other
        #: examples of the dominant class.  For example, cow/other will remain unchanged,
        #: but cow/cow/other will become cow/cow/cow.
        self.other_category_names = ['other','unknown','no cv result','animal','blank','mammal']

        #: We're not even going to mess around with classifications below this threshold.
        #:
        #: We won't count them, we won't over-write them, they don't exist during the
        #: within-image smoothing step.
        self.classification_confidence_threshold = 0.5

        #: We're not even going to mess around with detections below this threshold.
        #:
        #: We won't count them, we won't over-write them, they don't exist during the
        #: within-image smoothing step.
        self.detection_confidence_threshold = 0.15

        #: If classification descriptions are present and appear to represent taxonomic
        #: information, should we propagate classifications when lower-level taxa are more
        #: common in an image?  For example, if we see "carnivore/fox/fox/deer", should
        #: we make that "fox/fox/fox/deer"?
        self.propagate_classifications_through_taxonomy = True

        #: When propagating classifications down through taxonomy levels, we have to
        #: decide whether we prefer more frequent categories or more specific categories.
        #: taxonomy_propagation_level_weight and taxonomy_propagation_count_weight
        #: balance levels against counts in this process.
        self.taxonomy_propagation_level_weight = 1.0

        #: When propagating classifications down through taxonomy levels, we have to
        #: decide whether we prefer more frequent categories or more specific categories.
        #: taxonomy_propagation_level_weight and taxonomy_propagation_count_weight
        #: balance levels against counts in this process.
        #:
        #: With a very low default value, this just breaks ties.
        self.taxonomy_propagation_count_weight = 0.01

        #: Should we record information about the state of labels prior to smoothing?
        self.add_pre_smoothing_description = True

        #: When a dict (rather than a file) is passed to either smoothing function,
        #: if this is True, we'll make a copy of the input dict before modifying.
        self.modify_in_place = False

        #: Only include these categories in the smoothing process (None to use all categories)
        self.detection_category_names_to_smooth = ['animal']

        #: Debug options
        self.break_at_image = None

        ## Populated internally

        #: Only include these categories in the smoothing process (None to use all categories)
        self._detection_category_ids_to_smooth = None


#%% Utility functions

def _results_for_sequence(images_this_sequence,filename_to_results):
    """
    Fetch MD results for every image in this sequence, based on the 'file_name' field
    """

    results_this_sequence = []
    for im in images_this_sequence:
        fn = im['file_name']
        results_this_image = filename_to_results[fn]
        assert isinstance(results_this_image,dict)
        results_this_sequence.append(results_this_image)

    return results_this_sequence


def _sort_images_by_time(images):
    """
    Returns a copy of [images], sorted by the 'datetime' field (ascending).
    """
    return sorted(images, key = lambda im: im['datetime'])


def _detection_is_relevant_for_smoothing(det,options):
    """
    Determine whether [det] has classifications that might be meaningful for smoothing.
    """

    if ('classifications' not in det) or \
        (det['conf'] < options.detection_confidence_threshold):
        return False

    # Ignore non-smoothed categories
    if (options._detection_category_ids_to_smooth is not None) and \
        (det['category'] not in options._detection_category_ids_to_smooth):
        return False

    return True


def count_detections_by_classification_category(detections,options=None):
    """
    Count the number of instances of each classification category in the detections list
    [detections] that have an above-threshold detection.  Sort results in descending
    order by count.  Returns a dict mapping category ID --> count.  If no detections
    are above threshold, returns an empty dict.

    Only processes the top classification for each detection.

    Args:
        detections (list of dict): detections list
        options (ClassificationSmoothingOptions, optional): see ClassificationSmoothingOptions

    Returns:
        dict mapping above-threshold category IDs to counts
    """

    if detections is None or len(detections) == 0:
        return {}

    if options is None:
        options = ClassificationSmoothingOptions()

    category_to_count = defaultdict(int)

    for det in detections:

        if not _detection_is_relevant_for_smoothing(det,options):
            continue

        c = det['classifications'][0]
        if c[1] >= options.classification_confidence_threshold:
            category_to_count[c[0]] += 1

    category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                 key=lambda item: item[1],
                                                 reverse=True)}

    return category_to_count


def get_classification_description_string(category_to_count,classification_descriptions):
    """
    Return a string summarizing the image content according to [category_to_count].

    Args:
        category_to_count (dict): a dict mapping category IDs to counts
        classification_descriptions (dict): a dict mapping category IDs to description strings

    Returns:
        string: a description of this image's content, e.g. "rabbit (4), human (1)"
    """

    category_strings = []
    # category_id = next(iter(category_to_count))
    for category_id in category_to_count:
        category_description = classification_descriptions[category_id]
        tokens = category_description.split(';')
        assert len(tokens) == 7
        category_name = tokens[-1]
        if len(category_name) == 0:
            category_name = 'undefined category'
        count = category_to_count[category_id]
        category_string = '{} ({})'.format(category_name,count)
        category_strings.append(category_string)

    return ', '.join(category_strings)


def _print_counts_with_names(category_to_count,classification_descriptions):
    """
    Print a list of classification categories with counts, based in the name --> count
    dict [category_to_count]
    """

    for category_id in category_to_count:
        category_name = classification_descriptions[category_id]
        count = category_to_count[category_id]
        print('{}: {} ({})'.format(category_id,category_name,count))


def _prepare_results_for_smoothing(input_file,options):
    """
    Load results from [input_file] if necessary, prepare category descriptions
    for smoothing.  Adds pre-smoothing descriptions to every image if the options
    say we're supposed to do that.

    May modify some fields in [options].
    """

    if isinstance(input_file,str):
        with open(input_file,'r') as f:
            print('Loading results from:\n{}'.format(input_file))
            d = json.load(f)
    else:
        assert isinstance(input_file,dict)
        if options.modify_in_place:
            d = input_file
        else:
            print('modify_in_place is False, copying the input before modifying')
            d = copy.deepcopy(input_file)


    ## Category processing

    category_name_to_id = {d['classification_categories'][k]:k for k in d['classification_categories']}
    other_category_ids = []
    for s in options.other_category_names:
        if s in category_name_to_id:
            other_category_ids.append(category_name_to_id[s])

    # Possibly update the list of category IDs we should smooth
    if options.detection_category_names_to_smooth is None:
        options._detection_category_ids_to_smooth = None
    else:
        detection_category_id_to_name = d['detection_categories']
        detection_category_name_to_id = invert_dictionary(detection_category_id_to_name)
        options._detection_category_ids_to_smooth = []
        for category_name in options.detection_category_names_to_smooth:
            options._detection_category_ids_to_smooth.append(detection_category_name_to_id[category_name])

    # Before we do anything else, get rid of everything but the top classification
    # for each detection, and remove the 'classifications' field from detections with
    # no classifications.
    for im in tqdm(d['images']):

        if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
            continue

        detections = im['detections']

        for det in detections:

            if 'classifications' not in det:
                continue
            if len(det['classifications']) == 0:
                del det['classifications']
                continue

            classification_confidence_values = [c[1] for c in det['classifications']]
            assert is_list_sorted(classification_confidence_values,reverse=True)
            det['classifications'] = [det['classifications'][0]]

        # ...for each detection in this image

    # ...for each image


    ## Clean up classification descriptions...

    # ...so we can test taxonomic relationships by substring testing.

    classification_descriptions_clean = None
    classification_descriptions = None

    if 'classification_category_descriptions' in d:
        classification_descriptions = d['classification_category_descriptions']
        classification_descriptions_clean = {}
        # category_id = next(iter(classification_descriptions))
        for category_id in classification_descriptions:
            classification_descriptions_clean[category_id] = \
                clean_taxonomy_string(classification_descriptions[category_id]).strip(';').lower()


    ## Optionally add pre-smoothing descriptions to every image

    if options.add_pre_smoothing_description and (classification_descriptions is not None):

        for im in tqdm(d['images']):

            if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
                continue

            detections = im['detections']
            category_to_count = count_detections_by_classification_category(detections, options)

            im['pre_smoothing_description'] = \
                get_classification_description_string(category_to_count, classification_descriptions)


    return {
        'd':d,
        'other_category_ids':other_category_ids,
        'classification_descriptions_clean':classification_descriptions_clean,
        'classification_descriptions':classification_descriptions
    }

# ...def _prepare_results_for_smoothing(...)


def _smooth_classifications_for_list_of_detections(detections,
                                                   options,
                                                   other_category_ids,
                                                   classification_descriptions,
                                                   classification_descriptions_clean):
    """
    Smooth classifications for a list of detections, which may have come from a single
    image, or may represent an entire sequence.

    Returns None if no changes are made, else a dict.

    classification_descriptions_clean should be semicolon-delimited taxonomic strings
    from which common names and GUIDs have already been removed.

    Assumes there is only one classification per detection, i.e. that non-top classifications
    have already been remoevd.
    """

    ## Count the number of instances of each category in this image

    category_to_count = count_detections_by_classification_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    # get_classification_description_string(category_to_count, classification_descriptions)

    if len(category_to_count) <= 1:
        return None

    keys = list(category_to_count.keys())

    # Handle a quirky special case: if the most common category is "other" and
    # it's "tied" with the second-most-common category, swap them
    if (len(keys) > 1) and \
        (keys[0] in other_category_ids) and \
        (keys[1] not in other_category_ids) and \
        (category_to_count[keys[0]] == category_to_count[keys[1]]):
            keys[1], keys[0] = keys[0], keys[1]

    max_count = category_to_count[keys[0]]
    most_common_category = keys[0]
    del keys


    ## Debug tools

    verbose_debug_enabled = False

    if options.break_at_image is not None:
        for det in detections:
            if 'image_filename' in det and \
                det['image_filename'] == options.break_at_image:
                verbose_debug_enabled = True
                break

    if verbose_debug_enabled:
        _print_counts_with_names(category_to_count,classification_descriptions)
        # from IPython import embed; embed()


    ## Possibly change "other" classifications to the most common category

    # ...if the dominant category is not an "other" category.

    n_other_classifications_changed_this_image = 0

    # If we have at least *min_detections_to_overwrite_other* in a category that isn't
    # "other", change all "other" classifications to that category
    if (max_count >= options.min_detections_to_overwrite_other) and \
        (most_common_category not in other_category_ids):

        for det in detections:

            if not _detection_is_relevant_for_smoothing(det,options):
                continue

            assert len(det['classifications']) == 1
            c = det['classifications'][0]

            if (c[1] >= options.classification_confidence_threshold) and \
               (c[0] in other_category_ids):

                if verbose_debug_enabled:
                    print('Replacing {} with {}'.format(
                        classification_descriptions[c[0]],
                        most_common_category))

                n_other_classifications_changed_this_image += 1
                c[0] = most_common_category

            # ...if there are classifications for this detection

        # ...for each detection

    # ...if we should overwrite all "other" classifications

    if verbose_debug_enabled:
        print('Made {} other changes'.format(n_other_classifications_changed_this_image))


    ## Re-count

    category_to_count = count_detections_by_classification_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    keys = list(category_to_count.keys())
    max_count = category_to_count[keys[0]]
    most_common_category = keys[0]
    del keys


    ## Possibly change some non-dominant classifications to the dominant category

    process_taxonomic_rules = \
        (classification_descriptions_clean is not None) and \
        (len(classification_descriptions_clean) > 0) and \
        (len(category_to_count) > 1)

    n_detections_flipped_this_image = 0

    # Don't do this if the most common category is an "other" category, or
    # if we don't have enough of the most common category
    if (most_common_category not in other_category_ids) and \
       (max_count >= options.min_detections_to_overwrite_secondary):

        # i_det = 0; det = detections[i_det]
        for i_det,det in enumerate(detections):

            if not _detection_is_relevant_for_smoothing(det,options):
                continue

            assert len(det['classifications']) == 1
            c = det['classifications'][0]

            # Don't over-write the most common category with itself
            if c[0] == most_common_category:
                continue

            # Don't bother with below-threshold classifications
            if c[1] < options.classification_confidence_threshold:
                continue

            # If we're doing taxonomic processing, at this stage, don't turn children
            # into parents; we'll likely turn parents into children in the next stage.
            if process_taxonomic_rules:

                most_common_category_description = \
                    classification_descriptions_clean[most_common_category]

                category_id_this_classification = c[0]
                assert category_id_this_classification in category_to_count

                category_description_this_classification = \
                    classification_descriptions_clean[category_id_this_classification]

                # An empty description corresponds to the "animal" category.  We don't handle
                # "animal" here as a parent category, that would be handled in the "other smoothing"
                # step above.
                if len(category_description_this_classification) == 0:
                    continue

                most_common_category_is_parent_of_this_category = \
                    most_common_category_description in category_description_this_classification

                if most_common_category_is_parent_of_this_category:
                    continue

            # If we have fewer of this category than the most common category,
            # but not *too* many, flip it to the most common category.
            if (max_count > category_to_count[c[0]]) and \
               (category_to_count[c[0]] <= options.max_detections_nondominant_class):

                c[0] = most_common_category
                n_detections_flipped_this_image += 1

        # ...for each detection

    # ...if the dominant category is legit

    if verbose_debug_enabled:
        print('Made {} non-dominant --> dominant changes'.format(
            n_detections_flipped_this_image))


    ## Re-count

    category_to_count = count_detections_by_classification_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    keys = list(category_to_count.keys())
    max_count = category_to_count[keys[0]]
    most_common_category = keys[0]
    del keys


    ## Possibly collapse higher-level taxonomic predictions down to lower levels

    n_taxonomic_changes_this_image = 0

    process_taxonomic_rules = \
        (classification_descriptions_clean is not None) and \
        (len(classification_descriptions_clean) > 0) and \
        (len(category_to_count) > 1)

    if process_taxonomic_rules and options.propagate_classifications_through_taxonomy:

        # det = detections[3]
        for det in detections:

            if not _detection_is_relevant_for_smoothing(det,options):
                continue

            assert len(det['classifications']) == 1
            c = det['classifications'][0]

            # Don't bother with any classifications below the confidence threshold
            if c[1] < options.classification_confidence_threshold:
                continue

            category_id_this_classification = c[0]
            assert category_id_this_classification in category_to_count

            category_description_this_classification = \
                classification_descriptions_clean[category_id_this_classification]

            # An empty description corresponds to the "animal" category.  We don't handle
            # "animal" here as a parent category, that would be handled in the "other smoothing"
            # step above.
            if len(category_description_this_classification) == 0:
                continue

            # We may have multiple child categories to choose from; this keeps track of
            # the "best" we've seen so far.  "Best" is based on the level (species is better
            # than genus) and number.
            child_category_to_score = defaultdict(float)

            for category_id_of_candidate_child in category_to_count.keys():

                # A category is never its own child
                if category_id_of_candidate_child == category_id_this_classification:
                    continue

                # Is this candidate a child of the current classification?
                category_description_candidate_child = \
                    classification_descriptions_clean[category_id_of_candidate_child]

                # An empty description corresponds to "animal", which can never
                # be a child of another category.
                if len(category_description_candidate_child) == 0:
                    continue

                # This handles a case that doesn't come up with "pure" SpeciesNet results;
                # if two categories have different IDs but the same "clean" description, this
                # means they're different common names for the same species, which we use
                # for things like "white-tailed deer buck" and "white-tailed deer fawn".
                #
                # Currently we don't support smoothing those predictions, because it's not a
                # parent/child relationship.
                if category_description_candidate_child == \
                    category_description_this_classification:
                    continue

                # As long as we're using "clean" descriptions, parent/child taxonomic
                # relationships are defined by a substring relationship
                is_child = category_description_this_classification in \
                    category_description_candidate_child

                if not is_child:
                    continue

                # How many instances of this child category are there?
                child_category_count = category_to_count[category_id_of_candidate_child]

                # What taxonomy level is this child category defined at?
                child_category_level = taxonomy_level_index(
                    classification_descriptions[category_id_of_candidate_child])

                child_category_to_score[category_id_of_candidate_child] = \
                    child_category_level * options.taxonomy_propagation_level_weight + \
                    child_category_count * options.taxonomy_propagation_count_weight

            # ...for each category we are considering reducing this classification to

            # Did we find a category we want to change this classification to?
            if len(child_category_to_score) > 0:

                # Find the child category with the highest score
                child_category_to_score = sort_dictionary_by_value(
                    child_category_to_score,reverse=True)
                best_child_category = next(iter(child_category_to_score.keys()))

                if verbose_debug_enabled:
                    old_category_name = \
                        classification_descriptions_clean[c[0]]
                    new_category_name = \
                        classification_descriptions_clean[best_child_category]
                    print('Replacing {} with {}'.format(
                        old_category_name,new_category_name))

                c[0] = best_child_category
                n_taxonomic_changes_this_image += 1

        # ...for each detection

    # ...if we have taxonomic information available


    ## Re-count

    category_to_count = count_detections_by_classification_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    keys = list(category_to_count.keys())
    max_count = category_to_count[keys[0]]
    most_common_category = keys[0]
    del keys


    ## Possibly do within-family smoothing

    n_within_family_smoothing_changes = 0

    # min_detections_to_overwrite_secondary_same_family = -1
    # max_detections_nondominant_class_same_family = 1
    family_level = taxonomy_level_string_to_index('family')

    if process_taxonomic_rules:

        category_description_most_common_category = \
            classification_descriptions[most_common_category]
        most_common_category_taxonomic_level = \
            taxonomy_level_index(category_description_most_common_category)
        n_most_common_category = category_to_count[most_common_category]
        tokens = category_description_most_common_category.split(';')
        assert len(tokens) == 7
        most_common_category_family = tokens[3]
        most_common_category_genus = tokens[4]

    # Only consider remapping to genus or species level, and only when we have
    # a high enough count in the most common category
    if process_taxonomic_rules and \
        (options.min_detections_to_overwrite_secondary_same_family > 0) and \
        (most_common_category not in other_category_ids) and \
        (most_common_category_taxonomic_level > family_level) and \
        (n_most_common_category >= options.min_detections_to_overwrite_secondary_same_family):

        # det = detections[0]
        for det in detections:

            if not _detection_is_relevant_for_smoothing(det,options):
                continue

            assert len(det['classifications']) == 1
            c = det['classifications'][0]

            # Don't over-write the most common category with itself
            if c[0] == most_common_category:
                continue

            # Don't bother with below-threshold classifications
            if c[1] < options.classification_confidence_threshold:
               continue

            n_candidate_flip_category = category_to_count[c[0]]

            # Do we have too many of the non-dominant category to do this kind of swap?
            if n_candidate_flip_category > \
                options.max_detections_nondominant_class_same_family:
                continue

            # Don't flip classes when it's a tie
            if n_candidate_flip_category == n_most_common_category:
                continue

            category_description_candidate_flip = \
                classification_descriptions[c[0]]
            tokens = category_description_candidate_flip.split(';')
            assert len(tokens) == 7
            candidate_flip_category_family = tokens[3]
            candidate_flip_category_genus = tokens[4]
            candidate_flip_category_taxonomic_level = \
                taxonomy_level_index(category_description_candidate_flip)

            # Only proceed if we have valid family strings
            if (len(candidate_flip_category_family) == 0) or \
                (len(most_common_category_family) == 0):
                continue

            # Only proceed if the candidate and the most common category are in the same family
            if candidate_flip_category_family != most_common_category_family:
                continue

            # Don't flip from a species to the genus level in the same genus
            if (candidate_flip_category_genus == most_common_category_genus) and \
                (candidate_flip_category_taxonomic_level > \
                 most_common_category_taxonomic_level):
                continue

            old_category_name = classification_descriptions_clean[c[0]]
            new_category_name = classification_descriptions_clean[most_common_category]

            c[0] = most_common_category
            n_within_family_smoothing_changes += 1

        # ...for each detection

    # ...if the dominant category is legit and we have taxonomic information available

    return {'n_other_classifications_changed_this_image':n_other_classifications_changed_this_image,
            'n_detections_flipped_this_image':n_detections_flipped_this_image,
            'n_taxonomic_changes_this_image':n_taxonomic_changes_this_image,
            'n_within_family_smoothing_changes':n_within_family_smoothing_changes}

# ...def _smooth_classifications_for_list_of_detections(...)


def _smooth_single_image(im,
                         options,
                         other_category_ids,
                         classification_descriptions,
                         classification_descriptions_clean):
    """
    Smooth classifications for a single image.  Returns None if no changes are made,
    else a dict.

    classification_descriptions_clean should be semicolon-delimited taxonomic strings
    from which common names and GUIDs have already been removed.

    Assumes there is only one classification per detection, i.e. that non-top classifications
    have already been remoevd.
    """

    if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
        return

    detections = im['detections']

    # Simplify debugging
    for det in detections:
        det['image_filename'] = im['file']

    to_return = _smooth_classifications_for_list_of_detections(detections,
        options=options,
        other_category_ids=other_category_ids,
        classification_descriptions=classification_descriptions,
        classification_descriptions_clean=classification_descriptions_clean)

    # Clean out debug information
    for det in detections:
        del det['image_filename']

    return to_return

# ...def smooth_single_image


#%% Image-level smoothing

def smooth_classification_results_image_level(input_file,output_file=None,options=None):
    """
    Smooth classifications at the image level for all results in the MD-formatted results
    file [input_file], optionally writing a new set of results to [output_file].

    This function generally expresses the notion that an image with 700 cows and one deer
    is really just 701 cows.

    Only count detections with a classification confidence threshold above
    [options.classification_confidence_threshold], which in practice means we're only
    looking at one category per detection.

    If an image has at least [options.min_detections_to_overwrite_secondary] such detections
    in the most common category, and no more than [options.max_detections_nondominant_class]
    in the second-most-common category, flip all detections to the most common
    category.

    Optionally treat some classes as particularly unreliable, typically used to overwrite an
    "other" class.

    This function also removes everything but the non-dominant classification for each detection.

    Args:
        input_file (str): MegaDetector-formatted classification results file to smooth.  Can
            also be an already-loaded results dict.
        output_file (str, optional): .json file to write smoothed results
        options (ClassificationSmoothingOptions, optional): see
          ClassificationSmoothingOptions for details.

    Returns:
        dict: MegaDetector-results-formatted dict, identical to what's written to
        [output_file] if [output_file] is not None.
    """

    ## Input validation

    if options is None:
        options = ClassificationSmoothingOptions()

    r = _prepare_results_for_smoothing(input_file, options)
    d = r['d']
    other_category_ids = r['other_category_ids']
    classification_descriptions_clean = r['classification_descriptions_clean']
    classification_descriptions = r['classification_descriptions']


    ## Smoothing

    n_other_classifications_changed = 0
    n_other_images_changed = 0
    n_taxonomic_images_changed = 0

    n_detections_flipped = 0
    n_images_changed = 0
    n_taxonomic_classification_changes = 0

    # im = d['images'][0]
    for im in tqdm(d['images']):

        r = _smooth_single_image(im,
                                 options,
                                 other_category_ids,
                                 classification_descriptions=classification_descriptions,
                                 classification_descriptions_clean=classification_descriptions_clean)

        if r is None:
            continue

        n_detections_flipped_this_image = r['n_detections_flipped_this_image']
        n_other_classifications_changed_this_image = \
            r['n_other_classifications_changed_this_image']
        n_taxonomic_changes_this_image = r['n_taxonomic_changes_this_image']

        n_detections_flipped += n_detections_flipped_this_image
        n_other_classifications_changed += n_other_classifications_changed_this_image
        n_taxonomic_classification_changes += n_taxonomic_changes_this_image

        if n_detections_flipped_this_image > 0:
            n_images_changed += 1
        if n_other_classifications_changed_this_image > 0:
            n_other_images_changed += 1
        if n_taxonomic_changes_this_image > 0:
            n_taxonomic_images_changed += 1

    # ...for each image

    print('Classification smoothing: changed {} detections on {} images'.format(
        n_detections_flipped,n_images_changed))

    print('"Other" smoothing: changed {} detections on {} images'.format(
          n_other_classifications_changed,n_other_images_changed))

    print('Taxonomic smoothing: changed {} detections on {} images'.format(
          n_taxonomic_classification_changes,n_taxonomic_images_changed))


    ## Write output

    if output_file is not None:
        print('Writing results after image-level smoothing to:\n{}'.format(output_file))
        write_json(output_file,d)

    return d

# ...def smooth_classification_results_image_level(...)


#%% Sequence-level smoothing

def smooth_classification_results_sequence_level(input_file,
                                                 cct_sequence_information,
                                                 output_file=None,
                                                 options=None):
    """
    Smooth classifications at the sequence level for all results in the MD-formatted results
    file [md_results_file], optionally writing a new set of results to [output_file].

    This function generally expresses the notion that a sequence that looks like
    deer/deer/deer/elk/deer/deer/deer/deer is really just a deer.

    Args:
        input_file (str or dict): MegaDetector-formatted classification results file to smooth
          (or already-loaded results).  If you supply a dict, it's copied by default, but
          in-place modification is supported via options.modify_in_place.
        cct_sequence_information (str, dict, or list): COCO Camera Traps file containing sequence IDs for
          each image (or an already-loaded CCT-formatted dict, or just the 'images' list from a CCT dict).
        output_file (str, optional): .json file to write smoothed results
        options (ClassificationSmoothingOptions, optional): see
          ClassificationSmoothingOptions for details.

    Returns:
        dict: MegaDetector-results-formatted dict, identical to what's written to
        [output_file] if [output_file] is not None.
    """

    ## Input validation

    if options is None:
        options = ClassificationSmoothingOptions()

    r = _prepare_results_for_smoothing(input_file, options)
    d = r['d']
    other_category_ids = r['other_category_ids']
    classification_descriptions_clean = r['classification_descriptions_clean']
    classification_descriptions = r['classification_descriptions']


    ## Make a list of images appearing in each sequence

    if isinstance(cct_sequence_information,list):
        image_info = cct_sequence_information
    elif isinstance(cct_sequence_information,str):
        print('Loading sequence information from {}'.format(cct_sequence_information))
        with open(cct_sequence_information,'r') as f:
            cct_sequence_information = json.load(f)
            image_info = cct_sequence_information['images']
    else:
        assert isinstance(cct_sequence_information,dict)
        image_info = cct_sequence_information['images']

    sequence_to_image_filenames = defaultdict(list)

    # im = image_info[0]
    for im in tqdm(image_info):
        sequence_to_image_filenames[im['seq_id']].append(im['file_name'])
    del image_info

    image_fn_to_classification_results = {}
    for im in d['images']:
        fn = im['file']
        assert fn not in image_fn_to_classification_results
        image_fn_to_classification_results[fn] = im


    ## Smoothing

    n_other_classifications_changed = 0
    n_other_sequences_changed = 0
    n_taxonomic_sequences_changed = 0
    n_within_family_sequences_changed = 0

    n_detections_flipped = 0
    n_sequences_changed = 0
    n_taxonomic_classification_changes = 0
    n_within_family_changes = 0

    # sequence_id = list(sequence_to_image_filenames.keys())[0]
    for sequence_id in sequence_to_image_filenames.keys():

        image_filenames_this_sequence = sequence_to_image_filenames[sequence_id]

        # if 'file' in image_filenames_this_sequence:
        #    from IPython import embed; embed()

        detections_this_sequence = []
        for image_filename in image_filenames_this_sequence:
            if image_filename not in image_fn_to_classification_results:
                print('Warning: {} in sequence list but not in results'.format(
                    image_filename))
                continue
            im = image_fn_to_classification_results[image_filename]
            if 'detections' not in im or im['detections'] is None:
                continue
            detections_this_sequence.extend(im['detections'])

            # Temporarily add image filenames to every detection,
            # for debugging
            for det in im['detections']:
                det['image_filename'] = im['file']

        if len(detections_this_sequence) == 0:
            continue

        r = _smooth_classifications_for_list_of_detections(
            detections=detections_this_sequence,
            options=options,
            other_category_ids=other_category_ids,
            classification_descriptions=classification_descriptions,
            classification_descriptions_clean=classification_descriptions_clean)

        if r is None:
            continue

        n_detections_flipped_this_sequence = r['n_detections_flipped_this_image']
        n_other_classifications_changed_this_sequence = \
            r['n_other_classifications_changed_this_image']
        n_taxonomic_changes_this_sequence = r['n_taxonomic_changes_this_image']
        n_within_family_changes_this_sequence = r['n_within_family_smoothing_changes']

        n_detections_flipped += n_detections_flipped_this_sequence
        n_other_classifications_changed += n_other_classifications_changed_this_sequence
        n_taxonomic_classification_changes += n_taxonomic_changes_this_sequence
        n_within_family_changes += n_within_family_changes_this_sequence

        if n_detections_flipped_this_sequence > 0:
            n_sequences_changed += 1
        if n_other_classifications_changed_this_sequence > 0:
            n_other_sequences_changed += 1
        if n_taxonomic_changes_this_sequence > 0:
            n_taxonomic_sequences_changed += 1
        if n_within_family_changes_this_sequence > 0:
            n_within_family_sequences_changed += 1

    # ...for each sequence

    print('Classification smoothing: changed {} detections in {} sequences'.format(
        n_detections_flipped,n_sequences_changed))

    print('"Other" smoothing: changed {} detections in {} sequences'.format(
          n_other_classifications_changed,n_other_sequences_changed))

    print('Taxonomic smoothing: changed {} detections in {} sequences'.format(
          n_taxonomic_classification_changes,n_taxonomic_sequences_changed))

    print('Within-family smoothing: changed {} detections in {} sequences'.format(
          n_within_family_changes,n_within_family_sequences_changed))


    ## Clean up debug information

    for im in d['images']:
        if 'detections' not in im or im['detections'] is None:
            continue
        for det in im['detections']:
            if 'image_filename' in det:
                del det['image_filename']


    ## Write output

    if output_file is not None:
        print('Writing sequence-smoothed classification results to {}'.format(
            output_file))
        write_json(output_file,d)

    return d

# ...smooth_classification_results_sequence_level(...)


def restrict_to_taxa_list(taxa_list,
                          speciesnet_taxonomy_file,
                          input_file,
                          output_file,
                          allow_walk_down=False,
                          add_pre_filtering_description=True,
                          allow_redundant_latin_names=True,
                          protected_common_names=None,
                          use_original_common_names_if_available=True,
                          verbose=True):
    """
    Given a prediction file in MD .json format, likely without having had
    a geofence applied, apply a custom taxa list.

    Args:
        taxa_list (str): .csv file with at least the columns "latin" and "common".
        speciesnet_taxonomy_file (str): taxonomy filename, in the same format used for
            model release (with 7-token taxonomy entries)
        input_file (str): .json file to read, in MD format.  This can be None, in which
            case this function just validates [taxa_list].
        output_file (str): .json file to write, in MD format
        allow_walk_down (bool, optional): should we walk down the taxonomy tree
            when making mappings if a parent has only a single allowable child?
            For example, if only a single felid species is allowed, should other
            felid predictions be mapped to that species, as opposed to being mapped
            to the family?
        add_pre_filtering_description (bool, optional): should we add a new metadata
            field that summarizes each image's classifications prior to taxonomic
            restriction?
        allow_redundant_latin_names (bool, optional): if False, we'll raise an Exception
            if the same latin name appears twice in the taxonomy list; if True, we'll
            just print a warning and ignore all entries other than the first for this
            latin name
        protected_common_names (list, optional): these categories should be
            unmodified, even if they aren't used, or have the same taxonomic
            description as other categories
        use_original_common_names_if_available (bool, optional): if an "original_common"
            column is present in [taxa_list], use those common names instead of the ones
            in the taxonomy file
        verbose (bool, optional): enable additional debug output
    """

    ##%% Read target taxa list

    taxa_list_df = pd.read_csv(taxa_list)

    required_columns = ('latin','common')
    for s in required_columns:
        assert s in taxa_list_df.columns, \
            'Required column {} missing from taxonomy list file {}'.format(
                s,taxa_list)

    # Convert the "latin" and "common" columns in taxa_list_df to lowercase
    taxa_list_df['latin'] = taxa_list_df['latin'].str.lower()
    taxa_list_df['common'] = taxa_list_df['common'].str.lower()

    # Remove rows from taxa_list_df where the "latin" column is nan,
    # printing a warning for each row (with a string representation of the whole row)
    for i_row,row in taxa_list_df.iterrows():
        if pd.isna(row['latin']):
            if verbose:
                print('Warning: Skipping row with empty "latin" column in {}:\n{}\n'.format(
                    taxa_list,str(row.to_dict())))
            taxa_list_df.drop(index=i_row, inplace=True)

    # Convert all NaN values in the "common" column to empty strings
    taxa_list_df['common'] = taxa_list_df['common'].fillna('')

    # Create a dictionary mapping source Latin names to target common names
    target_latin_to_common = {}

    for i_row,row in taxa_list_df.iterrows():

        latin = row['latin']
        common = row['common']

        if use_original_common_names_if_available and \
            ('original_common' in row) and \
            (not is_empty(row['original_common'])):
                common = row['original_common'].strip().lower()

        # Valid latin names have either one token (e.g. "canidae"),
        # two tokens (e.g. "bos taurus"), or three tokens (e.g. "canis lupus familiaris")
        assert len(latin.split(' ')) in (1,2,3), \
            'Illegal binomial name {} in taxaonomy list {}'.format(
                latin,taxa_list)

        if latin in target_latin_to_common:
            error_string = \
                'scientific name {} appears multiple times in the taxonomy list'.format(
                latin)
            if allow_redundant_latin_names:
                if verbose:
                    print('Warning: {}'.format(error_string))
            else:
                raise ValueError(error_string)

        target_latin_to_common[latin] = common

    # ...for each row in the custom taxonomy list


    ##%% Read taxonomy file

    with open(speciesnet_taxonomy_file,'r') as f:
        speciesnet_taxonomy_list = f.readlines()
    speciesnet_taxonomy_list = [s.strip() for s in \
                                speciesnet_taxonomy_list if len(s.strip()) > 0]

    # Maps the latin name of every taxon to the corresponding full taxon string
    #
    # For species, the key is a binomial name
    speciesnet_latin_name_to_taxon_string = {}
    speciesnet_common_name_to_taxon_string = {}

    def _insert_taxonomy_string(s):

        tokens = s.split(';')
        assert len(tokens) == 7, 'Illegal taxonomy string {}'.format(s)

        guid = tokens[0] # noqa
        class_name = tokens[1]
        order = tokens[2]
        family = tokens[3]
        genus = tokens[4]
        species = tokens[5]
        common_name = tokens[6]

        if len(class_name) == 0:
            assert common_name in ('animal','vehicle','blank'), \
                'Illegal common name {}'.format(common_name)
            return

        if len(species) > 0:
            assert all([len(s) > 0 for s in [genus,family,order]]), \
                'Higher-level taxa missing for {}: {},{},{}'.format(s,genus,family,order)
            binomial_name = genus + ' ' + species
            if binomial_name not in speciesnet_latin_name_to_taxon_string:
                speciesnet_latin_name_to_taxon_string[binomial_name] = s
        elif len(genus) > 0:
            assert all([len(s) > 0 for s in [family,order]]), \
                'Higher-level taxa missing for {}: {},{}'.format(s,family,order)
            if genus not in speciesnet_latin_name_to_taxon_string:
                speciesnet_latin_name_to_taxon_string[genus] = s
        elif len(family) > 0:
            assert len(order) > 0, \
                'Higher-level taxa missing for {}: {}'.format(s,order)
            if family not in speciesnet_latin_name_to_taxon_string:
                speciesnet_latin_name_to_taxon_string[family] = s
        elif len(order) > 0:
            if order not in speciesnet_latin_name_to_taxon_string:
                speciesnet_latin_name_to_taxon_string[order] = s
        else:
            if class_name not in speciesnet_latin_name_to_taxon_string:
                speciesnet_latin_name_to_taxon_string[class_name] = s

        if len(common_name) > 0:
            if common_name not in speciesnet_common_name_to_taxon_string:
                speciesnet_common_name_to_taxon_string[common_name] = s

    for s in speciesnet_taxonomy_list:

        _insert_taxonomy_string(s)


    ##%% Make sure all parent taxa are represented in the taxonomy

    # In theory any taxon that appears as the parent of another taxon should
    # also be in the taxonomy, but this isn't always true, so we fix it here.
    new_taxon_string_to_missing_tokens = defaultdict(list)

    # While we're making this loop, also see whether we need to store any custom
    # common name mappings based on the taxonomy list.
    speciesnet_latin_name_to_output_common_name = {}

    # latin_name = next(iter(speciesnet_latin_name_to_taxon_string.keys()))
    for latin_name in speciesnet_latin_name_to_taxon_string.keys():

        if latin_name in target_latin_to_common:
            speciesnet_latin_name_to_output_common_name[latin_name] = \
                target_latin_to_common[latin_name]

        if 'no cv result' in latin_name:
            continue

        taxon_string = speciesnet_latin_name_to_taxon_string[latin_name]
        tokens = taxon_string.split(';')

        # Don't process GUID, species, or common name
        # i_token = 6
        for i_token in range(1,len(tokens)-2):

            test_token = tokens[i_token]
            if len(test_token) == 0:
                continue

            # Do we need to make up a taxon for this token?
            if test_token not in speciesnet_latin_name_to_taxon_string:

                new_tokens = [''] * 7
                new_tokens[0] = 'fake_guid'
                for i_copy_token in range(1,i_token+1):
                    new_tokens[i_copy_token] = tokens[i_copy_token]
                new_tokens[-1] = test_token + ' species'
                assert new_tokens[-2] == '', \
                    'Illegal taxonomy string {}'.format(taxon_string)
                new_taxon_string = ';'.join(new_tokens)
                # assert new_taxon_string not in new_taxon_strings
                new_taxon_string_to_missing_tokens[new_taxon_string].append(test_token)

        # ...for each token

    # ...for each taxon

    new_taxon_string_to_missing_tokens = \
        sort_dictionary_by_key(new_taxon_string_to_missing_tokens)

    if verbose:

        print(f'Found {len(new_taxon_string_to_missing_tokens)} taxa that need to be inserted to ' + \
              'make the taxonomy valid, showing only mammals and birds here:\n')

        for taxon_string in new_taxon_string_to_missing_tokens:
            if 'mammalia' not in taxon_string and 'aves' not in taxon_string:
                continue
            missing_taxa = ','.join(new_taxon_string_to_missing_tokens[taxon_string])
            print('{} ({})'.format(taxon_string,missing_taxa))

    for new_taxon_string in new_taxon_string_to_missing_tokens:
        _insert_taxonomy_string(new_taxon_string)


    ##%% Make sure all taxa on the allow-list are in the taxonomy

    n_failed_mappings = 0

    for target_taxon_latin_name in target_latin_to_common.keys():
        if target_taxon_latin_name not in speciesnet_latin_name_to_taxon_string:
            common_name = target_latin_to_common[target_taxon_latin_name]
            s = '{} ({}) not in speciesnet taxonomy'.format(
                target_taxon_latin_name,common_name)
            if common_name in speciesnet_common_name_to_taxon_string:
                s += ' (common name maps to {})'.format(
                    speciesnet_common_name_to_taxon_string[common_name])
            print(s)
            n_failed_mappings += 1

    if n_failed_mappings > 0:
        raise ValueError('Cannot continue with taxonomic restriction')


    ##%% For the allow-list, map each parent taxon to a set of allowable child taxa

    # Maps parent names to all allowed child names, or None if this is the
    # lowest-level allowable taxon on this path
    allowed_parent_taxon_to_child_taxa = defaultdict(set)

    # latin_name = next(iter(target_latin_to_common.keys()))
    for latin_name in target_latin_to_common:

        taxon_string = speciesnet_latin_name_to_taxon_string[latin_name]
        tokens = taxon_string.split(';')
        assert len(tokens) == 7, \
            'Illegal taxonomy string {}'.format(taxon_string)

        # Remove GUID and common mame
        #
        # This is now always class/order/family/genus/species
        tokens = tokens[1:-1]

        child_taxon = None

        # If this is a species
        if len(tokens[-1]) > 0:
            binomial_name = tokens[-2] + ' ' + tokens[-1]
            assert binomial_name == latin_name, \
                'Binomial/latin mismatch: {} vs {}'.format(binomial_name,latin_name)
            # If this already exists, it should only allow "None"
            if binomial_name in allowed_parent_taxon_to_child_taxa:
                assert len(allowed_parent_taxon_to_child_taxa[binomial_name]) == 1, \
                    'Species-level entry {} has multiple children'.format(binomial_name)
                assert None in allowed_parent_taxon_to_child_taxa[binomial_name], \
                    'Species-level entry {} has non-None children'.format(binomial_name)
            allowed_parent_taxon_to_child_taxa[binomial_name].add(None)
            child_taxon = binomial_name

        # The first level that can ever be a parent taxon is the genus level
        parent_token_index = len(tokens) - 2

        # Walk up from genus to family
        while(parent_token_index >= 0):

            # "None" is our leaf node marker, we should never have ''
            if child_taxon is not None:
                assert len(child_taxon) > 0

            parent_taxon = tokens[parent_token_index]

            # Don't create entries for blank taxa
            if (len(parent_taxon) > 0):

                create_child = True

                # This is the lowest-level taxon in this entry
                if (child_taxon is None):

                    # ...but we don't want to remove existing children from any parents
                    if (parent_taxon in allowed_parent_taxon_to_child_taxa) and \
                       (len(allowed_parent_taxon_to_child_taxa[parent_taxon]) > 0):
                        if verbose:
                            existing_children_string = str(allowed_parent_taxon_to_child_taxa[parent_taxon])
                            print('Not creating empty child for parent {} (already has children {})'.format(
                                parent_taxon,existing_children_string))
                        create_child = False

                # If we're adding a new child entry, clear out any leaf node markers
                else:

                    if (parent_taxon in allowed_parent_taxon_to_child_taxa) and \
                       (None in allowed_parent_taxon_to_child_taxa[parent_taxon]):

                        assert len(allowed_parent_taxon_to_child_taxa[parent_taxon]) == 1, \
                            'Illlegal parent/child configuration'

                        if verbose:
                            print('Un-marking parent {} as a leaf node because of child {}'.format(
                                parent_taxon,child_taxon))

                        allowed_parent_taxon_to_child_taxa[parent_taxon] = set()

                if create_child:
                    allowed_parent_taxon_to_child_taxa[parent_taxon].add(child_taxon)

                # If we haven't hit a non-empty taxon yet, don't update "child_taxon"
                assert len(parent_taxon) > 0
                child_taxon = parent_taxon

            # ...if we have a non-empty taxon

            parent_token_index -= 1

        # ...for each taxonomic level

    # ...for each allowed latin name

    allowed_parent_taxon_to_child_taxa = \
        sort_dictionary_by_key(allowed_parent_taxon_to_child_taxa)

    for parent_taxon in allowed_parent_taxon_to_child_taxa:
        # "None" should only ever appear alone; this marks a leaf node with no children
        if None in allowed_parent_taxon_to_child_taxa[parent_taxon]:
            assert len(allowed_parent_taxon_to_child_taxa[parent_taxon]) == 1, \
                '"None" should only appear alone in a child taxon list'


    ##%% If we were just validating the custom taxa file, we're done

    if input_file is None:
        print('Finished validating custom taxonomy list')
        return


    ##%% Map all predictions that exist in this dataset...

    # ...to the prediction we should generate.

    with open(input_file,'r') as f:
        input_data = json.load(f)

    input_category_id_to_common_name = input_data['classification_categories'] #noqa
    input_category_id_to_taxonomy_string = \
        input_data['classification_category_descriptions']

    input_category_id_to_output_taxon_string = {}

    # input_category_id = next(iter(input_category_id_to_taxonomy_string.keys()))
    for input_category_id in input_category_id_to_taxonomy_string.keys():

        input_taxon_string = input_category_id_to_taxonomy_string[input_category_id]
        input_taxon_tokens = input_taxon_string.split(';')
        assert len(input_taxon_tokens) == 7, \
            'Illegal taxonomy string: {}'.format(input_taxon_string)

        # Don't mess with blank/no-cv-result/human (or "animal", which is really "unknown")
        if (not is_taxonomic_prediction_string(input_taxon_string)) or \
           (input_taxon_string == human_prediction_string):
            if verbose:
                print('Not messing with non-taxonomic category {}'.format(input_taxon_string))
            input_category_id_to_output_taxon_string[input_category_id] = \
                input_taxon_string
            continue

        # Don't mess with protected categories
        common_name = input_taxon_tokens[-1]

        if (protected_common_names is not None) and \
            (common_name in protected_common_names):
            if verbose:
                print('Not messing with protected category {}:\n{}'.format(
                    common_name,input_taxon_string))
            input_category_id_to_output_taxon_string[input_category_id] = \
                input_taxon_string
            continue

        # Remove GUID and common mame

        # This is now always class/order/family/genus/species
        input_taxon_tokens = input_taxon_tokens[1:-1]

        test_index = len(input_taxon_tokens) - 1
        target_taxon = None

        # Start at the species level, and see whether each taxon is allowed
        while((test_index >= 0) and (target_taxon is None)):

            # Species are represented as binomial names
            if (test_index == (len(input_taxon_tokens) - 1)) and \
                (len(input_taxon_tokens[-1]) > 0):
                test_taxon_name = \
                    input_taxon_tokens[-2] + ' ' + input_taxon_tokens[-1]
            else:
                test_taxon_name = input_taxon_tokens[test_index]

            # If we haven't yet found the level at which this taxon is non-empty,
            # keep going up
            if len(test_taxon_name) == 0:
                test_index -= 1
                continue

            assert test_taxon_name in speciesnet_latin_name_to_taxon_string, \
                '{} should be a substring of {}'.format(test_taxon_name,
                                                        speciesnet_latin_name_to_taxon_string)

            # Is this taxon allowed according to the custom species list?
            if test_taxon_name in allowed_parent_taxon_to_child_taxa:

                allowed_child_taxa = allowed_parent_taxon_to_child_taxa[test_taxon_name]
                assert allowed_child_taxa is not None, \
                    'allowed_child_taxa should not be None: {}'.format(test_taxon_name)

                # If this is the lowest-level allowable token or there is not a
                # unique child, don't walk any further, even if walking down
                # is enabled.
                if None in allowed_child_taxa:
                    assert len(allowed_child_taxa) == 1, \
                        '"None" should not be listed as a child taxa with other child taxa'

                if (None in allowed_child_taxa) or (len(allowed_child_taxa) > 1):
                    target_taxon = test_taxon_name
                elif not allow_walk_down:
                    target_taxon = test_taxon_name
                else:
                    # If there's a unique child, walk back *down* the allowable
                    # taxa until we run out of unique children
                    while ((next(iter(allowed_child_taxa)) is not None) and \
                          (len(allowed_child_taxa) == 1)):
                        candidate_taxon = next(iter(allowed_child_taxa))
                        assert candidate_taxon in allowed_parent_taxon_to_child_taxa, \
                            '{} should be a subset of {}'.format(
                                candidate_taxon,allowed_parent_taxon_to_child_taxa)
                        assert candidate_taxon in speciesnet_latin_name_to_taxon_string, \
                            '{} should be a subset of {}'.format(
                                candidate_taxon,speciesnet_latin_name_to_taxon_string)
                        allowed_child_taxa = \
                            allowed_parent_taxon_to_child_taxa[candidate_taxon]
                    target_taxon = candidate_taxon

            # ...if this is an allowed taxon

            test_index -= 1

        # ...for each token

        if target_taxon is None:
            output_taxon_string = animal_prediction_string
        else:
            output_taxon_string = speciesnet_latin_name_to_taxon_string[target_taxon]
        input_category_id_to_output_taxon_string[input_category_id] = output_taxon_string

    # ...for each category (mapping input category IDs to output taxon strings)


    ##%% Map input category IDs to output category IDs

    speciesnet_taxon_string_to_latin_name = \
        invert_dictionary(speciesnet_latin_name_to_taxon_string)

    input_category_id_to_output_category_id = {}
    output_taxon_string_to_category_id = {}
    output_category_id_to_common_name = {}

    for input_category_id in input_category_id_to_output_taxon_string:

        output_taxon_string = \
            input_category_id_to_output_taxon_string[input_category_id]

        output_common_name = output_taxon_string.split(';')[-1]

        # Possibly substitute a custom common name
        if output_taxon_string in speciesnet_taxon_string_to_latin_name:

            speciesnet_latin_name = speciesnet_taxon_string_to_latin_name[output_taxon_string]

            if speciesnet_latin_name in speciesnet_latin_name_to_output_common_name:
                custom_common_name = speciesnet_latin_name_to_output_common_name[speciesnet_latin_name]
                if custom_common_name != output_common_name:
                    if verbose:
                        print('Substituting common name {} for {}'.format(custom_common_name,output_common_name))
                    output_common_name = custom_common_name

        # Do we need to create a new output category?
        if output_taxon_string not in output_taxon_string_to_category_id:
            output_category_id = str(len(output_taxon_string_to_category_id))
            output_taxon_string_to_category_id[output_taxon_string] = \
                output_category_id
            output_category_id_to_common_name[output_category_id] = \
                output_common_name
        else:
            output_category_id = \
                output_taxon_string_to_category_id[output_taxon_string]

        input_category_id_to_output_category_id[input_category_id] = \
            output_category_id

        # Sometimes-useful debug printouts
        if False:
            original_common_name = \
              input_category_id_to_common_name[input_category_id]
            original_taxon_string = \
                input_category_id_to_taxonomy_string[input_category_id]
            print('Mapping {} ({}) to:\n{} ({})\n'.format(
                original_common_name,original_taxon_string,
                output_common_name,output_taxon_string))

    # ...for each category (mapping input category IDs to output category IDs)


    ##%% Remap all category labels

    assert len(set(output_taxon_string_to_category_id.keys())) == \
           len(set(output_taxon_string_to_category_id.values())), \
           'Category ID/value non-uniqueness error'

    output_category_id_to_taxon_string = \
        invert_dictionary(output_taxon_string_to_category_id)

    with open(input_file,'r') as f:
        output_data = json.load(f)

    classification_descriptions = None
    if 'classification_category_descriptions' in output_data:
        classification_descriptions = output_data['classification_category_descriptions']

    for im in tqdm(output_data['images']):

        if 'detections' not in im or im['detections'] is None:
            continue

        # Possibly prepare a pre-filtering description
        pre_filtering_description = None
        if classification_descriptions is not None and add_pre_filtering_description:
            category_to_count = count_detections_by_classification_category(im['detections'])
            pre_filtering_description = \
                get_classification_description_string(category_to_count,classification_descriptions)
            im['pre_filtering_description'] = pre_filtering_description

        for det in im['detections']:
            if 'classifications' in det:
                for classification in det['classifications']:
                    classification[0] = \
                        input_category_id_to_output_category_id[classification[0]]

    # ...for each image

    output_data['classification_categories'] = output_category_id_to_common_name
    output_data['classification_category_descriptions'] = \
        output_category_id_to_taxon_string


    ##%% Write output

    write_json(output_file,output_data)


# ...def restrict_to_taxa_list(...)
