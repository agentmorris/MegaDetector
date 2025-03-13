"""

classification_postprocessing.py

Functions for postprocessing species classification results, particularly:
    
* Smoothing results within a sequence (a sequence that looks like deer/deer/deer/elk/deer/deer
  is really just a deer)
* Smoothing results within an image (an image with 700 cows and one deer is really just 701
  cows)
  
"""

#%% Constants and imports

import json
import copy

from collections import defaultdict
from tqdm import tqdm

from megadetector.utils.ct_utils import is_list_sorted
from megadetector.utils.wi_utils import clean_taxonomy_string


#%% Options classes

class ClassificationSmoothingOptionsImageLevel:
    """
    Options used to parameterize smooth_classification_results_image_level()
    """

    def __init__(self):
        
        #: How many detections do we need in a dominant category to overwrite 
        #: non-dominant classifications?
        self.min_detections_to_overwrite_secondary = 4
    
        #: Even if we have a dominant class, if a non-dominant class has at least 
        #: this many classifications in an image, leave them alone.
        #:
        #: If this is <= 1, we won't replace non-dominant, non-other classes
        #: with the dominant class, even if there are 900 cows and 1 deer.
        self.max_detections_nondominant_class = 1
    
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
        
        #: Should we record information about the state of labels prior to smoothing?
        self.add_pre_smoothing_description = True
        

class ClassificationSmoothingOptionsSequenceLevel:
    """
    Options used to parameterize smooth_classification_results_sequence_level()
    """
    
    def __init__(self):
        
        #: Only process detections in this category
        self.animal_detection_category = '1'
        
        #: Treat category names on this list as "other", which can be flipped to common
        #: categories.
        self.other_category_names = set(['other'])
        
        #: These are the only classes to which we're going to switch "other" classifications.
        #:
        #: Example:
        #:
        #: ['deer','elk','cow','canid','cat','bird','bear']
        self.category_names_to_smooth_to = None
        
        #: Only switch classifications to the dominant class if we see the dominant class at least
        #: this many times
        self.min_dominant_class_classifications_above_threshold_for_class_smoothing = 5 # 2
        
        #: If we see more than this many of a class that are above threshold, don't switch those
        #: classifications to the dominant class.
        self.max_secondary_class_classifications_above_threshold_for_class_smoothing = 5
        
        #: If the ratio between a dominant class and a secondary class count is greater than this, 
        #: regardless of the secondary class count, switch those classifications (i.e., ignore
        #: max_secondary_class_classifications_above_threshold_for_class_smoothing).
        #:
        #: This may be different for different dominant classes, e.g. if we see lots of cows, they really
        #: tend to be cows.  Less so for canids, so we set a higher "override ratio" for canids.
        #:
        #: Should always include a "None" category as the default ratio.
        #:
        #: Example:
        #:
        #: {'cow':2,None:3}
        self.min_dominant_class_ratio_for_secondary_override_table = {None:3}
        
        #: If there are at least this many classifications for the dominant class in a sequence,
        #: regardless of what that class is, convert all 'other' classifications (regardless of 
        #: confidence) to that class.
        self.min_dominant_class_classifications_above_threshold_for_other_smoothing = 3 # 2
        
        #: If there are at least this many classifications for the dominant class in a sequence,
        #: regardless of what that class is, classify all previously-unclassified detections
        #: as that class.
        self.min_dominant_class_classifications_above_threshold_for_unclassified_smoothing = 3 # 2
        
        #: Only count classifications above this confidence level when determining the dominant
        #: class, and when deciding whether to switch other classifications.
        self.classification_confidence_threshold = 0.6
        
        #: Confidence values to use when we change a detection's classification (the
        #: original confidence value is irrelevant at that point) (for the "other" class)
        self.flipped_other_confidence_value = 0.6
        
        #: Confidence values to use when we change a detection's classification (the
        #: original confidence value is irrelevant at that point) (for all non-other classes)
        self.flipped_class_confidence_value = 0.6
        
        #: Confidence values to use when we change a detection's classification (the
        #: original confidence value is irrelevant at that point) (for previously unclassified detections)
        self.flipped_unclassified_confidence_value = 0.6
        
        #: Only flip the class label unclassified detections if the detection confidence exceeds this threshold
        self.min_detection_confidence_for_unclassified_flipping = 0.15
        
        #: Only relevant when MegaDetector results are supplied as a dict rather than a file; determines
        #: whether smoothing happens in place.
        self.modify_in_place = True        
        
# ...class ClassificationSmoothingOptionsSequenceLevel()


#%% Utility functions

def _count_detections_by_category(detections,options):
    """
    Count the number of instances of each category in the detections list
    [detections] that have an above-threshold detection.  Sort results in descending 
    order by count.  Returns a dict mapping category ID --> count.  If no detections
    are above threshold, returns an empty dict.
    
    Assumes that if the 'classifications' field is present for a detection, it has
    length 1, i.e. that non-top classifications have already been removed.
    """
    
    category_to_count = defaultdict(int)
    
    for det in detections:
        if ('classifications' in det) and (det['conf'] >= options.detection_confidence_threshold):
            assert len(det['classifications']) == 1
            c = det['classifications'][0]
            if c[1] >= options.classification_confidence_threshold:
                category_to_count[c[0]] += 1            
                    
    category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    return category_to_count


#%% Image-level smoothing

def _get_description_string(category_to_count,classification_descriptions):
    """
    Return a string summarizing the image content according to [category_to_count].
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
    
    
def _smooth_single_image(im,options,
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
    
    ## Argument validation
    
    if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
        return
    
    detections = im['detections']
        
    # Useful debug snippet
    #
    # if 'filename' in im['file']:
    #    import pdb; pdb.set_trace()    
    
    
    ## Count the number of instances of each category in this image
    
    category_to_count = _count_detections_by_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    # _get_description_string(category_to_count, classification_descriptions)
        
    if options.add_pre_smoothing_description:
        im['pre_smoothing_description'] = \
            _get_description_string(category_to_count, classification_descriptions)
        
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
    
    
    ## Possibly change "other" classifications to the most common category
    
    # ...if the dominant category is not an "other" category.
    
    n_other_classifications_changed_this_image = 0
    
    # If we have at least *min_detections_to_overwrite_other* in a category that isn't
    # "other", change all "other" classifications to that category
    if (max_count >= options.min_detections_to_overwrite_other) and \
        (most_common_category not in other_category_ids):
        
        for det in detections:
            
            if ('classifications' in det) and \
                (det['conf'] >= options.detection_confidence_threshold): 
                
                assert len(det['classifications']) == 1
                c = det['classifications'][0]
                    
                if c[1] >= options.classification_confidence_threshold and \
                    c[0] in other_category_ids:
                        
                    n_other_classifications_changed_this_image += 1
                    c[0] = most_common_category
                                        
            # ...if there are classifications for this detection
            
        # ...for each detection
                
    # ...if we should overwrite all "other" classifications
    
    
    ## Re-count
    
    category_to_count = _count_detections_by_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    keys = list(category_to_count.keys())
    max_count = category_to_count[keys[0]]    
    most_common_category = keys[0]
    del keys
    
    ## At this point, we know we have a dominant category; possibly change 
    ## above-threshold classifications to that category.
    
    n_detections_flipped_this_image = 0
    
    # Don't do this if the most common category is an "other" category
    if most_common_category not in other_category_ids:
        
        # det = detections[0]
        for det in detections:
            
            if ('classifications' in det) and \
                (det['conf'] >= options.detection_confidence_threshold):
                
                assert len(det['classifications']) == 1
                c = det['classifications'][0]
                
                if (c[1] >= options.classification_confidence_threshold) and \
                   (c[0] != most_common_category) and \
                   (max_count > category_to_count[c[0]]) and \
                   (max_count > options.min_detections_to_overwrite_secondary) and \
                   (category_to_count[c[0]] <= options.max_detections_nondominant_class):
                        
                    c[0] = most_common_category
                    n_detections_flipped_this_image += 1
                                        
            # ...if there are classifications for this detection
            
        # ...for each detection

    # ...if the dominant category is legit    
    
    
    ## Re-count
    
    category_to_count = _count_detections_by_category(detections, options)
    # _print_counts_with_names(category_to_count,classification_descriptions)
    keys = list(category_to_count.keys())
    max_count = category_to_count[keys[0]]    
    most_common_category = keys[0]
    del keys
    
    
    ## Possibly collapse higher-level taxonomic predictions down to lower levels
    
    n_taxonomic_changes_this_image = 0
        
    if (classification_descriptions_clean is not None) and \
        (len(classification_descriptions_clean) > 0) and \
        (len(category_to_count) > 1):
    
        # det = detections[3]
        for det in detections:
            
            if ('classifications' in det) and \
                (det['conf'] >= options.detection_confidence_threshold):
                
                assert len(det['classifications']) == 1
                c = det['classifications'][0]
                
                # Don't bother with any classifications below the confidence threshold
                if c[1] < options.classification_confidence_threshold:
                    continue
    
                category_id_this_classification = c[0]
                assert category_id_this_classification in category_to_count
                
                category_description_this_classification = \
                    classification_descriptions_clean[category_id_this_classification]
                
                count_this_category = category_to_count[category_id_this_classification]
                
                # Are there any child categories with more classifications?
                for category_id_of_candidate_child in category_to_count.keys():
                    
                    if category_id_of_candidate_child == category_id_this_classification:
                        continue
                    
                    category_description_candidate_child = \
                        classification_descriptions_clean[category_id_of_candidate_child]
                    
                    # An empty description corresponds to "animal", which can never
                    # be a child of another category.
                    if len(category_description_candidate_child) == 0:
                        continue
                        
                    # Parent/child taxonomic relationships are defined by a substring relationship
                    is_child = category_description_this_classification in \
                        category_description_candidate_child
                    if not is_child:
                        continue
                    
                    child_category_count = category_to_count[category_id_of_candidate_child]
                    if child_category_count > count_this_category:
                        
                        c[0] = category_id_of_candidate_child
                        n_taxonomic_changes_this_image += 1
                    
                # ...for each classification
                
            # ...if there are classifications for this detection
            
        # ...for each detection
        
    # ...if we have taxonomic information available
    
    
    return {'n_other_classifications_changed_this_image':n_other_classifications_changed_this_image,
            'n_detections_flipped_this_image':n_detections_flipped_this_image,
            'n_taxonomic_changes_this_image':n_taxonomic_changes_this_image}

# ...def smooth_single_image


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
        input_file (str): MegaDetector-formatted classification results file to smooth
        output_file (str, optional): .json file to write smoothed results
        options (ClassificationSmoothingOptionsImageLevel, optional): see 
          ClassificationSmoothingOptionsImageLevel for details.
            
    Returns:
        dict: MegaDetector-results-formatted dict, identical to what's written to
        [output_file] if [output_file] is not None.
    """
        
    if options is None:
        options = ClassificationSmoothingOptionsImageLevel()
        
    with open(input_file,'r') as f:
        print('Loading results from:\n{}'.format(input_file))
        d = json.load(f)
        
    category_name_to_id = {d['classification_categories'][k]:k for k in d['classification_categories']}
    other_category_ids = []
    for s in options.other_category_names:
        if s in category_name_to_id:
            other_category_ids.append(category_name_to_id[s])
        else:
            print('Warning: "other" category {} not present in file {}'.format(
                s,input_file))
        
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
    
    
    ## Clean up classification descriptions so we can test taxonomic relationships
    ## by substring testing.
    
    classification_descriptions_clean = None
    
    if 'classification_category_descriptions' in d:
        classification_descriptions = d['classification_category_descriptions']
        classification_descriptions_clean = {}
        # category_id = next(iter(classification_descriptions))
        for category_id in classification_descriptions:            
            classification_descriptions_clean[category_id] = \
                clean_taxonomy_string(classification_descriptions[category_id]).strip(';').lower()
    
    
    ## Do the actual smoothing
    
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

    for im in d['images']:
        if 'failure' in im and im['failure'] is None:
            del im['failure']
            
    if output_file is not None:    
        print('Writing results after image-level smoothing to:\n{}'.format(output_file))
        with open(output_file,'w') as f:
            json.dump(d,f,indent=1)        

    return d

# ...def smooth_classification_results_image_level(...)


#%% Sequence-level smoothing

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
            
    
def _top_classifications_for_sequence(images_this_sequence,filename_to_results,options):
    """
    Return all top-1 animal classifications for every detection in this 
    sequence, regardless of  confidence

    May modify [images_this_sequence] (removing non-top-1 classifications)
    """
    
    classifications_this_sequence = []

    # im = images_this_sequence[0]
    for im in images_this_sequence:
        
        fn = im['file_name']
        results_this_image = filename_to_results[fn]
        
        if results_this_image['detections'] is None:
            continue
        
        # det = results_this_image['detections'][0]
        for det in results_this_image['detections']:
            
            # Only process animal detections
            if det['category'] != options.animal_detection_category:
                continue
            
            # Only process detections with classification information
            if 'classifications' not in det:
                continue
            
            # We only care about top-1 classifications, remove everything else
            if len(det['classifications']) > 1:
                
                # Make sure the list of classifications is already sorted by confidence
                classification_confidence_values = [c[1] for c in det['classifications']]
                assert is_list_sorted(classification_confidence_values,reverse=True)
                
                # ...and just keep the first one
                det['classifications'] = [det['classifications'][0]]
                
            # Confidence values should be sorted within a detection; verify this, and ignore 
            top_classification = det['classifications'][0]
            
            classifications_this_sequence.append(top_classification)
    
        # ...for each detection in this image
        
    # ...for each image in this sequence

    return classifications_this_sequence

# ..._top_classifications_for_sequence()


def _count_above_threshold_classifications(classifications_this_sequence,options):
    """
    Given a list of classification objects (tuples), return a dict mapping
    category IDs to the count of above-threshold classifications.
    
    This dict's keys will be sorted in descending order by frequency.
    """
    
    # Count above-threshold classifications in this sequence
    category_to_count = defaultdict(int)
    for c in classifications_this_sequence:
        if c[1] >= options.classification_confidence_threshold:
            category_to_count[c[0]] += 1
    
    # Sort the dictionary in descending order by count
    category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    keys_sorted_by_frequency = list(category_to_count.keys())
        
    # Handle a quirky special case: if the most common category is "other" and 
    # it's "tied" with the second-most-common category, swap them.
    if (options.other_category_names is not None) and (len(options.other_category_names) > 0):
        if (len(keys_sorted_by_frequency) > 1) and \
            (keys_sorted_by_frequency[0] in options.other_category_names) and \
            (keys_sorted_by_frequency[1] not in options.other_category_names) and \
            (category_to_count[keys_sorted_by_frequency[0]] == \
             category_to_count[keys_sorted_by_frequency[1]]):
                keys_sorted_by_frequency[1], keys_sorted_by_frequency[0] = \
                    keys_sorted_by_frequency[0], keys_sorted_by_frequency[1]

    sorted_category_to_count = {}    
    for k in keys_sorted_by_frequency:
        sorted_category_to_count[k] = category_to_count[k]
        
    return sorted_category_to_count

# ...def _count_above_threshold_classifications()
 
   
def _sort_images_by_time(images):
    """
    Returns a copy of [images], sorted by the 'datetime' field (ascending).
    """
    return sorted(images, key = lambda im: im['datetime'])        
    

def _get_first_key_from_sorted_dictionary(di):
    if len(di) == 0:
        return None
    return next(iter(di.items()))[0]


def _get_first_value_from_sorted_dictionary(di):
    if len(di) == 0:
        return None
    return next(iter(di.items()))[1]


def smooth_classification_results_sequence_level(md_results,
                                                 cct_sequence_information,
                                                 output_file=None,
                                                 options=None):
    """
    Smooth classifications at the sequence level for all results in the MD-formatted results
    file [md_results_file], optionally writing a new set of results to [output_file].
    
    This function generally expresses the notion that a sequence that looks like
    deer/deer/deer/elk/deer/deer/deer/deer is really just a deer.
    
    Args:
        md_results (str or dict): MegaDetector-formatted classification results file to smooth
          (or already-loaded results).  If you supply a dict, it's modified in place by default, but
          a copy can be forced by setting options.modify_in_place=False.
        cct_sequence_information (str, dict, or list): COCO Camera Traps file containing sequence IDs for
          each image (or an already-loaded CCT-formatted dict, or just the 'images' list from a CCT dict).
        output_file (str, optional): .json file to write smoothed results
        options (ClassificationSmoothingOptionsSequenceLevel, optional): see 
          ClassificationSmoothingOptionsSequenceLevel for details.
            
    Returns:
        dict: MegaDetector-results-formatted dict, identical to what's written to
        [output_file] if [output_file] is not None.
    """
    
    if options is None:
        options = ClassificationSmoothingOptionsSequenceLevel()
    
    if options.category_names_to_smooth_to is None:
        options.category_names_to_smooth_to = []
    
    if options.other_category_names is None:
        options.other_category_names = []
        
    assert None in options.min_dominant_class_ratio_for_secondary_override_table, \
        'Oops, it looks like you removed the default (None) key from ' + \
            'options.min_dominant_class_ratio_for_secondary_override_table'
    
    if isinstance(md_results,str):
        print('Loading MD results from {}'.format(md_results))
        with open(md_results,'r') as f:
            md_results = json.load(f)
    else:
        assert isinstance(md_results,dict)
        if not options.modify_in_place:
            print('Copying MD results instead of modifying in place')
            md_results = copy.deepcopy(md_results)
        else:
            print('Smoothing MD results in place')
    
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
        
    
    ##%% Make a list of images appearing at each location
    
    sequence_to_images = defaultdict(list)
    
    # im = image_info[0]
    for im in tqdm(image_info):
        sequence_to_images[im['seq_id']].append(im)
    
    all_sequences = list(sorted(sequence_to_images.keys()))
    
    
    ##%% Load classification results
    
    # Map each filename to classification results for that file
    filename_to_results = {}
    
    for im in tqdm(md_results['images']):
        filename_to_results[im['file'].replace('\\','/')] = im
    
    
    ##%% Smooth classification results over sequences (prep)
    
    classification_category_id_to_name = md_results['classification_categories']
    classification_category_name_to_id = {v: k for k, v in classification_category_id_to_name.items()}
    
    class_names = list(classification_category_id_to_name.values())
    
    assert(md_results['detection_categories'][options.animal_detection_category] == 'animal')
    
    other_category_ids = set([classification_category_name_to_id[s] for s in options.other_category_names])
    
    category_ids_to_smooth_to = set([classification_category_name_to_id[s] for s in options.category_names_to_smooth_to])
    assert all([s in class_names for s in options.category_names_to_smooth_to])    
    
    
    ##%% Smooth classifications at the sequence level (main loop)
    
    n_other_flips = 0
    n_classification_flips = 0
    n_unclassified_flips = 0
    
    # Break if this token is contained in a filename (set to None for normal operation)
    debug_fn = None
    
    # i_sequence = 0; seq_id = all_sequences[i_sequence]
    for i_sequence,seq_id in tqdm(enumerate(all_sequences),total=len(all_sequences)):
        
        images_this_sequence = sequence_to_images[seq_id]
        
        # Count top-1 classifications in this sequence (regardless of confidence)
        classifications_this_sequence = _top_classifications_for_sequence(images_this_sequence,
                                                                          filename_to_results,
                                                                          options)
        
        # Handy debugging code for looking at the numbers for a particular sequence
        for im in images_this_sequence:
            if debug_fn is not None and debug_fn in im['file_name']:
                raise ValueError('')
                 
        if len(classifications_this_sequence) == 0:
            continue
        
        # Count above-threshold classifications for each category
        sorted_category_to_count = _count_above_threshold_classifications(
            classifications_this_sequence,options)
        
        if len(sorted_category_to_count) == 0:
            continue
        
        max_count = _get_first_value_from_sorted_dictionary(sorted_category_to_count)    
        dominant_category_id = _get_first_key_from_sorted_dictionary(sorted_category_to_count)
        
        # If our dominant category ID isn't something we want to smooth to, 
        # don't mess around with this sequence
        if dominant_category_id not in category_ids_to_smooth_to:
            continue
            
        
        ## Smooth "other" classifications ##
        
        if max_count >= options.min_dominant_class_classifications_above_threshold_for_other_smoothing:        
            for c in classifications_this_sequence:           
                if c[0] in other_category_ids:
                    n_other_flips += 1
                    c[0] = dominant_category_id
                    c[1] = options.flipped_other_confidence_value
    
    
        # By not re-computing "max_count" here, we are making a decision that the count used
        # to decide whether a class should overwrite another class does not include any "other"
        # classifications we changed to be the dominant class.  If we wanted to include those...
        # 
        # sorted_category_to_count = count_above_threshold_classifications(classifications_this_sequence)
        # max_count = get_first_value_from_sorted_dictionary(sorted_category_to_count)    
        # assert dominant_category_id == get_first_key_from_sorted_dictionary(sorted_category_to_count)
        
        
        ## Smooth non-dominant classes ##
        
        if max_count >= options.min_dominant_class_classifications_above_threshold_for_class_smoothing:
            
            # Don't flip classes to the dominant class if they have a large number of classifications
            category_ids_not_to_flip = set()
            
            for category_id in sorted_category_to_count.keys():
                secondary_class_count = sorted_category_to_count[category_id]
                dominant_to_secondary_ratio = max_count / secondary_class_count
                
                # Don't smooth over this class if there are a bunch of them, and the ratio
                # if primary to secondary class count isn't too large
                
                # Default ratio
                ratio_for_override = options.min_dominant_class_ratio_for_secondary_override_table[None]
                
                # Does this dominant class have a custom ratio?
                dominant_category_name = classification_category_id_to_name[dominant_category_id]
                if dominant_category_name in options.min_dominant_class_ratio_for_secondary_override_table:
                    ratio_for_override = \
                        options.min_dominant_class_ratio_for_secondary_override_table[dominant_category_name]
                        
                if (dominant_to_secondary_ratio < ratio_for_override) and \
                    (secondary_class_count > \
                     options.max_secondary_class_classifications_above_threshold_for_class_smoothing):
                    category_ids_not_to_flip.add(category_id)
                    
            for c in classifications_this_sequence:
                if c[0] not in category_ids_not_to_flip and c[0] != dominant_category_id:
                    c[0] = dominant_category_id
                    c[1] = options.flipped_class_confidence_value
                    n_classification_flips += 1
            
            
        ## Smooth unclassified detections ##
            
        if max_count >= options.min_dominant_class_classifications_above_threshold_for_unclassified_smoothing:
            
            results_this_sequence = _results_for_sequence(images_this_sequence,filename_to_results)
            detections_this_sequence = []
            for r in results_this_sequence:
                if r['detections'] is not None:
                    detections_this_sequence.extend(r['detections'])
            for det in detections_this_sequence:
                if 'classifications' in det and len(det['classifications']) > 0:
                    continue
                if det['category'] != options.animal_detection_category:
                    continue
                if det['conf'] < options.min_detection_confidence_for_unclassified_flipping:
                    continue
                det['classifications'] = [[dominant_category_id,options.flipped_unclassified_confidence_value]]
                n_unclassified_flips += 1
                                
    # ...for each sequence    
        
    print('\Finished sequence smoothing\n')
    print('Flipped {} "other" classifications'.format(n_other_flips))
    print('Flipped {} species classifications'.format(n_classification_flips))
    print('Flipped {} unclassified detections'.format(n_unclassified_flips))
        
    
    ##%% Write smoothed classification results
    
    if output_file is not None:
        
        print('Writing sequence-smoothed classification results to {}'.format(
            output_file))
        
        with open(output_file,'w') as f:
            json.dump(md_results,f,indent=1)
            
    return md_results

# ...smooth_classification_results_sequence_level(...)
