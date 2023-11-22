########
#
# split_locations_into_train_val.py
#
# Split a list of location IDs into training and validation, targeting a specific
# train/val split for each category, but allowing some categories to be tighter or looser
# than others.  Does nothing particularly clever, just randomly splits locations into 
# train/val lots of times using the target val fraction, and picks the one that meets the 
# specified constraints and minimizes weighted error, where "error" is defined as the
# sum of each class's absolute divergence from the target val fraction.
#
########

#%% Imports/constants

import random
import numpy as np

from collections import defaultdict
from md_utils.ct_utils import sort_dictionary_by_value
from tqdm import tqdm


#%% Main function

def split_locations_into_train_val(location_to_category_counts,
                                   n_random_seeds=10000,
                                   target_val_fraction=0.15,
                                   category_to_max_allowable_error=None,                                   
                                   category_to_error_weight=None,
                                   default_max_allowable_error=0.1):
    """
    Split a list of location IDs into training and validation, targeting a specific
    train/val split for each category, but allowing some categories to be tighter or looser
    than others.  Does nothing particularly clever, just randomly splits locations into 
    train/val lots of times using the target val fraction, and picks the one that meets the 
    specified constraints and minimizes weighted error, where "error" is defined as the
    sum of each class's absolute divergence from the target val fraction.    
    
    location_to_category_counts should be a dict mapping location IDs to dicts,
    with each dict mapping a category name to a count.  Any categories not present in a 
    particular dict are assumed to have a count of zero for that location.
    
    If not None, category_to_max_allowable_error should be a dict mapping category names
    to maximum allowable errors.  These are hard constraints, but you can specify a subset
    of categories.  Categories not included here have a maximum error of Inf.
    
    If not None, category_to_error_weight should be a dict mapping category names to
    error weights.  You can specify a subset of categories.  Categories not included here
    have a weight of 1.0.
    
    default_max_allowable_error is the maximum allowable error for categories not present in
    category_to_max_allowable_error.  Set to None (or >= 1.0) to disable hard constraints for 
    categories not present in category_to_max_allowable_error
    
    returns val_locations,category_to_val_fraction
    
    """
    
    location_ids = list(location_to_category_counts.keys())
    
    n_val_locations = int(target_val_fraction*len(location_ids))
    
    if category_to_max_allowable_error is None:
        category_to_max_allowable_error = {}
        
    if category_to_error_weight is None:
        category_to_error_weight = {}
        
    # category ID to total count; the total count is used only for printouts
    category_id_to_count = {}
    for location_id in location_to_category_counts:
        for category_id in location_to_category_counts[location_id].keys():
            if category_id not in category_id_to_count:
                category_id_to_count[category_id] = 0
            category_id_to_count[category_id] += \
                location_to_category_counts[location_id][category_id]
    
    category_ids = set(category_id_to_count.keys())
    
    print('Splitting {} categories over {} locations'.format(
        len(category_ids),len(location_ids)))
    
    # random_seed = 0
    def compute_seed_errors(random_seed):
        """
        Compute the per-category error for a specific random seed.
        
        returns weighted_average_error,category_to_val_fraction
        """
        
        # Randomly split into train/val
        random.seed(random_seed)
        val_locations = random.sample(location_ids,k=n_val_locations)
        val_locations_set = set(val_locations)
        
        # For each category, measure the % of images that went into the val set
        category_to_val_fraction = defaultdict(float)
        
        for category_id in category_ids:
            category_val_count = 0
            category_train_count = 0
            for location_id in location_to_category_counts:
                if category_id not in location_to_category_counts[location_id]:
                    location_category_count = 0
                else:
                    location_category_count = location_to_category_counts[location_id][category_id]
                if location_id in val_locations_set:
                    category_val_count += location_category_count
                else:
                    category_train_count += location_category_count
            category_val_fraction = category_val_count / (category_val_count + category_train_count)
            category_to_val_fraction[category_id] = category_val_fraction
        
        # Absolute deviation from the target val fraction for each categorys
        category_errors = {}
        weighted_category_errors = {}
        
        # category = next(iter(category_to_val_fraction))
        for category in category_to_val_fraction:
            
            category_val_fraction = category_to_val_fraction[category]
            
            category_error = abs(category_val_fraction-target_val_fraction)
            category_errors[category] = category_error
        
            category_weight = 1.0
            if category in category_to_error_weight:
                category_weight = category_to_error_weight[category]
            weighted_category_error = category_error * category_weight
            weighted_category_errors[category] = weighted_category_error
        
        weighted_average_error = np.mean(list(weighted_category_errors.values()))
        
        return weighted_average_error,weighted_category_errors,category_to_val_fraction
    
    # ... def compute_seed_errors(...)
    
    # This will only include random seeds that satisfy the hard constraints
    random_seed_to_weighted_average_error = {}
    
    # random_seed = 0
    for random_seed in tqdm(range(0,n_random_seeds)):
        
        weighted_average_error,weighted_category_errors,category_to_val_fraction = \
            compute_seed_errors(random_seed)
            
        seed_satisfies_hard_constraints = True
        
        for category in category_to_val_fraction:
            if category in category_to_max_allowable_error:
                max_allowable_error = category_to_max_allowable_error[category]
            else:
                if default_max_allowable_error is None:
                    continue
                max_allowable_error = default_max_allowable_error
            val_fraction = category_to_val_fraction[category]
            category_error = abs(val_fraction - target_val_fraction)
            if category_error > max_allowable_error:
                seed_satisfies_hard_constraints = False
                break
        
        if seed_satisfies_hard_constraints:            
            random_seed_to_weighted_average_error[random_seed] = weighted_average_error
        
    # ...for each random seed
    
    assert len(random_seed_to_weighted_average_error) > 0, \
        'No random seed met all the hard constraints'
        
    print('\n{} of {} random seeds satisfied hard constraints'.format(
        len(random_seed_to_weighted_average_error),n_random_seeds))
    
    min_error = None
    min_error_seed = None
    
    for random_seed in random_seed_to_weighted_average_error.keys():
        error_metric = random_seed_to_weighted_average_error[random_seed]
        if min_error is None or error_metric < min_error:
            min_error = error_metric
            min_error_seed = random_seed
    
    random.seed(min_error_seed)
    val_locations = random.sample(location_ids,k=n_val_locations)
    train_locations = []
    for location_id in location_ids:
        if location_id not in val_locations:
            train_locations.append(location_id)
            
    print('\nVal locations:\n')        
    for loc in val_locations:
        print('{}'.format(loc))
    print('')
    
    weighted_average_error,weighted_category_errors,category_to_val_fraction = \
        compute_seed_errors(min_error_seed)
        
    random_seed = min_error_seed
    
    category_to_val_fraction = sort_dictionary_by_value(category_to_val_fraction,reverse=True)
    category_to_val_fraction = sort_dictionary_by_value(category_to_val_fraction,
                                                        sort_values=category_id_to_count,
                                                        reverse=True)
    
    
    print('Val fractions by category:\n')
    
    for category in category_to_val_fraction:
        print('{} ({}) {:.2f}'.format(
            category,category_id_to_count[category],
            category_to_val_fraction[category]))
          
    return val_locations,category_to_val_fraction

# ...def split_locations_into_train_val(...)
