r"""

subset_json_detector_output.py

Creates one or more subsets of a detector results file (.json), doing either
or both of the following (if both are requested, they happen in this order):

1) Retrieve all elements where filenames contain a specified query string, 
   optionally replacing that query with a replacement token. If the query is blank, 
   can also be used to prepend content to all filenames.

   Does not support regex's, but supports a special case of ^string to indicate "must start with
   to match".

2) Create separate .jsons for each unique path, optionally making the filenames 
   in those .json's relative paths.  In this case, you specify an output directory, 
   rather than an output path.  All images in the folder blah/foo/bar will end up 
   in a .json file called blah_foo_bar.json.

Can also apply a confidence threshold.

Can also subset by categories above a threshold (programmatic invocation only, this is
not supported at the command line yet).

To subset a COCO Camera Traps .json database, see subset_json_db.py

**Sample invocation (splitting into multiple json's)**

Read from "1800_idfg_statewide_wolf_detections_w_classifications.json", split up into 
individual .jsons in 'd:/temp/idfg/output', making filenames relative to their individual
folders:

python subset_json_detector_output.py "d:/temp/idfg/1800_idfg_statewide_wolf_detections_w_classifications.json" "d:/temp/idfg/output" --split_folders --make_folder_relative

Now do the same thing, but instead of writing .json's to d:/temp/idfg/output, write them to *subfolders*
corresponding to the subfolders for each .json file.

python subset_json_detector_output.py "d:/temp/idfg/1800_detections_S2.json" "d:/temp/idfg/output_to_folders" --split_folders --make_folder_relative --copy_jsons_to_folders

**Sample invocation (creating a single subset matching a query)**

Read from "1800_detections.json", write to "1800_detections_2017.json"

Include only images matching "2017", and change "2017" to "blah"

python subset_json_detector_output.py "d:/temp/1800_detections.json" "d:/temp/1800_detections_2017_blah.json" --query 2017 --replacement blah

Include all images, prepend with "prefix/"

python subset_json_detector_output.py "d:/temp/1800_detections.json" "d:/temp/1800_detections_prefix.json" --replacement "prefix/"

"""

#%% Constants and imports

import argparse
import sys
import copy
import json
import os
import re

from tqdm import tqdm

from megadetector.utils.ct_utils import args_to_object, get_max_conf, invert_dictionary
from megadetector.utils.path_utils import top_level_folder


#%% Helper classes

class SubsetJsonDetectorOutputOptions:
    """
    Options used to parameterize subset_json_detector_output()
    """

    def __init__(self):
            
        #: Only process files containing the token 'query'
        self.query = None
        
        #: Replace 'query' with 'replacement' if 'replacement' is not None.  If 'query' is None,
        #: prepend 'replacement'
        self.replacement = None
        
        #: Should we split output into individual .json files for each folder?
        self.split_folders = False
        
        #: Folder level to use for splitting ['bottom','top','n_from_bottom','n_from_top','dict']
        #:
        #: 'dict' requires 'split_folder_param' to be a dictionary mapping each filename
        #: to a token.
        self.split_folder_mode = 'bottom'  # 'top'
        
        #: When using the 'n_from_bottom' parameter to define folder splitting, this
        #: defines the number of directories from the bottom.  'n_from_bottom' with
        #: a parameter of zero is the same as 'bottom'.
        #:
        #: Same story with 'n_from_top'.
        #:
        #: When 'split_folder_mode' is 'dict', this should be a dictionary mapping each filename
        #: to a token.
        self.split_folder_param = 0
        
        #: Only meaningful if split_folders is True: should we convert pathnames to be relative
        #: the folder for each .json file?
        self.make_folder_relative = False
        
        #: Only meaningful if split_folders and make_folder_relative are True: if not None, 
        #: will copy .json files to their corresponding output directories, relative to 
        #: output_filename
        self.copy_jsons_to_folders = False
        
        #: Should we over-write .json files?
        self.overwrite_json_files = False
        
        #: If copy_jsons_to_folders is true, do we require that directories already exist?
        self.copy_jsons_to_folders_directories_must_exist = True
        
        #: Optional confidence threshold; if not None, detections below this confidence won't be
        #: included in the output.
        self.confidence_threshold = None
        
        #: Should we remove failed images?
        self.remove_failed_images = False
        
        #: Either a list of category IDs (as string-ints) (not names), or a dictionary mapping category *IDs* 
        #: (as string-ints) (not names) to thresholds.  Removes non-matching detections, does not 
        #: remove images.  Not technically mutually exclusize with category_names_to_keep, but it's an esoteric 
        #: scenario indeed where you would want to specify both.
        self.categories_to_keep = None
        
        #: Either a list of category names (not IDs), or a dictionary mapping category *names* (not IDs) to thresholds.  
        #: Removes non-matching detections, does not remove images.  Not technically mutually exclusize with 
        #: category_ids_to_keep, but it's an esoteric scenario indeed where you would want to specify both.
        self.category_names_to_keep = None
        
        #: Set to >0 during testing to limit the number of images that get processed.
        self.debug_max_images = -1
    
    
#%% Main function

def _write_detection_results(data, output_filename, options):
    """
    Writes the detector-output-formatted dict *data* to *output_filename*.
    """
    
    if (not options.overwrite_json_files) and os.path.isfile(output_filename):
        raise ValueError('File {} exists'.format(output_filename))
    
    basedir = os.path.dirname(output_filename)
    
    if options.copy_jsons_to_folders and options.copy_jsons_to_folders_directories_must_exist:
        if not os.path.isdir(basedir):
            raise ValueError('Directory {} does not exist'.format(basedir))
    else:
        os.makedirs(basedir, exist_ok=True)
    
    print('Writing detection output to {}'.format(output_filename))
    with open(output_filename, 'w') as f:
        json.dump(data,f,indent=1)

# ..._write_detection_results()


def subset_json_detector_output_by_confidence(data, options):
    """
    Removes all detections below options.confidence_threshold.
    
    Args:
        data (dict): data loaded from a MD results file
        options (SubsetJsonDetectorOutputOptions): parameters for subsetting
    
    Returns:
        dict: Possibly-modified version of data (also modifies in place)
    """
    
    if options.confidence_threshold is None:
        return data
    
    images_in = data['images']
    images_out = []    
    
    print('Subsetting by confidence >= {}'.format(options.confidence_threshold))
    
    n_max_changes = 0
    
    # im = images_in[0]
    for i_image, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        # Always keep failed images; if the caller wants to remove these, they
        # will use remove_failed_images
        if ('detections' not in im) or (im['detections'] is None):
            images_out.append(im)
            continue
        
        p_orig = get_max_conf(im)

        # Find all detections above threshold for this image
        detections = [d for d in im['detections'] if d['conf'] >= options.confidence_threshold]

        # If there are no detections above threshold, set the max probability
        # to -1, unless it already had a negative probability.
        if len(detections) == 0:
            if p_orig <= 0:                
                p = p_orig
            else:
                p = -1

        # Otherwise find the max confidence
        else:
            p = max([d['conf'] for d in detections])
        
        im['detections'] = detections

        # Did this thresholding result in a max-confidence change?
        if abs(p_orig - p) > 0.00001:

            # We should only be *lowering* max confidence values (i.e., making them negative)
            assert (p_orig <= 0) or (p < p_orig), \
                'Confidence changed from {} to {}'.format(p_orig, p)
            n_max_changes += 1
        
        if 'max_detection_conf' in im:
            im['max_detection_conf'] = p
            
        images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    print('done, found {} matches (of {}), {} max conf changes'.format(
            len(data['images']),len(images_in),n_max_changes))
    
    return data

# ...subset_json_detector_output_by_confidence()


def subset_json_detector_output_by_categories(data, options):
    """
    Removes all detections without detections above a threshold for specific categories.
    
    Args:
        data (dict): data loaded from a MD results file
        options (SubsetJsonDetectorOutputOptions): parameters for subsetting
    
    Returns:
        dict: Possibly-modified version of data (also modifies in place)
    """
    
    # If categories_to_keep is supplied as a list, convert to a dict
    if options.categories_to_keep is not None:
        if not isinstance(options.categories_to_keep, dict):
            dict_categories_to_keep = {}
            for category_id in options.categories_to_keep:
                # Set unspecified thresholds to a silly negative value
                dict_categories_to_keep[category_id] = -100000.0
            options.categories_to_keep = dict_categories_to_keep
    
    # If category_names_to_keep is supplied as a list, convert to a dict
    if options.category_names_to_keep is not None:
        if not isinstance(options.category_names_to_keep, dict):
            dict_category_names_to_keep = {}
            for category_name in options.category_names_to_keep:
                # Set unspecified thresholds to a silly negative value
                dict_category_names_to_keep[category_name] = -100000.0
            options.category_names_to_keep = dict_category_names_to_keep
            
    category_name_to_category_id = invert_dictionary(data['detection_categories'])
    
    # If some categories are supplied as names, convert all to IDs and add to "categories_to_keep"
    if options.category_names_to_keep is not None:
        if options.categories_to_keep is None:
            options.categories_to_keep = {}
        for category_name in options.category_names_to_keep:
            assert category_name in category_name_to_category_id, \
                'Category {} not in detection categories'.format(category_name)
            category_id = category_name_to_category_id[category_name]
            assert category_id not in options.categories_to_keep, \
                'Category {} ({}) specified as both a name and an ID'.format(
                    category_name,category_id)
            options.categories_to_keep[category_id] = options.category_names_to_keep[category_name]
    
    if options.categories_to_keep is None:
        return data
    
    images_in = data['images']
    images_out = []    
    
    print('Subsetting by categories (keeping {} categories):'.format(
        len(options.categories_to_keep)))
    
    for category_id in sorted(list(options.categories_to_keep.keys())):
        if category_id not in data['detection_categories']:
            print('Warning: category ID {} not in category map in this file'.format(category_id))
        else:
            print('{} ({}) (threshold {})'.format(
                category_id,
                data['detection_categories'][category_id],
                options.categories_to_keep[category_id]))
    
    n_detections_in = 0
    n_detections_kept = 0
    
    # im = images_in[0]
    for i_image, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        # Always keep failed images; if the caller wants to remove these, they
        # will use remove_failed_images        
        if ('detections' not in im) or (im['detections'] is None):
            images_out.append(im)
            continue
        
        n_detections_in += len(im['detections'])
                                  
        # Find all matching detections for this image
        detections = []
        for d in im['detections']:
            if (d['category'] in options.categories_to_keep) and \
               (d['conf'] > options.categories_to_keep[d['category']]):
               detections.append(d)
                       
        im['detections'] = detections

        if 'max_detection_conf' in im:
            if len(detections) == 0:
                p = 0
            else:
                p = max([d['conf'] for d in detections])
            im['max_detection_conf'] = p
        
        n_detections_kept += len(im['detections'])
            
        images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    print('done, kept {} detections (of {})'.format(
        n_detections_kept,n_detections_in))
    
    return data

# ...subset_json_detector_output_by_categories()


def remove_failed_images(data,options):
    """
    Removed failed images from [data]
    
    Args:
        data (dict): data loaded from a MD results file
        options (SubsetJsonDetectorOutputOptions): parameters for subsetting
    
    Returns:
        dict: Possibly-modified version of data (also modifies in place)
    """
    
    images_in = data['images']
    images_out = []    
    
    if not options.remove_failed_images:
        return data
        
    print('Removing failed images...', end='')
    
    # i_image = 0; im = images_in[0]
    for i_image, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        if 'failure' in im and isinstance(im['failure'],str):
            continue
        else:
            images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    n_removed = len(images_in) - len(data['images'])
    print('Done, removed {} of {}'.format(n_removed, len(images_in)))
    
    return data

# ...remove_failed_images()


def subset_json_detector_output_by_query(data, options):
    """
    Subsets to images whose filename matches options.query; replace all instances of 
    options.query with options.replacement.  No-op if options.query_string is None or ''.
    
    Args:
        data (dict): data loaded from a MD results file
        options (SubsetJsonDetectorOutputOptions): parameters for subsetting
    
    Returns:
        dict: Possibly-modified version of data (also modifies in place)
    """
    
    images_in = data['images']
    images_out = []    
    
    print('Subsetting by query {}, replacement {}...'.format(options.query, options.replacement), end='')
    
    query_string = options.query
    query_starts_with = False
    
    # Support a special case regex-like notation for "starts with"
    if query_string is not None and query_string.startswith('^'):
        query_string = query_string[1:]
        query_starts_with = True
        
    # i_image = 0; im = images_in[0]
    for i_image, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        fn = im['file']
        
        # Only take images that match the query
        if query_string is not None:
            if query_starts_with:
                if (not fn.startswith(query_string)):
                    continue
            else:
                if query_string not in fn:
                    continue
        
        if options.replacement is not None:
            if query_string is not None:
                fn = fn.replace(query_string, options.replacement)
            else:
                fn = options.replacement + fn
            
        im['file'] = fn
        
        images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    print('done, found {} matches (of {})'.format(len(data['images']), len(images_in)))
    
    return data

# ...subset_json_detector_output_by_query()

    
def subset_json_detector_output(input_filename, output_filename, options, data=None):
    """
    Main entry point; creates one or more subsets of a detector results file.  See the 
    module header comment for more information about the available subsetting approaches.
        
    Makes a copy of [data] before modifying if a data dictionary is supplied.
    
    Args:
        input_filename (str): filename to load and subset; can be None if [data] is supplied
        output_filename (str): file or folder name (depending on [options]) to which we should
            write subset results.
        options (SubsetJsonDetectorOutputOptions): parameters for .json splitting/subsetting;
            see SubsetJsonDetectorOutputOptions for details.
        data (dict, optional): data loaded from a .json file; if this is not None, [input_filename]
            will be ignored.  If supplied, this will be copied before it's modified.
    
    Returns:
        dict: Results that are either loaded from [input_filename] and processed, or copied
            from [data] and processed.
    
    """
    
    if options is None:    
        options = SubsetJsonDetectorOutputOptions()
    else:
        options = copy.deepcopy(options)
            
    # Input validation        
    if options.copy_jsons_to_folders:
        assert options.split_folders and options.make_folder_relative, \
            'copy_jsons_to_folders set without make_folder_relative and split_folders'
                
    if options.split_folders:
        if os.path.isfile(output_filename):
            raise ValueError('When splitting by folders, output must be a valid directory name, you specified an existing file')
            
    if data is None:
        print('Reading json...', end='')
        with open(input_filename) as f:
            data = json.load(f)
        print(' ...done, read {} images'.format(len(data['images'])))
        if options.debug_max_images > 0:
            print('Trimming to {} images'.format(options.debug_max_images))
            data['images'] = data['images'][:options.debug_max_images]
    else:
        print('Copying data')
        data = copy.deepcopy(data)
        print('...done')
        
    if options.query is not None:
        
        data = subset_json_detector_output_by_query(data, options)
    
    if options.remove_failed_images:
        
        data = remove_failed_images(data, options)
        
    if options.confidence_threshold is not None:
        
        data = subset_json_detector_output_by_confidence(data, options)
        
    if (options.categories_to_keep is not None) or (options.category_names_to_keep is not None):
        
        data = subset_json_detector_output_by_categories(data, options)
        
    if not options.split_folders:
        
        _write_detection_results(data, output_filename, options)
        return data
    
    else:
        
        # Map images to unique folders
        print('Finding unique folders')
        
        folders_to_images = {}
        
        # im = data['images'][0]
        for im in tqdm(data['images']):
            
            fn = im['file']
            
            if options.split_folder_mode == 'bottom':
                                
                dirname = os.path.dirname(fn)
                
            elif options.split_folder_mode == 'n_from_bottom':
                
                dirname = os.path.dirname(fn)
                for n in range(0, options.split_folder_param):
                    dirname = os.path.dirname(dirname)
                    
            elif options.split_folder_mode == 'n_from_top':
                
                # Split string into folders, keeping delimiters
                
                # Don't use this, it removes delimiters
                # tokens = _split_path(fn)
                tokens = re.split(r'([\\/])',fn)
                
                n_tokens_to_keep = ((options.split_folder_param + 1) * 2) - 1;
                
                if n_tokens_to_keep > len(tokens):
                    raise ValueError('Cannot walk {} folders from the top in path {}'.format(
                                options.split_folder_param, fn))
                dirname = ''.join(tokens[0:n_tokens_to_keep])
                
            elif options.split_folder_mode == 'top':
                
                dirname = top_level_folder(fn)                
                
            elif options.split_folder_mode == 'dict':
                
                assert isinstance(options.split_folder_param, dict)
                dirname = options.split_folder_param[fn]
                
            else:
                
                raise ValueError('Unrecognized folder split mode {}'.format(options.split_folder_mode))
                
            folders_to_images.setdefault(dirname, []).append(im)
        
        # ...for each image
                
        print('Found {} unique folders'.format(len(folders_to_images)))
        
        # Optionally make paths relative
        # dirname = list(folders_to_images.keys())[0]
        if options.make_folder_relative:
            
            print('Converting database-relative paths to individual-json-relative paths...')
        
            for dirname in tqdm(folders_to_images):
                # im = folders_to_images[dirname][0]
                for im in folders_to_images[dirname]:
                    fn = im['file']
                    relfn = os.path.relpath(fn, dirname).replace('\\', '/')
                    im['file'] = relfn
        
        # ...if we need to convert paths to be folder-relative
        
        print('Finished converting to json-relative paths, writing output')
                       
        os.makedirs(output_filename, exist_ok=True)
        all_images = data['images']
        
        # dirname = list(folders_to_images.keys())[0]
        for dirname in tqdm(folders_to_images):
                        
            json_fn = dirname.replace('/', '_').replace('\\', '_') + '.json'
            
            if options.copy_jsons_to_folders:
                json_fn = os.path.join(output_filename, dirname, json_fn)
            else:
                json_fn = os.path.join(output_filename, json_fn)
            
            # Recycle the 'data' struct, replacing 'images' every time... medium-hacky, but 
            # forward-compatible in that I don't take dependencies on the other fields
            dir_data = data
            dir_data['images'] = folders_to_images[dirname]
            _write_detection_results(dir_data, json_fn, options)
            print('Wrote {} images to {}'.format(len(dir_data['images']), json_fn))
            
        # ...for each directory
        
        data['images'] = all_images
        
        return data
    
    # ...if we're splitting folders

# ...subset_json_detector_output()

    
#%% Interactive driver
                
if False:

    #%%
    
    #%% Subset a file without splitting
    
    input_filename = r"c:\temp\sample.json"
    output_filename = r"c:\temp\output.json"
     
    options = SubsetJsonDetectorOutputOptions()
    options.replacement = None
    options.query = 'S2'
        
    data = subset_json_detector_output(input_filename,output_filename,options,None)
    

    #%% Subset and split, but don't copy to individual folders

    input_filename = r"C:\temp\xxx-20201028_detections.filtered_rde_0.60_0.85_10_0.05_r2_export\xxx-20201028_detections.filtered_rde_0.60_0.85_10_0.05_r2_export.json"
    output_filename = r"c:\temp\out"
    
    options = SubsetJsonDetectorOutputOptions()
    options.split_folders = True    
    options.make_folder_relative = True
    options.split_folder_mode = 'n_from_top'
    options.split_folder_param = 1
    
    data = subset_json_detector_output(input_filename,output_filename,options,None)
    
    
    #%% Subset and split, copying to individual folders
    
    input_filename = r"c:\temp\sample.json"
    output_filename = r"c:\temp\out"
     
    options = SubsetJsonDetectorOutputOptions()
    options.split_folders = True    
    options.make_folder_relative = True
    options.copy_jsons_to_folders = True
    
    data = subset_json_detector_output(input_filename,output_filename,options,data)
    

#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input .json filename')
    parser.add_argument('output_file', type=str, help='Output .json filename')
    parser.add_argument('--query', type=str, default=None, 
                        help='Query string to search for (omitting this matches all)')
    parser.add_argument('--replacement', type=str, default=None, 
                        help='Replace [query] with this')
    parser.add_argument('--confidence_threshold', type=float, default=None, 
                        help='Remove detections below this confidence level')
    parser.add_argument('--split_folders', action='store_true', 
                        help='Split .json files by leaf-node folder')
    parser.add_argument('--split_folder_param', type=int,
                        help='Directory level count for n_from_bottom and n_from_top splitting')
    parser.add_argument('--split_folder_mode', type=str,
                        help='Folder level to use for splitting ("top" or "bottom")')
    parser.add_argument('--make_folder_relative', action='store_true', 
                        help='Make image paths relative to their containing folder (only meaningful with split_folders)')
    parser.add_argument('--overwrite_json_files', action='store_true', 
                        help='Overwrite output files')
    parser.add_argument('--copy_jsons_to_folders', action='store_true', 
                        help='When using split_folders and make_folder_relative, copy jsons to their corresponding folders (relative to output_file)')
    parser.add_argument('--create_folders', action='store_true',
                        help='When using copy_jsons_to_folders, create folders that don''t exist')    
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SubsetJsonDetectorOutputOptions()
    if args.create_folders:
        options.copy_jsons_to_folders_directories_must_exist = False
        
    args_to_object(args, options)
    
    subset_json_detector_output(args.input_file, args.output_file, options)
    
if __name__ == '__main__':    
    main()
