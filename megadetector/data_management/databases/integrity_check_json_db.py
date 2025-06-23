"""

integrity_check_json_db.py

Does some integrity-checking and computes basic statistics on a COCO Camera Traps .json file, specifically:

* Verifies that required fields are present and have the right types
* Verifies that annotations refer to valid images
* Verifies that annotations refer to valid categories
* Verifies that image, category, and annotation IDs are unique
* Optionally checks file existence
* Finds un-annotated images
* Finds unused categories
* Prints a list of categories sorted by count

"""

#%% Constants and environment

import argparse
import json
import os
import sys

from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from operator import itemgetter
from tqdm import tqdm

from megadetector.visualization.visualization_utils import open_image
from megadetector.utils import ct_utils
from megadetector.utils.path_utils import find_images


#%% Classes and environment

class IntegrityCheckOptions:
    """
    Options for integrity_check_json_db()
    """

    def __init__(self):

        #: Image path; the filenames in the .json file should be relative to this folder
        self.baseDir = ''

        #: Should we validate the image sizes?
        self.bCheckImageSizes = False

        #: Should we check that all the images in the .json file exist on disk?
        self.bCheckImageExistence = False

        #: Should we search [baseDir] for images that are not used in the .json file?
        self.bFindUnusedImages = False

        #: Should we require that all images in the .json file have a 'location' field?
        self.bRequireLocation = True

        #: For debugging, limit the number of images we'll process
        self.iMaxNumImages = -1

        #: Number of threads to use for parallelization, set to <= 1 to disable parallelization
        self.nThreads = 10

        #: Whether to use threads (rather than processes for parallelization)
        self.parallelizeWithThreads = True

        #: Enable additional debug output
        self.verbose = True

        #: Allow integer-valued image and annotation IDs (COCO uses this, CCT files use strings)
        self.allowIntIDs = False

        #: If True, error if the 'info' field is not present
        self.requireInfo = False


#%% Functions

def _check_image_existence_and_size(image,options=None):
    """
    Validate the image represented in the CCT image dict [image], which should have fields:

    * file_name
    * width
    * height

    Args:
        image (dict): image to validate
        options (IntegrityCheckOptions): parameters impacting validation

    Returns:
        str: None if this image passes validation, otherwise an error string
    """

    if options is None:
        options = IntegrityCheckOptions()

    assert options.bCheckImageExistence

    file_path = os.path.join(options.baseDir,image['file_name'])
    if not os.path.isfile(file_path):
        s = 'Image path {} does not exist'.format(file_path)
        return s

    if options.bCheckImageSizes:
        if not ('height' in image and 'width' in image):
            s = 'Missing image size in {}'.format(file_path)
            return s

        # width, height = Image.open(file_path).size
        try:
            pil_im = open_image(file_path)
        except Exception as e:
            s = 'Error opening {}: {}'.format(file_path,str(e))
            return s

        width,height = pil_im.size
        if (not (width == image['width'] and height == image['height'])):
            s = 'Size mismatch for image {}: {} (reported {},{}, actual {},{})'.format(
                    image['id'], file_path, image['width'], image['height'], width, height)
            return s

    return None


def integrity_check_json_db(json_file, options=None):
    """
    Does some integrity-checking and computes basic statistics on a COCO Camera Traps .json file; see
    module header comment for a list of the validation steps.

    Args:
        json_file (str): filename to validate, or an already-loaded dict
        options (IntegrityCheckOptions, optional): see IntegrityCheckOptions

    Returns:
        tuple: tuple containing:
            - sorted_categories (dict): list of categories used in [json_file], sorted by frequency
            - data (dict): the data loaded from [json_file]
            - error_info (dict): specific validation errors
    """

    if options is None:
        options = IntegrityCheckOptions()

    if options.bCheckImageSizes:
        options.bCheckImageExistence = True

    if options.verbose:
        print(options.__dict__)

    if options.baseDir is None:
        options.baseDir = ''

    base_dir = options.baseDir


    ##%% Read .json file if necessary, integrity-check fields

    if isinstance(json_file,dict):

        data = json_file

    elif isinstance(json_file,str):

        assert os.path.isfile(json_file), '.json file {} does not exist'.format(json_file)

        if options.verbose:
            print('Reading .json {} with base dir [{}]...'.format(
                    json_file,base_dir))

        with open(json_file,'r') as f:
            data = json.load(f)

    else:

        raise ValueError('Illegal value for json_file')

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    if options.requireInfo:
        assert 'info' in data, 'No info struct in database'

    if len(base_dir) > 0:
        assert os.path.isdir(base_dir), \
            'Base directory {} does not exist'.format(base_dir)


    ##%% Build dictionaries, checking ID uniqueness and internal validity as we go

    image_id_to_image = {}
    ann_id_to_ann = {}
    category_id_to_category = {}
    category_name_to_category = {}
    image_location_set = set()

    if options.verbose:
        print('Checking categories...')

    for cat in tqdm(categories):

        # Confirm that required fields are present
        assert 'name' in cat
        assert 'id' in cat

        assert isinstance(cat['id'],int), \
            'Illegal category ID type: [{}]'.format(str(cat['id']))
        assert isinstance(cat['name'],str), \
            'Illegal category name type [{}]'.format(str(cat['name']))

        category_id = cat['id']
        category_name = cat['name']

        # Confirm ID uniqueness
        assert category_id not in category_id_to_category, \
            'Category ID {} is used more than once'.format(category_id)
        category_id_to_category[category_id] = cat
        cat['_count'] = 0

        assert category_name not in category_name_to_category, \
            'Category name {} is used more than once'.format(category_name)
        category_name_to_category[category_name] = cat

    # ...for each category

    if options.verbose:
        print('\nChecking image records...')

    if options.iMaxNumImages > 0 and len(images) > options.iMaxNumImages:

        if options.verbose:
            print('Trimming image list to {}'.format(options.iMaxNumImages))
        images = images[0:options.iMaxNumImages]

    image_paths_in_json = set()

    sequences = set()

    # image = images[0]
    for image in tqdm(images):

        image['_count'] = 0

        # Confirm that required fields are present
        assert 'file_name' in image
        assert 'id' in image

        image['file_name'] = image['file_name'].replace('\\','/')

        image_paths_in_json.add(image['file_name'])

        assert isinstance(image['file_name'],str), 'Illegal image filename type'

        if options.allowIntIDs:
            assert isinstance(image['id'],str) or isinstance(image['id'],int), \
                'Illegal image ID type'
        else:
            assert isinstance(image['id'],str), 'Illegal image ID type'

        image_id = image['id']

        # Confirm ID uniqueness
        assert image_id not in image_id_to_image, 'Duplicate image ID {}'.format(image_id)

        image_id_to_image[image_id] = image

        if 'height' in image:
            assert 'width' in image, 'Image with height but no width: {}'.format(image['id'])

        if 'width' in image:
            assert 'height' in image, 'Image with width but no height: {}'.format(image['id'])

        if options.bRequireLocation:
            assert 'location' in image, 'No location available for: {}'.format(image['id'])

        if 'location' in image:
            # We previously supported ints here; this should be strings now
            # assert isinstance(image['location'], str) or isinstance(image['location'], int), \
            #  'Illegal image location type'
            assert isinstance(image['location'], str)
            image_location_set.add(image['location'])

        if 'seq_id' in image:
            sequences.add(image['seq_id'])

        assert not ('sequence_id' in image or 'sequence' in image), 'Illegal sequence identifier'

    unused_files = []

    image_paths_relative = None

    # Are we checking for unused images?
    if (len(base_dir) > 0) and options.bFindUnusedImages:

        if options.verbose:
            print('\nEnumerating images...')

        image_paths_relative = find_images(base_dir,return_relative_paths=True,recursive=True)

        for fn_relative in image_paths_relative:
            if fn_relative not in image_paths_in_json:
                unused_files.append(fn_relative)

    # List of (filename,error_string) tuples
    validation_errors = []

    # If we're checking image existence but not image size, we don't need to read the images
    if options.bCheckImageExistence and not options.bCheckImageSizes:

        if image_paths_relative is None:
            image_paths_relative = find_images(base_dir,return_relative_paths=True,recursive=True)

        image_paths_relative_set = set(image_paths_relative)

        for im in images:
            if im['file_name'] not in image_paths_relative_set:
                validation_errors.append((im['file_name'],'not found in relative path list'))

    # If we're checking image size, we need to read the images
    if options.bCheckImageSizes:

        if len(base_dir) == 0:
            print('Warning: checking image sizes without a base directory, assuming "."')

        if options.verbose:
            print('Checking image existence and/or image sizes...')

        if options.nThreads is not None and options.nThreads > 1:

            if options.parallelizeWithThreads:
                worker_string = 'threads'
            else:
                worker_string = 'processes'

            if options.verbose:
                print('Starting a pool of {} {}'.format(options.nThreads,worker_string))
            if options.parallelizeWithThreads:
                pool = ThreadPool(options.nThreads)
            else:
                pool = Pool(options.nThreads)
            try:
                results = list(tqdm(pool.imap(
                    partial(_check_image_existence_and_size,options=options), images),
                    total=len(images)))
            finally:
                pool.close()
                pool.join()
                print("Pool closed and joined for image size checks")
        else:
            results = []
            for im in tqdm(images):
                results.append(_check_image_existence_and_size(im,options))

        for i_image,result in enumerate(results):
            if result is not None:
                validation_errors.append((images[i_image]['file_name'],result))

    # ...for each image

    if options.verbose:
        print('{} validation errors (of {})'.format(len(validation_errors),len(images)))
        print('Checking annotations...')

    n_boxes = 0

    for ann in tqdm(annotations):

        # Confirm that required fields are present
        assert 'image_id' in ann
        assert 'id' in ann
        assert 'category_id' in ann

        if options.allowIntIDs:
            assert isinstance(ann['id'],str) or isinstance(ann['id'],int), \
                'Illegal annotation ID type'
            assert isinstance(ann['image_id'],str) or isinstance(ann['image_id'],int), \
                'Illegal annotation image ID type'
        else:
            assert isinstance(ann['id'],str), 'Illegal annotation ID type'
            assert isinstance(ann['image_id'],str), 'Illegal annotation image ID type'

        assert isinstance(ann['category_id'],int), 'Illegal annotation category ID type'

        if 'bbox' in ann:
            n_boxes += 1

        ann_id = ann['id']

        # Confirm ID uniqueness
        assert ann_id not in ann_id_to_ann
        ann_id_to_ann[ann_id] = ann

        # Confirm validity
        assert ann['category_id'] in category_id_to_category, \
            'Category {} not found in category list'.format(ann['category_id'])
        assert ann['image_id'] in image_id_to_image, \
          'Image ID {} referred to by annotation {}, not available'.format(
            ann['image_id'],ann['id'])

        image_id_to_image[ann['image_id']]['_count'] += 1
        category_id_to_category[ann['category_id']]['_count'] +=1

    # ...for each annotation

    sorted_categories = sorted(categories, key=itemgetter('_count'), reverse=True)


    ##%% Print statistics

    if options.verbose:

        # Find un-annotated images and multi-annotation images
        n_unannotated = 0
        n_multi_annotated = 0

        for image in images:
            if image['_count'] == 0:
                n_unannotated += 1
            elif image['_count'] > 1:
                n_multi_annotated += 1

        print('\nFound {} unannotated images, {} images with multiple annotations'.format(
                n_unannotated,n_multi_annotated))

        if (len(base_dir) > 0) and options.bFindUnusedImages:
            print('Found {} unused image files'.format(len(unused_files)))

        n_unused_categories = 0

        # Find unused categories
        for cat in categories:
            if cat['_count'] == 0:
                print('Unused category: {}'.format(cat['name']))
                n_unused_categories += 1

        print('Found {} unused categories'.format(n_unused_categories))

        sequence_string = 'no sequence info'
        if len(sequences) > 0:
            sequence_string = '{} sequences'.format(len(sequences))

        print('\nDB contains {} images, {} annotations, {} bboxes, {} categories, {}\n'.format(
                len(images),len(annotations),n_boxes,len(categories),sequence_string))

        if len(image_location_set) > 0:
            print('DB contains images from {} locations\n'.format(len(image_location_set)))

        print('Categories and annotation (not image) counts:\n')

        for cat in sorted_categories:
            print('{:6} {}'.format(cat['_count'],cat['name']))

        print('')

    error_info = {}
    error_info['unused_files'] = unused_files
    error_info['validation_errors'] = validation_errors

    return sorted_categories, data, error_info

# ...def integrity_check_json_db()


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file',type=str,
                        help='COCO-formatted .json file to validate')
    parser.add_argument('--bCheckImageSizes', action='store_true',
                        help='Validate image size, requires baseDir to be specified. ' + \
                             'Implies existence checking.')
    parser.add_argument('--bCheckImageExistence', action='store_true',
                        help='Validate image existence, requires baseDir to be specified')
    parser.add_argument('--bFindUnusedImages', action='store_true',
                        help='Check for images in baseDir that aren\'t in the database, ' + \
                             'requires baseDir to be specified')
    parser.add_argument('--baseDir', action='store', type=str, default='',
                        help='Base directory for images')
    parser.add_argument('--bAllowNoLocation', action='store_true',
                        help='Disable errors when no location is specified for an image')
    parser.add_argument('--iMaxNumImages', action='store', type=int, default=-1,
                        help='Cap on total number of images to check')
    parser.add_argument('--nThreads', action='store', type=int, default=10,
                        help='Number of threads (only relevant when verifying image ' + \
                             'sizes and/or existence)')

    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    args.bRequireLocation = (not args.bAllowNoLocation)
    options = IntegrityCheckOptions()
    ct_utils.args_to_object(args, options)
    integrity_check_json_db(args.json_file,options)

if __name__ == '__main__':
    main()


#%% Interactive driver(s)

if False:

    #%%

    """
    python integrity_check_json_db.py ~/data/ena24.json --baseDir ~/data/ENA24 --bAllowNoLocation
    """

    # Integrity-check .json files for LILA
    json_files = [os.path.expanduser('~/data/ena24.json')]

    options = IntegrityCheckOptions()
    options.baseDir = os.path.expanduser('~/data/ENA24')
    options.bCheckImageSizes = False
    options.bFindUnusedImages = True
    options.bRequireLocation = False

    # options.iMaxNumImages = 10

    for json_file in json_files:

        sorted_categories,data,_ = integrity_check_json_db(json_file, options)
