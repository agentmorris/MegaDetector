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

from multiprocessing.pool import ThreadPool
from operator import itemgetter
from tqdm import tqdm

from megadetector.visualization.visualization_utils import open_image
from megadetector.utils import ct_utils


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
        
        #: Enable additional debug output
        self.verbose = True
        
        #: Allow integer-valued image and annotation IDs (COCO uses this, CCT files use strings)
        self.allowIntIDs = False
    
# This is used in a medium-hacky way to share modified options across threads
defaultOptions = IntegrityCheckOptions()


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
        bool: whether this image passes validation
    """

    if options is None:        
        options = defaultOptions
    
    assert options.bCheckImageExistence
    
    filePath = os.path.join(options.baseDir,image['file_name'])
    if not os.path.isfile(filePath):
        # print('Image path {} does not exist'.format(filePath))
        return False
    
    if options.bCheckImageSizes:
        if not ('height' in image and 'width' in image):
            print('Missing image size in {}'.format(filePath))
            return False

        # width, height = Image.open(filePath).size
        pil_im = open_image(filePath)
        width,height = pil_im.size
        if (not (width == image['width'] and height == image['height'])):
            print('Size mismatch for image {}: {} (reported {},{}, actual {},{})'.format(
                    image['id'], filePath, image['width'], image['height'], width, height))
            return False
        
    return True

  
def integrity_check_json_db(jsonFile, options=None):
    """
    Does some integrity-checking and computes basic statistics on a COCO Camera Traps .json file; see
    module header comment for a list of the validation steps.
    
    Args:
        jsonFile (str): filename to validate, or an already-loaded dict
    
    Returns:
        tuple: tuple containing:
            - sortedCategories (dict): list of categories used in [jsonFile], sorted by frequency
            - data (dict): the data loaded from [jsonFile]
            - errorInfo (dict): specific validation errors
    """
    
    if options is None:       
        options = IntegrityCheckOptions()
    
    if options.bCheckImageSizes:        
        options.bCheckImageExistence = True
     
    if options.verbose:
        print(options.__dict__)
    
    if options.baseDir is None:
        options.baseDir = ''
        
    baseDir = options.baseDir
    
    
    ##%% Read .json file if necessary, integrity-check fields
    
    if isinstance(jsonFile,dict):
        
        data = jsonFile
        
    elif isinstance(jsonFile,str):
        
        assert os.path.isfile(jsonFile), '.json file {} does not exist'.format(jsonFile)
    
        if options.verbose:
            print('Reading .json {} with base dir [{}]...'.format(
                    jsonFile,baseDir))
        
        with open(jsonFile,'r') as f:
            data = json.load(f) 
            
    else:
        
        raise ValueError('Illegal value for jsonFile')
            
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    # info = data['info']
    assert 'info' in data, 'No info struct in database'
    
    if len(baseDir) > 0:        
        assert os.path.isdir(baseDir), 'Base directory {} does not exist'.format(baseDir)
        
        
    ##%% Build dictionaries, checking ID uniqueness and internal validity as we go
    
    imageIdToImage = {}
    annIdToAnn = {}
    catIdToCat = {}
    catNameToCat = {}
    imageLocationSet = set()
    
    if options.verbose:
        print('Checking categories...')
    
    for cat in tqdm(categories):
        
        # Confirm that required fields are present
        assert 'name' in cat
        assert 'id' in cat
        
        assert isinstance(cat['id'],int), 'Illegal category ID type: [{}]'.format(str(cat['id']))
        assert isinstance(cat['name'],str), 'Illegal category name type [{}]'.format(str(cat['name']))
        
        catId = cat['id']
        catName = cat['name']
        
        # Confirm ID uniqueness
        assert catId not in catIdToCat, 'Category ID {} is used more than once'.format(catId)
        catIdToCat[catId] = cat
        cat['_count'] = 0
        
        assert catName not in catNameToCat, 'Category name {} is used more than once'.format(catName)
        catNameToCat[catName] = cat        
        
    # ...for each category
        
    if options.verbose:
        print('\nChecking images...')
    
    if options.iMaxNumImages > 0 and len(images) > options.iMaxNumImages:
        
        if options.verbose:
            print('Trimming image list to {}'.format(options.iMaxNumImages))
        images = images[0:options.iMaxNumImages]
        
    imagePathsInJson = set()
    
    sequences = set()
    
    # image = images[0]
    for image in tqdm(images):
        
        image['_count'] = 0
        
        # Confirm that required fields are present
        assert 'file_name' in image
        assert 'id' in image

        image['file_name'] = os.path.normpath(image['file_name'])
                
        imagePathsInJson.add(image['file_name'])
        
        assert isinstance(image['file_name'],str), 'Illegal image filename type'
        
        if options.allowIntIDs:
            assert isinstance(image['id'],str) or isinstance(image['id'],int), \
                'Illegal image ID type'
        else:
            assert isinstance(image['id'],str), 'Illegal image ID type'
        
        imageId = image['id']        
        
        # Confirm ID uniqueness
        assert imageId not in imageIdToImage, 'Duplicate image ID {}'.format(imageId)
        
        imageIdToImage[imageId] = image
        
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
            imageLocationSet.add(image['location'])
    
        if 'seq_id' in image:
            sequences.add(image['seq_id'])
            
        assert not ('sequence_id' in image or 'sequence' in image), 'Illegal sequence identifier'
        
    unusedFiles = []
                
    # Are we checking for unused images?
    if (len(baseDir) > 0) and options.bFindUnusedImages:    
        
        if options.verbose:
            print('\nEnumerating images...')
        
        # Recursively enumerate images
        imagePaths = []
        for root, dirs, files in os.walk(baseDir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    relDir = os.path.relpath(root, baseDir)
                    relFile = os.path.join(relDir,file)
                    relFile = os.path.normpath(relFile)
                    if len(relFile) > 2 and \
                        (relFile[0:2] == './' or relFile[0:2] == '.\\'):                     
                            relFile = relFile[2:]
                    imagePaths.append(relFile)
          
        for p in imagePaths:
            if p not in imagePathsInJson:
                # print('Image {} is unused'.format(p))
                unusedFiles.append(p)
                
    validationErrors = []
    
    # Are we checking file existence and/or image size?
    if options.bCheckImageSizes or options.bCheckImageExistence:
        
        if len(baseDir) == 0:
            print('Warning: checking image sizes without a base directory, assuming "."')
         
        if options.verbose:
            print('Checking image existence and/or image sizes...')
        
        if options.nThreads is not None and options.nThreads > 1:
            pool = ThreadPool(options.nThreads)
            # results = pool.imap_unordered(lambda x: fetch_url(x,nImages), indexedUrlList)
            defaultOptions.baseDir = options.baseDir
            defaultOptions.bCheckImageSizes = options.bCheckImageSizes
            defaultOptions.bCheckImageExistence = options.bCheckImageExistence
            results = tqdm(pool.imap(_check_image_existence_and_size, images), total=len(images))
        else:
            results = []
            for im in tqdm(images):
                results.append(_check_image_existence_and_size(im,options))
                
        for iImage,r in enumerate(results):
            if not r:            
                validationErrors.append(os.path.join(options.baseDir,images[iImage]['file_name']))
                            
    # ...for each image
    
    if options.verbose:
        print('{} validation errors (of {})'.format(len(validationErrors),len(images)))
        print('Checking annotations...')
    
    nBoxes = 0
    
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
            nBoxes += 1
            
        annId = ann['id']        
        
        # Confirm ID uniqueness
        assert annId not in annIdToAnn
        annIdToAnn[annId] = ann
    
        # Confirm validity
        assert ann['category_id'] in catIdToCat, \
            'Category {} not found in category list'.format(ann['category_id'])
        assert ann['image_id'] in imageIdToImage, \
          'Image ID {} referred to by annotation {}, not available'.format(
            ann['image_id'],ann['id'])
    
        imageIdToImage[ann['image_id']]['_count'] += 1
        catIdToCat[ann['category_id']]['_count'] +=1 
        
    # ...for each annotation
    
    sortedCategories = sorted(categories, key=itemgetter('_count'), reverse=True)
    
    
    ##%% Print statistics
    
    if options.verbose:
    
        # Find un-annotated images and multi-annotation images
        nUnannotated = 0
        nMultiAnnotated = 0
        
        for image in images:
            if image['_count'] == 0:
                nUnannotated += 1
            elif image['_count'] > 1:
                nMultiAnnotated += 1
                
        print('Found {} unannotated images, {} images with multiple annotations'.format(
                nUnannotated,nMultiAnnotated))
        
        if (len(baseDir) > 0) and options.bFindUnusedImages:
            print('Found {} unused image files'.format(len(unusedFiles)))
            
        nUnusedCategories = 0
        
        # Find unused categories
        for cat in categories:
            if cat['_count'] == 0:
                print('Unused category: {}'.format(cat['name']))
                nUnusedCategories += 1
        
        print('Found {} unused categories'.format(nUnusedCategories))
                
        sequenceString = 'no sequence info'
        if len(sequences) > 0:
            sequenceString = '{} sequences'.format(len(sequences))
            
        print('\nDB contains {} images, {} annotations, {} bboxes, {} categories, {}\n'.format(
                len(images),len(annotations),nBoxes,len(categories),sequenceString))
    
        if len(imageLocationSet) > 0:
            print('DB contains images from {} locations\n'.format(len(imageLocationSet)))
                
        print('Categories and annotation (not image) counts:\n')
        
        for cat in sortedCategories:
            print('{:6} {}'.format(cat['_count'],cat['name']))
        
        print('')
    
    errorInfo = {}
    errorInfo['unusedFiles'] = unusedFiles
    errorInfo['validationErrors'] = validationErrors
    
    return sortedCategories, data, errorInfo

# ...def integrity_check_json_db()
    

#%% Command-line driver
    
def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonFile')
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
    integrity_check_json_db(args.jsonFile,options)

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
        
        sortedCategories,data,_ = integrity_check_json_db(json_file, options)
