"""

 missouri_to_json.py

 Create .json files from the original source files for the Missouri Camera Traps
 data set.  Metadata was provided here in two formats:

 1) In one subset of the data, folder names indicated species names.  In Set 1,
    there are no empty sequences.  Set 1 has a metadata file to indicate image-level
    bounding boxes.

 2) A subset of the data (overlapping with (1)) was annotated with bounding
    boxes, specified in a whitespace-delimited text file.  In set 2, there are
    some sequences omitted from the metadata file, which implied emptiness.
 
 In the end, set 2 labels were not reliable enough to publish, so LILA includes only set 1.

"""

#%% Constants and imports

import json
import os
import uuid
import time
import humanfriendly
import warnings
import ntpath
import datetime

from PIL import Image

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)

baseDir = os.path.expanduser('~/tmp/mct')
imageDir = os.path.join(baseDir,'images')

outputJsonFilenameSet1 = os.path.join(baseDir,'missouri_camera_traps_set1.json')
outputEncoding = 'utf-8'
fileListJsonFilename = os.path.join(baseDir,'images.json')

# This will be a list of filenames that need re-annotation due to redundant boxes
set1RedundantBoxListFilename = os.path.join(baseDir,'mct_images_with_redundant_boxes.txt')

set1BaseDir = os.path.join(imageDir,'Set1')

metadataFilenameSet1 = os.path.join(set1BaseDir,'labels.txt')

assert(os.path.isdir(baseDir))
assert(os.path.isfile(metadataFilenameSet1))

info = {}
info['year'] = 2019
info['version'] = '1.21'
info['description'] = 'Missouri Camera Traps (set 1)'
info['contributor'] = ''
info['date_created'] = str(datetime.date.today())
infoSet1 = info

maxFiles = -1
emptyCategoryId = 0
emptyCategoryName = 'empty'


#%% Enumerate files, read image sizes

# Takes a few minutes, since we're reading image sizes.

# Each element will be a list of relative path/full path/width/height
fileInfo = []
nonImages = []
nFiles = 0

relPathToIm = {}
imageIdToImage = {}

set1ImageIDs = []
   
sequenceIDtoCount = {}

print('Enumerating files from {} to {}'.format(imageDir,fileListJsonFilename))

startTime = time.time()

for root, subdirs, files in os.walk(imageDir):
            
    if root == imageDir:
        continue
    
    bn = ntpath.basename(root)
    
    # Only process leaf nodes corresponding to sequences, which look like:
    #
    # Set1/1.02-Agouti/SEQ75583    
    #
    if ('Set1' in root and 'SEQ' in bn):
        sequenceID = bn
        assert sequenceID not in sequenceIDtoCount
        sequenceIDtoCount[sequenceID] = 0
    else:
        print('Skipping folder {}:{}'.format(root,bn))
        continue
        # assert len(files) <= 2
    
    for fname in files:
  
        nFiles = nFiles + 1
        if maxFiles >= 0 and nFiles > maxFiles:            
            print('Warning: early break at {} files'.format(maxFiles))
            break
        
        fullPath = os.path.join(root,fname)            
        relativePath = os.path.relpath(fullPath,imageDir)
        
        if maxFiles >= 0:
            print(relativePath)
    
        h = -1
        w = -1

        # Read the image
        try:
        
            im = Image.open(fullPath)
            h = im.height
            w = im.width
            
        except:
            
            # Not an image...
            continue
        
        # Store file info
        im = {}
        im['id'] = str(uuid.uuid1())
        im['file_name'] = relativePath
        im['height'] = h
        im['width'] = w
        im['location'] = 'missouri_camera_traps'
        
        im['seq_id'] = sequenceID
        im['seq_num_frames'] = -1
        
        frame_number = sequenceIDtoCount[sequenceID]
        im['frame_num'] = frame_number
        sequenceIDtoCount[sequenceID] = sequenceIDtoCount[sequenceID] + 1
        
        imageIdToImage[im['id']] = im
        relPathToIm[relativePath] = im
        
        if 'Set1' in relativePath:
            set1ImageIDs.append(im['id'])
        else:
            raise Exception('Oops, can''t assign this image to a set')
            
    # ...if we didn't hit the max file limit, keep going
    
    else:
        
        continue
    
    break

# ...for each file

elapsed = time.time() - startTime
print('Finished file enumeration in {}'.format(
      humanfriendly.format_timespan(elapsed)))


#%% Add sequence lengths
    
for imageID in imageIdToImage:
    
    im = imageIdToImage[imageID]
    sequenceID = im['seq_id']
    seq_num_frames = sequenceIDtoCount[sequenceID]
    assert(im['seq_num_frames'] == -1)
    im['seq_num_frames'] = seq_num_frames
    

#%% Load the metadata (.txt) file

with open(metadataFilenameSet1) as f:
    metadataSet1Lines = f.readlines()

metadataSet1Lines = [x.strip() for x in metadataSet1Lines] 


#%% Map relative paths to metadata

# List of lists, length varies according to number of bounding boxes
#
# Preserves original ordering
missingFilesSet1 = []
correctedFiles = []

relPathToMetadataSet1 = {}

# iLine = 0; line = metadataSet1Lines[0]
for iLine,line in enumerate(metadataSet1Lines):
    
    tokens = line.split()
    nTokens = len(tokens)
    
    # Lines should be filename, number of bounding boxes, boxes (four values per box)
    assert ((nTokens - 2) % 4) == 0
    relPath = tokens[0].replace('/',os.sep).replace('\\',os.sep)
    relPath = os.path.join('Set1',relPath)
    absPath = os.path.join(imageDir,relPath)
    
    originalAbsPath = absPath
    originalRelPath = relPath
        
    if not os.path.isfile(absPath):
        
        absPath = originalAbsPath.replace('IMG','IMG_')
        relPath = originalRelPath.replace('IMG','IMG_')
        if os.path.isfile(absPath):
            correctedFiles.append([relPath,originalRelPath,absPath,originalAbsPath])
    
    if not os.path.isfile(absPath):
        
        absPath = originalAbsPath.replace('Red_Deer','Red_Brocket_Deer').replace('IMG','IMG_')
        relPath = originalRelPath.replace('Red_Deer','Red_Brocket_Deer').replace('IMG','IMG_')
        if os.path.isfile(absPath):
            correctedFiles.append([relPath,originalRelPath,absPath,originalAbsPath])
            
    if not os.path.isfile(absPath):
        
        missingFilesSet1.append([originalRelPath,originalAbsPath])
        
    else:
        
        relPathToMetadataSet1[relPath] = tokens
        
        # Make sure we have image info for this image
        assert relPath in relPathToIm

print('Corrected {} paths, missing {} images of {}'.format(len(correctedFiles),
      len(missingFilesSet1),len(metadataSet1Lines)))


#%% Print missing files from Set 1 metadata

# The only missing file (and it's really just missing):
#    
# Set1/1.58-Roe_Deer/SEQ75631/SEQ75631_IMG_0011.JPG

print('Missing files in Set 1:\n')
for iFile,fInfo in enumerate(missingFilesSet1):
    print(fInfo[0])
    

#%% Create categories and annotations for set 1

imagesSet1 = []
categoriesSet1 = []
annotationsSet1 = []

categoryNameToId = {}
idToCategory = {}

# Though we have no empty sequences, we do have empty images in this set
emptyCat = {}
emptyCat['id'] = emptyCategoryId
emptyCat['name'] = emptyCategoryName
emptyCat['count'] = 0
categoriesSet1.append(emptyCat) 

nextCategoryId = emptyCategoryId + 1
    
nFoundMetadata = 0
nTotalBoxes = 0
nImageLevelEmpties = 0
nSequenceLevelAnnotations = 0
nRedundantBoxes = 0

imageIDsWithRedundantBoxes = set()

# For each image
#
# iImage = 0; imageID = set1ImageIDs[iImage]
for iImage,imageID in enumerate(set1ImageIDs):
    
    im = imageIdToImage[imageID]
    imagesSet1.append(im)
    
    # E.g. Set1\\1.80-Coiban_Agouti\\SEQ83155\\SEQ83155_IMG_0010.JPG
    relPath = im['file_name']

    # Find the species name
    tokens = os.path.normpath(relPath).split(os.sep)
    speciesTag = tokens[1]
    tokens = speciesTag.split('-',1)
    assert(len(tokens) == 2)
    categoryName = tokens[1].lower()
    
    category = None
    categoryId = None
    
    if categoryName not in categoryNameToId:
        
        categoryId = nextCategoryId
        nextCategoryId += 1
        categoryNameToId[categoryName] = categoryId
        newCat = {}
        newCat['id'] = categoryNameToId[categoryName]
        newCat['name'] = categoryName
        newCat['count'] = 0
        categoriesSet1.append(newCat) 
        idToCategory[categoryId] = newCat
        category = newCat
        
    else:
        
        categoryId = categoryNameToId[categoryName]
        category = idToCategory[categoryId]
        
        # This image may still be empty...
        # category['count'] = category['count'] + 1
                
    # If we have bounding boxes, create image-level annotations    
    if relPath in relPathToMetadataSet1:
        
        nFoundMetadata += 1
        
        # This tuple is:
        #
        # filename (possibly no longer correct)
        # number of bounding boxes
        # [...boxes (four values per box)]
        imageMetadata = relPathToMetadataSet1[relPath]
        
        nBoxes = int(imageMetadata[1])
        im['n_boxes'] = nBoxes
        
        if nBoxes == 0:
            
            ann = {}
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']
            ann['category_id'] = emptyCategoryId
            ann['sequence_level_annotation'] = False
            annotationsSet1.append(ann)
            emptyCat['count'] = emptyCat['count'] + 1
            nImageLevelEmpties += 1
            
        else:
            
            # This image is non-empty
            category['count'] = category['count'] + 1                        
            
            for iBox in range(0,nBoxes):
                                
                boxCoords = imageMetadata[2+(iBox*4):6+(iBox*4)]
                boxCoords = list(map(int, boxCoords))
                
                # Some redundant bounding boxes crept in, don't add them twice
                bRedundantBox = False
                
                # Check this bbox against previous bboxes
                #
                # Inefficient?  Yes.  In an important way?  No.
                for iBoxComparison in range(0,iBox):
                    assert iBox != iBoxComparison                        
                    boxCoordsComparison = imageMetadata[2+(iBoxComparison*4):6+(iBoxComparison*4)]
                    boxCoordsComparison = list(map(int, boxCoordsComparison))
                    if boxCoordsComparison == boxCoords:
                        # print('Warning: redundant box on image {}'.format(relPath))
                        bRedundantBox = True
                        nRedundantBoxes += 1
                        break
                
                if bRedundantBox:
                    imageIDsWithRedundantBoxes.add(im['id'])
                    continue
                    
                # Bounding box values are in absolute coordinates, with the origin 
                # at the upper-left of the image, as [xmin1 ymin1 xmax1 ymax1].
                #
                # Convert to floats and to x/y/w/h, as per CCT standard
                bboxW = boxCoords[2] - boxCoords[0]
                bboxH = boxCoords[3] - boxCoords[1]
                
                box = [boxCoords[0], boxCoords[1], bboxW, bboxH]
                box = list(map(float, box))
                
                ann = {}
                ann['id'] = str(uuid.uuid1())
                ann['image_id'] = im['id']
                ann['category_id'] = categoryId
                ann['sequence_level_annotation'] = False
                ann['bbox'] = box
                annotationsSet1.append(ann)
                nTotalBoxes += 1
            
            # ...for each box
            
        # if we do/don't have boxes for this image
        
    # Else create a sequence-level annotation
    else:
        
        ann = {}
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']
        ann['category_id'] = categoryId
        ann['sequence_level_annotation'] = True
        annotationsSet1.append(ann)
        nSequenceLevelAnnotations += 1
        
# ...for each image
        
print('Finished processing set 1, found metadata for {} of {} images'.format(nFoundMetadata,len(set1ImageIDs)))
print('Created {} annotations and {} boxes in {} categories'.format(
    len(annotationsSet1),nTotalBoxes,len(categoriesSet1)))
print('Found {} redundant annotations'.format(nRedundantBoxes))

assert len(annotationsSet1) == nSequenceLevelAnnotations + nTotalBoxes + nImageLevelEmpties
assert len(set1ImageIDs) == nSequenceLevelAnnotations + nFoundMetadata

print('Found {} images with redundant boxes'.format(len(imageIDsWithRedundantBoxes)))


#%% Write out the list of images with redundant boxes

imageFileNamesWithRedundantBoxes = []
for image_id in imageIDsWithRedundantBoxes:
    im = imageIdToImage[image_id]
    imageFileNamesWithRedundantBoxes.append(im['file_name'])
imageFileNamesWithRedundantBoxes.sort()

with open(set1RedundantBoxListFilename,'w') as f:
    for fn in imageFileNamesWithRedundantBoxes:
        f.write(fn + '\n')
    
    
#%% The 'count' field isn't really meaningful, delete it

# It's really the count of image-level annotations, not total images assigned to a class
for d in categoriesSet1:
    del d['count']
    
    
#%% Write output .json files

data = {}
data['info'] = infoSet1
data['images'] = imagesSet1
data['annotations'] = annotationsSet1
data['categories'] = categoriesSet1
json.dump(data, open(outputJsonFilenameSet1,'w'), indent=4)    
print('Finished writing json to {}'.format(outputJsonFilenameSet1))


#%% Consistency-check final set 1 .json file

from megadetector.data_management.databases import integrity_check_json_db
options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = imageDir
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False
options.nThreads = 10
sortedCategories,data,_ = integrity_check_json_db.integrity_check_json_db(outputJsonFilenameSet1, options)
sortedCategories


#%% Generate previews

from megadetector.visualization import visualize_db

output_dir = os.path.join(baseDir,'preview')

options = visualize_db.DbVizOptions()
options.num_to_visualize = 5000
options.sort_by_filename = False
options.classes_to_exclude = None
options.trim_to_images_with_bboxes = False
options.parallelize_rendering = True

htmlOutputFile,_ = visualize_db.visualize_db(outputJsonFilenameSet1,output_dir,imageDir,options)

from megadetector.utils.path_utils import open_file
open_file(htmlOutputFile)
