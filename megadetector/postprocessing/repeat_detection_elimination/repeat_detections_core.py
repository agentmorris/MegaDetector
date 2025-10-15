"""

repeat_detections_core.py

Core utilities shared by find_repeat_detections and remove_repeat_detections.

Nothing in this file (in fact nothing in this subpackage) will make sense until you read
the RDE user's guide:

https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination

"""

#%% Imports and environment

import os
import copy
import warnings
import sklearn.cluster
import numpy as np
import jsonpickle
import traceback
import pandas as pd
import json
import shutil

from tqdm import tqdm
from operator import attrgetter
from datetime import datetime
from itertools import compress

import pyqtree

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial

from megadetector.utils import path_utils
from megadetector.utils import ct_utils
from megadetector.postprocessing.load_api_results import load_api_results, write_api_results
from megadetector.postprocessing.postprocess_batch_results import is_sas_url
from megadetector.postprocessing.postprocess_batch_results import relative_sas_url
from megadetector.visualization.visualization_utils import open_image, render_detection_bounding_boxes
from megadetector.visualization import render_images_with_thumbnails
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.utils.path_utils import flatten_path
from megadetector.utils.ct_utils import invert_dictionary

# "PIL cannot read EXIF metainfo for the images"
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

# "Metadata Warning, tag 256 had too many entries: 42, expected 1"
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)

jsonpickle.set_encoder_options('json', sort_keys=True, indent=1)


#%% Constants

detection_index_file_name_base = 'detectionIndex.json'


#%% Classes

class RepeatDetectionOptions:
    """
    Options that control the behavior of repeat detection elimination
    """

    def __init__(self):

        #: Folder where images live; filenames in the MD results .json file should
        #: be relative to this folder.
        #:
        #: imageBase can also be a SAS URL, in which case some error-checking is
        #: disabled.
        self.imageBase = ''

        #: Folder where we should write temporary output.
        self.outputBase = ''

        #: Don't consider detections with confidence lower than this as suspicious
        self.confidenceMin = 0.1

        #: Don't consider detections with confidence higher than this as suspicious
        self.confidenceMax = 1.0

        #: What's the IOU threshold for considering two boxes the same?
        self.iouThreshold = 0.9

        #: How many occurrences of a single location (as defined by the IOU threshold)
        #: are required before we declare it suspicious?
        self.occurrenceThreshold = 20

        #: Ignore "suspicious" detections smaller than some size
        self.minSuspiciousDetectionSize = 0.0

        #: Ignore "suspicious" detections larger than some size; these are often animals
        #: taking up the whole image.  This is expressed as a fraction of the image size.
        self.maxSuspiciousDetectionSize = 0.2

        #: Ignore folders with more than this many images in them
        self.maxImagesPerFolder = None

        #: A list of category IDs (ints) that we don't want consider as candidate repeat detections.
        #:
        #: Typically used to say, e.g., "don't bother analyzing people or vehicles for repeat
        #: detections", which you could do by saying excludeClasses = [2,3].
        self.excludeClasses = []

        #: For very large sets of results, passing chunks of results to and from workers as
        #: parameters ('memory') can be memory-intensive, so we can serialize to intermediate
        #: files instead ('file').
        #:
        #: The use of 'file' here is still experimental.
        self.pass_detections_to_processes_method = 'memory'

        #: Number of workers to use for parallel operations
        self.nWorkers = 10

        #: Should we use threads (True) or processes (False) for parallelization?
        #:
        #: Not relevant if nWorkers <= 1, or if bParallelizeComparisons and
        #: bParallelizeRendering are both False.
        self.parallelizationUsesThreads = True

        #: If this is not empty, we'll load detections from a filter file rather than finding them
        #: from the detector output.  This should be a .json file containing detections, generally this
        #: is the detectionIndex.json file in the filtering_* folder produced by find_repeat_detections().
        self.filterFileToLoad = ''

        #: (optional) List of filenames remaining after deletion of identified
        #: repeated detections that are actually animals.  This should be a flat
        #: text file, one relative filename per line.
        #:
        #: This is a pretty esoteric code path and a candidate for removal.
        #:
        #: The scenario where I see it being most useful is the very hypothetical one
        #: where we use an external tool for image handling that allows us to do something
        #: smarter and less destructive than deleting images to mark them as non-false-positives.
        self.filteredFileListToLoad = None

        #: Should we write the folder of images used to manually review repeat detections?
        self.bWriteFilteringFolder = True

        #: For debugging: limit comparisons to a specific number of folders
        self.debugMaxDir = -1

        #: For debugging: limit rendering to a specific number of folders
        self.debugMaxRenderDir = -1

        #: For debugging: limit comparisons to a specific number of detections
        self.debugMaxRenderDetection = -1

        #: For debugging: limit comparisons to a specific number of instances
        self.debugMaxRenderInstance = -1

        #: Should we parallelize (across cameras) comparisons to find repeat detections?
        self.bParallelizeComparisons = True

        #: Should we parallelize image rendering?
        self.bParallelizeRendering = True

        #: If this is False (default), a detection from class A is *not* considered to be "the same"
        #: as a detection from class B, even if they're at the same location.
        self.categoryAgnosticComparisons = False

        #: Determines whether bounding-box rendering errors (typically network errors) should
        #: be treated as failures
        self.bFailOnRenderError = False

        #: Should we print a warning if images referred to in the MD results file are missing?
        self.bPrintMissingImageWarnings = True

        #: If bPrintMissingImageWarnings is True, should we print a warning about missing images
        #: just once ('once') or every time ('all')?
        self.missingImageWarningType = 'once'  # 'all'

        #: Image width for rendered images (it's called "max" because we don't resize smaller images).
        #:
        #: Original size is preserved if this is None.
        #:
        #: This does *not* include the tile image grid.
        self.maxOutputImageWidth = 2000

        #: Line thickness (in pixels) for box rendering
        self.lineThickness = 10

        #: Box expansion (in pixels)
        self.boxExpansion = 2

        #: Progress bar used during comparisons and rendering.  Do not set externally.
        #:
        #: :meta private:
        self.pbar = None

        #: Replace filename tokens after reading, useful when the directory structure
        #: has changed relative to the structure the detector saw.
        self.filenameReplacements = {}

        #: How many folders up from the leaf nodes should we be going to aggregate images into
        #: cameras?
        #:
        #: If this is zero, each leaf folder is treated as a camera.
        self.nDirLevelsFromLeaf = 0

        #: An optional function that takes a string (an image file name) and returns
        #: a string (the corresponding  folder ID), typically used when multiple folders
        #: actually correspond to the same camera in a manufacturer-specific way (e.g.
        #: a/b/c/RECONYX100 and a/b/c/RECONYX101 may really be the same camera).
        #:
        #: See ct_utils for a common replacement function that handles most common
        #: manufacturer folder names:
        #:
        #: from megadetector.utils import ct_utils
        #: self.customDirNameFunction = ct_utils.image_file_to_camera_folder
        self.customDirNameFunction = None

        #: Include only specific folders, mutually exclusive with [excludeFolders]
        self.includeFolders = None

        #: Exclude specific folders, mutually exclusive with [includeFolders]
        self.excludeFolders = None

        #: Optionally show *other* detections (i.e., detections other than the
        #: one the user is evaluating), typically in a light gray.
        self.bRenderOtherDetections = False

        #: Threshold to use for *other* detections
        self.otherDetectionsThreshold = 0.2

        #: Line width (in pixels) for *other* detections
        self.otherDetectionsLineWidth = 1

        #: Optionally show a grid that includes a sample image for the detection, plus
        #: the top N additional detections
        self.bRenderDetectionTiles = True

        #: Width of the original image (within the larger output image) when bRenderDetectionTiles
        #: is True.
        #:
        #: If this is None, we'll render the original image in the detection tile image
        #: at its original width.
        self.detectionTilesPrimaryImageWidth = None

        #: Width to use for the grid of detection instances.
        #:
        #: Can be a width in pixels, or a number from 0 to 1 representing a fraction
        #: of the primary image width.
        #:
        #: If you want to render the grid at exactly 1 pixel wide, I guess you're out
        #: of luck.
        self.detectionTilesCroppedGridWidth = 0.6

        #: Location of the primary image within the mosaic ('right' or 'left)
        self.detectionTilesPrimaryImageLocation = 'right'

        #: Maximum number of individual detection instances to include in the mosaic
        self.detectionTilesMaxCrops = 150

        #: If bRenderOtherDetections is True, what color should we use to render the
        #: (hopefully pretty subtle) non-target detections?
        #:
        #: In theory I'd like these "other detection" rectangles to be partially
        #: transparent, but this is not straightforward, and the alpha is ignored
        #: here.  But maybe if I leave it here and wish hard enough, someday it
        #: will work.
        #:
        #: otherDetectionsColors = ['dimgray']
        self.otherDetectionsColors = [(105,105,105,100)]

        #: Sort detections within a directory so nearby detections are adjacent
        #: in the list, for faster review.
        #:
        #: Can be None, 'xsort', or 'clustersort'
        #:
        #: * None sorts detections chronologically by first occurrence
        #: * 'xsort' sorts detections from left to right
        #: * 'clustersort' clusters detections and sorts by cluster
        self.smartSort = 'xsort'

        #: Only relevant if smartSort == 'clustersort'
        self.smartSortDistanceThreshold = 0.1


class RepeatDetectionResults:
    """
    The results of an entire repeat detection analysis
    """

    def __init__(self):

        #: The data table (Pandas DataFrame), as loaded from the input json file via
        #: load_api_results().  Has columns ['file', 'detections','failure'].
        self.detectionResults = None

        #: The other fields in the input json file, loaded via load_api_results()
        self.otherFields = None

        #: The data table after modification
        self.detectionResultsFiltered = None

        #: dict mapping folder names to whole rows from the data table
        self.rows_by_directory = None

        #: dict mapping filenames to rows in the master table
        self.filename_to_row = None

        #: An array of length nDirs, where each element is a list of DetectionLocation
        #: objects for that directory that have been flagged as suspicious
        self.suspicious_detections = None

        #: The location of the .json file written with information about the RDE
        #: review images (typically detectionIndex.json)
        self.filterFile = None


class IndexedDetection:
    """
    A single detection event on a single image
    """

    def __init__(self, i_detection=-1, filename='', bbox=None, confidence=-1, category='unknown'):

        if bbox is None:
            bbox = []
        assert isinstance(i_detection,int)
        assert isinstance(filename,str)
        assert isinstance(bbox,list)
        assert isinstance(category,str)

        #: index of this detection within all detections for this filename
        self.i_detection = i_detection

        #: path to the image corresponding to this detection
        self.filename = filename

        #: [x_min, y_min, width_of_box, height_of_box]
        self.bbox = bbox

        #: confidence value of this detection
        self.confidence = confidence

        #: category ID (not name) of this detection
        self.category = category

    def __repr__(self):
        s = ct_utils.pretty_print_object(self, False)
        return s


class DetectionLocation:
    """
    A unique-ish detection location, meaningful in the context of one
    directory.  All detections within an IoU threshold of self.bbox
    will be stored in IndexedDetection objects.
    """

    def __init__(self, instance, detection, relative_dir, category, id=None):

        assert isinstance(detection,dict)
        assert isinstance(instance,IndexedDetection)
        assert isinstance(relative_dir,str)
        assert isinstance(category,str)

        #: list of IndexedDetections that match this detection
        self.instances = [instance]

        #: category ID (not name) for this detection
        self.category = category

        #: bbox as x,y,w,h
        self.bbox = detection['bbox']

        #: relative folder (i.e., camera name) in which this detectin was found
        self.relativeDir = relative_dir

        #: relative path to the canonical image representing this detection
        self.sampleImageRelativeFileName = ''

        #: list of detections on that canonical image that match this detection
        self.sampleImageDetections = None

        #: ID for this detection; this ID is only guaranteed to be unique within a directory
        self.id = id

        #: only used when doing cluster-based sorting
        self.clusterLabel = None

    def __repr__(self):
        s = ct_utils.pretty_print_object(self, False)
        return s

    def to_api_detection(self):
        """
        Converts this detection to a 'detection' dictionary, making the semi-arbitrary
        assumption that the first instance is representative of confidence.

        Returns:
            dict: dictionary in the format used to store detections in MD results
        """

        # This is a bit of a hack right now, but for future-proofing, I don't want to call this
        # to retrieve anything other than the highest-confidence detection, and I'm assuming this
        # is already sorted, so assert() that.
        confidences = [i.confidence for i in self.instances]
        assert confidences[0] == max(confidences), \
            'Cannot convert an unsorted DetectionLocation to an API detection'

        # It's not clear whether it's better to use instances[0].bbox or self.bbox
        # here... they should be very similar, unless iouThreshold is very low.
        # self.bbox is a better representation of the overall DetectionLocation.
        detection = {'conf':self.instances[0].confidence,
                     'bbox':self.bbox,'category':self.instances[0].category}
        return detection


#%% Support functions

def _render_bounding_box(detection,
                         input_file_name,
                         output_file_name,
                         line_width=5,
                         expansion=0):
    """
    Rendering the detection [detection] on the image [input_file_name], writing the result
    to [output_file_name].
    """

    im = open_image(input_file_name)
    d = detection.to_api_detection()
    render_detection_bounding_boxes([d],im,thickness=line_width,expansion=expansion,
                                    confidence_threshold=-10)
    im.save(output_file_name)


def _detection_rect_to_rtree_rect(detection_rect):
    """
    We store detections as x/y/w/h, rtree and pyqtree use l/b/r/t.  Convert from
    our representation to rtree's.
    """

    left = detection_rect[0]
    bottom = detection_rect[1]
    right = detection_rect[0] + detection_rect[2]
    top = detection_rect[1] + detection_rect[3]
    return (left,bottom,right,top)


def _rtree_rect_to_detection_rect(rtree_rect):
    """
    We store detections as x/y/w/h, rtree and pyqtree use l/b/r/t.  Convert from
    rtree's representation to ours.
    """

    x = rtree_rect[0]
    y = rtree_rect[1]
    w = rtree_rect[2] - rtree_rect[0]
    h = rtree_rect[3] - rtree_rect[1]
    return (x,y,w,h)


def _sort_detections_for_directory(candidate_detections,options):
    """
    candidate_detections is a list of DetectionLocation objects.  Sorts them to
    put nearby detections next to each other, for easier visual review.  Returns
    a sorted copy of candidate_detections, does not sort in-place.
    """

    if len(candidate_detections) <= 1 or options.smartSort is None:
        return candidate_detections

    # Just sort by the X location of each box
    if options.smartSort == 'xsort':
        candidate_detections_sorted = sorted(candidate_detections,
                                           key=lambda x: (
                                               (x.bbox[0]) + (x.bbox[2]/2.0)
                                               ))
        return candidate_detections_sorted

    elif options.smartSort == 'clustersort':

        cluster = sklearn.cluster.AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=options.smartSortDistanceThreshold,
            linkage='complete')

        # Prepare a list of points to represent each box,
        # that's what we'll use for clustering
        points = []
        for det in candidate_detections:
            # To use the upper-left of the box as the clustering point
            # points.append([det.bbox[0],det.bbox[1]])

            # To use the center of the box as the clustering point
            points.append([det.bbox[0]+det.bbox[2]/2.0,
                           det.bbox[1]+det.bbox[3]/2.0])
        points_array = np.array(points)

        labels = cluster.fit_predict(points_array)
        unique_labels = np.unique(labels)

        # Labels *could* be any unique labels according to the docs, but in practice
        # they are unique integers from 0:nClusters.
        #
        # Make sure the labels are unique incrementing integers.
        for i_label in range(1,len(unique_labels)):
            assert unique_labels[i_label] == 1 + unique_labels[i_label-1]

        assert len(labels) == len(candidate_detections)

        # Store the label assigned to each cluster
        for i_label,label in enumerate(labels):
            candidate_detections[i_label].clusterLabel = label

        # Now sort the clusters by their x coordinate, and re-assign labels
        # so the labels are sortable
        label_x_means = []

        for label in unique_labels:
            detections_this_label = [d for d in candidate_detections if (
                d.clusterLabel == label)]
            points_this_label = [ [d.bbox[0],d.bbox[1]] for d in detections_this_label]
            x = [p[0] for p in points_this_label]
            y = [p[1] for p in points_this_label]

            # Compute the centroid for debugging, but we're only going to use the x
            # coordinate.  This is the centroid of points used to represent detections,
            # which may be box centers or box corners.
            centroid = [ sum(x) / len(points_this_label), sum(y) / len(points_this_label) ]
            label_xval = centroid[0]
            label_x_means.append(label_xval)

        old_cluster_label_to_new_cluster_label = {}
        new_cluster_labels = np.argsort(label_x_means)
        assert len(new_cluster_labels) == len(np.unique(new_cluster_labels))
        for old_cluster_label in unique_labels:
            old_cluster_label_to_new_cluster_label[old_cluster_label] =\
                np.where(new_cluster_labels==old_cluster_label)[0][0]

        for i_cluster in range(0,len(unique_labels)):
            old_label = unique_labels[i_cluster]
            assert i_cluster == old_label
            new_label = old_cluster_label_to_new_cluster_label[old_label]

        for i_det,det in enumerate(candidate_detections):
            old_label = det.clusterLabel
            new_label = old_cluster_label_to_new_cluster_label[old_label]
            det.clusterLabel = new_label

        candidate_detections_sorted = sorted(candidate_detections,
                                           key=lambda x: (x.clusterLabel,x.id))

        return candidate_detections_sorted

    else:
        raise ValueError('Unrecognized sort method {}'.format(
            options.smartSort))

# ...def _sort_detections_for_directory(...)


def _find_matches_in_directory(dir_name_and_rows, options):
    """
    dir_name_and_rows is a tuple of (name,rows).

    "name" is a location name, typically a folder name, though this may be an arbitrary
    location identifier.

    "rows" is a Pandas dataframe with one row per image in this location, with columns:

        * 'file': relative file name
        * 'detections': a list of MD detection objects, i.e. dicts with keys ['category','conf','bbox']
        * 'max_detection_conf': maximum confidence of any detection, in any category

    "rows" can also point to a .csv file, in which case the detection table will be read from that
    .csv file, and results will be written to a .csv file rather than being returned.

    Find all unique detections in this directory.

    Returns a list of DetectionLocation objects.
    """

    if options.pbar is not None:
        options.pbar.update()

    # Create a tree to store candidate detections
    candidate_detections_index = pyqtree.Index(bbox=(-0.1,-0.1,1.1,1.1))

    assert len(dir_name_and_rows) == 2, 'find_matches_in_directory: invalid input'
    assert isinstance(dir_name_and_rows[0],str), 'find_matches_in_directory: invalid location name'
    dir_name = dir_name_and_rows[0]
    rows = dir_name_and_rows[1]

    detections_loaded_from_csv_file = None

    if isinstance(rows,str):
        detections_loaded_from_csv_file = rows
        print('Loading results for location {} from {}'.format(
            dir_name,detections_loaded_from_csv_file))
        rows = pd.read_csv(detections_loaded_from_csv_file)
        # Pandas writes out detections out as strings, convert them back to lists
        rows['detections'] = rows['detections'].apply(lambda s: json.loads(s.replace('\'','"')))

    if options.maxImagesPerFolder is not None and len(rows) > options.maxImagesPerFolder:
        print('Ignoring directory {} because it has {} images (limit set to {})'.format(
            dir_name,len(rows),options.maxImagesPerFolder))
        return []

    if options.includeFolders is not None:
        assert options.excludeFolders is None, 'Cannot specify include and exclude folder lists'
        if dir_name not in options.includeFolders:
            print('Ignoring folder {}, not in inclusion list'.format(dir_name))
            return []

    if options.excludeFolders is not None:
        assert options.includeFolders is None, 'Cannot specify include and exclude folder lists'
        if dir_name in options.excludeFolders:
            print('Ignoring folder {}, on exclusion list'.format(dir_name))
            return []

    # For each image in this directory
    #
    # i_directory_row = 0; row = rows.iloc[i_directory_row]
    #
    # i_directory_row is a pandas index, so it may not start from zero;
    # for debugging, we maintain i_iteration as a loop index.
    i_iteration = -1
    n_boxes_evaluated = 0

    for i_directory_row, row in rows.iterrows():

        i_iteration += 1
        filename = row['file']
        if not path_utils.is_image_file(filename):
            continue

        if 'max_detection_conf' not in row or 'detections' not in row or \
            row['detections'] is None:
            # print('Skipping row {}'.format(i_directory_row))
            continue

        # Don't bother checking images with no detections above threshold
        max_p = float(row['max_detection_conf'])
        if max_p < options.confidenceMin:
            continue

        # Array of dicts, where each element is
        # {
        #   'category': '1',  # str value, category ID
        #   'conf': 0.926,    # confidence of this detections
        #
        #    (x_min, y_min) is upper-left, all in relative coordinates
        #   'bbox': [x_min, y_min, width_of_box, height_of_box]
        #
        # }
        detections = row['detections']
        if isinstance(detections,float):
            assert isinstance(row['failure'],str), 'Expected failure indicator'
            print('Skipping failed image {} ({})'.format(filename,row['failure']))
            continue

        assert len(detections) > 0

        # For each detection in this image
        for i_detection, detection in enumerate(detections):

            n_boxes_evaluated += 1

            if detection is None:
                print('Skipping detection {}'.format(i_detection))
                continue

            assert 'category' in detection and \
                'conf' in detection and \
                'bbox' in detection, 'Illegal detection'

            confidence = detection['conf']

            # This is no longer strictly true; I sometimes run RDE in stages, so
            # some probabilities have already been made negative
            #
            # assert confidence >= 0.0 and confidence <= 1.0

            assert confidence >= -1.0 and confidence <= 1.0

            if confidence < options.confidenceMin:
                continue
            if confidence > options.confidenceMax:
                continue

            # Optionally exclude some classes from consideration as suspicious
            if (options.excludeClasses is not None) and (len(options.excludeClasses) > 0):
                i_class = int(detection['category'])
                if i_class in options.excludeClasses:
                    continue

            bbox = detection['bbox']
            confidence = detection['conf']

            # Is this detection too big or too small for consideration?
            w, h = bbox[2], bbox[3]

            if (w == 0 or h == 0):
                continue

            area = h * w

            if area < 0:
                print('Warning: negative-area bounding box for file {}'.format(filename))
                area = abs(area); h = abs(h); w = abs(w)

            assert area >= 0.0 and area <= 1.0, \
                'Illegal bounding box area {} in image {}'.format(area,filename)

            if area < options.minSuspiciousDetectionSize:
                continue

            if area > options.maxSuspiciousDetectionSize:
                continue

            category = detection['category']

            instance = IndexedDetection(i_detection=i_detection,
                                        filename=row['file'], bbox=bbox,
                                        confidence=confidence, category=category)

            b_found_similar_detection = False

            rtree_rect = _detection_rect_to_rtree_rect(bbox)

            # This will return candidates of all classes
            overlapping_candidate_detections =\
                candidate_detections_index.intersect(rtree_rect)

            overlapping_candidate_detections.sort(
                key=lambda x: x.id, reverse=False)

            # For each detection in our candidate list
            for i_candidate, candidate in enumerate(
                    overlapping_candidate_detections):

                # Don't match across categories
                if (candidate.category != category) and (not (options.categoryAgnosticComparisons)):
                    continue

                # Is this a match?
                try:
                    iou = ct_utils.get_iou(bbox, candidate.bbox)
                except Exception as e:
                    print(\
                    'Warning: IOU computation error on boxes ({},{},{},{}),({},{},{},{}): {}'.\
                        format(
                        bbox[0],bbox[1],bbox[2],bbox[3],
                        candidate.bbox[0],candidate.bbox[1],
                        candidate.bbox[2],candidate.bbox[3], str(e)))
                    continue

                if iou >= options.iouThreshold:

                    b_found_similar_detection = True

                    # If so, add this example to the list for this detection
                    candidate.instances.append(instance)

                    # We *don't* break here; we allow this instance to possibly
                    # match multiple candidates.  There isn't an obvious right or
                    # wrong here.

            # ...for each detection on our candidate list

            # If we found no matches, add this to the candidate list
            if not b_found_similar_detection:

                candidate = DetectionLocation(instance=instance,
                                              detection=detection,
                                              relative_dir=dir_name,
                                              category=category,
                                              id=i_iteration)

                # pyqtree
                candidate_detections_index.insert(item=candidate,bbox=rtree_rect)

        # ...for each detection

    # ...for each row

    # Get all candidate detections

    candidate_detections = candidate_detections_index.intersect([-100,-100,100,100])

    # For debugging only, it's convenient to have these sorted
    # as if they had never gone into a tree structure.  Typically
    # this is in practice a sort by filename.
    candidate_detections.sort(
        key=lambda x: x.id, reverse=False)

    if detections_loaded_from_csv_file is not None:
        location_results_file = \
            os.path.splitext(detections_loaded_from_csv_file)[0] + \
            '_results.json'
        print('Writing results for location {} to {}'.format(
            dir_name,location_results_file))
        s = jsonpickle.encode(candidate_detections,make_refs=False)
        with open(location_results_file,'w') as f:
            f.write(s)
            # json.dump(candidate_detections,f,indent=1)
        return location_results_file
    else:
        return candidate_detections

# ...def _find_matches_in_directory(...)


def _update_detection_table(repeat_detection_results, options, output_file_name=None):
    """
    Changes confidence values in repeat_detection_results.detectionResults so that detections
    deemed to be possible false positives are given negative confidence values.

    repeat_detection_results is an object of type RepeatDetectionResults, with a pandas
    dataframe (detectionResults) containing all the detections loaded from the .json file,
    and a list of detections for each location (suspicious_detections) that are deemed to
    be suspicious.

    returns the modified pandas dataframe (repeat_detection_results.detectionResults), but
    also modifies it in place.
    """

    # This is the pandas dataframe that contains actual detection results.
    #
    # Has fields ['file', 'detections','failure'].
    detection_results = repeat_detection_results.detectionResults

    # An array of length nDirs, where each element is a list of DetectionLocation
    # objects for that directory that have been flagged as suspicious
    suspicious_detections_by_directory = repeat_detection_results.suspicious_detections

    n_bbox_changes = 0

    print('Updating output table')

    # For each directory
    for i_dir, directory_events in enumerate(suspicious_detections_by_directory):

        # For each suspicious detection group in this directory
        for i_detection_event, detection_event in enumerate(directory_events):

            location_bbox = detection_event.bbox

            # For each instance of this suspicious detection
            for i_instance, instance in enumerate(detection_event.instances):

                instance_bbox = instance.bbox

                # This should match the bbox for the detection event
                iou = ct_utils.get_iou(instance_bbox, location_bbox)

                # The bbox for this instance should be almost the same as the bbox
                # for this detection group, where "almost" is defined by the IOU
                # threshold.
                assert iou >= options.iouThreshold
                # if iou < options.iouThreshold:
                #    print('IOU warning: {},{}'.format(iou,options.iouThreshold))

                assert instance.filename in repeat_detection_results.filename_to_row
                i_row = repeat_detection_results.filename_to_row[instance.filename]
                row = detection_results.iloc[i_row]
                row_detections = row['detections']
                detection_to_modify = row_detections[instance.i_detection]

                # Make sure the bounding box matches
                assert (instance_bbox[0:4] == detection_to_modify['bbox'][0:4])

                # Make the probability negative, if it hasn't been switched by
                # another bounding box
                if detection_to_modify['conf'] >= 0:
                    detection_to_modify['conf'] = -1 * detection_to_modify['conf']
                    n_bbox_changes += 1

            # ...for each instance

        # ...for each detection

    # ...for each directory

    # Update maximum probabilities

    # For each row...
    n_prob_changes = 0
    n_prob_changes_to_negative = 0
    n_prob_changes_across_threshold = 0

    for i_row, row in detection_results.iterrows():

        detections = row['detections']
        if (detections is None) or isinstance(detections,float):
            assert isinstance(row['failure'],str)
            continue

        if len(detections) == 0:
            continue

        max_p_original = float(row['max_detection_conf'])

        # No longer strictly true; sometimes I run RDE on RDE output
        # assert max_p_original >= 0
        assert max_p_original >= -1.0

        max_p = None
        n_negative = 0

        for i_detection, detection in enumerate(detections):

            p = detection['conf']

            if p < 0:
                n_negative += 1

            if (max_p is None) or (p > max_p):
                max_p = p

        # We should only be making detections *less* likely in this process
        assert max_p <= max_p_original
        detection_results.at[i_row, 'max_detection_conf'] = max_p

        # If there was a meaningful change, count it
        if abs(max_p - max_p_original) > 1e-3:

            assert max_p < max_p_original

            n_prob_changes += 1

            if (max_p < 0) and (max_p_original >= 0):
                n_prob_changes_to_negative += 1

            if (max_p_original >= options.confidenceMin) and (max_p < options.confidenceMin):
                n_prob_changes_across_threshold += 1

            # Negative probabilities should be the only reason max_p changed, so
            # we should have found at least one negative value if we reached
            # this point.
            assert n_negative > 0

        # ...if there was a meaningful change to the max probability for this row

    # ...for each row

    # If we're also writing output...
    if output_file_name is not None and len(output_file_name) > 0:
        write_api_results(detection_results, repeat_detection_results.otherFields,
                          output_file_name)

    print(
        'Finished updating detection table\nChanged {} detections that impacted {} max_ps ({} to negative) ({} across confidence threshold)'.format( # noqa
            n_bbox_changes, n_prob_changes, n_prob_changes_to_negative, n_prob_changes_across_threshold))

    return detection_results

# ...def _update_detection_table(...)


def _render_sample_image_for_detection(detection,filtering_dir,options):
    """
    Render a sample image for one unique detection, possibly containing lightly-colored
    high-confidence detections from elsewhere in the sample image.

    "detections" is a DetectionLocation object.

    Depends on having already sorted instances within this detection by confidence, and
    having already generated an output file name for this sample image.
    """

    # Confidence values should already have been sorted in the previous loop
    instance_confidences = [instance.confidence for instance in detection.instances]
    assert ct_utils.is_list_sorted(instance_confidences,reverse=True)

    # Choose the highest-confidence index
    instance = detection.instances[0]
    relative_path = instance.filename

    output_relative_path = detection.sampleImageRelativeFileName
    assert len(output_relative_path) > 0

    output_full_path = os.path.join(filtering_dir, output_relative_path)

    if is_sas_url(options.imageBase):
        input_full_path = relative_sas_url(options.imageBase, relative_path)
    else:
        input_full_path = os.path.join(options.imageBase, relative_path)
        assert (os.path.isfile(input_full_path)), 'Not a file: {}'.\
            format(input_full_path)

    try:

        im = open_image(input_full_path)

        # Should we render (typically in a very light color) detections
        # *other* than the one we're highlighting here?
        if options.bRenderOtherDetections:

            # Optionally resize the output image
            if (options.maxOutputImageWidth is not None) and \
                (im.size[0] > options.maxOutputImageWidth):
                im = vis_utils.resize_image(im, options.maxOutputImageWidth,
                                            target_height=-1)

            assert detection.sampleImageDetections is not None

            # At this point, suspicious detections have already been flipped
            # negative, which we don't want for rendering purposes
            rendered_detections = []

            for det in detection.sampleImageDetections:
                rendered_det = copy.copy(det)
                rendered_det['conf'] = abs(rendered_det['conf'])
                rendered_detections.append(rendered_det)

            # Render other detections first (typically in a thin+light box)
            render_detection_bounding_boxes(rendered_detections,
                im,
                label_map=None,
                thickness=options.otherDetectionsLineWidth,
                expansion=options.boxExpansion,
                colormap=options.otherDetectionsColors,
                confidence_threshold=options.otherDetectionsThreshold)

            # Now render the example detection (on top of at least one
            # of the other detections)

            # This converts the *first* instance to an API standard detection;
            # because we just sorted this list in descending order by confidence,
            # this is the highest-confidence detection.
            d = detection.to_api_detection()

            render_detection_bounding_boxes([d],im,thickness=options.lineThickness,
                                            expansion=options.boxExpansion,
                                            confidence_threshold=-10)

            im.save(output_full_path)

        else:

            _render_bounding_box(detection,
                                 input_full_path,
                                 output_full_path,
                                 line_width=options.lineThickness,
                                 expansion=options.boxExpansion)

        # ...if we are/aren't rendering other bounding boxes

        # If we're rendering detection tiles, we'll re-load and re-write the image we
        # just wrote to output_full_path
        if options.bRenderDetectionTiles:

            assert not is_sas_url(options.imageBase), "Can't render detection tiles from SAS URLs"

            if options.detectionTilesPrimaryImageWidth is not None:
                primary_image_width = options.detectionTilesPrimaryImageWidth
            else:
                # "im" may be a resized version of the original image, if we've already run
                # the code to render other bounding boxes.
                primary_image_width = im.size[0]

            if options.detectionTilesCroppedGridWidth <= 1.0:
                cropped_grid_width = \
                    round(options.detectionTilesCroppedGridWidth * primary_image_width)
            else:
                cropped_grid_width = options.detectionTilesCroppedGridWidth

            secondary_image_filename_list = []
            secondary_image_bounding_box_list = []

            # If we start from zero, we include the sample crop
            for instance in detection.instances[0:]:
                secondary_image_filename_list.append(os.path.join(options.imageBase,
                                                               instance.filename))
                secondary_image_bounding_box_list.append(instance.bbox)

            # Optionally limit the number of crops we pass to the rendering function
            if (options.detectionTilesMaxCrops is not None) and \
                (len(detection.instances) > options.detectionTilesMaxCrops):
                    secondary_image_filename_list = \
                        secondary_image_filename_list[0:options.detectionTilesMaxCrops]
                    secondary_image_bounding_box_list = \
                        secondary_image_bounding_box_list[0:options.detectionTilesMaxCrops]

            # This will over-write the image we've already written to output_full_path
            render_images_with_thumbnails.render_images_with_thumbnails(
                primary_image_filename=output_full_path,
                primary_image_width=primary_image_width,
                secondary_image_filename_list=secondary_image_filename_list,
                secondary_image_bounding_box_list=secondary_image_bounding_box_list,
                cropped_grid_width=cropped_grid_width,
                output_image_filename=output_full_path,
                primary_image_location=options.detectionTilesPrimaryImageLocation)

        # ...if we are/aren't rendering detection tiles

    except Exception as e:

        stack_trace = traceback.format_exc()
        print('Warning: error rendering bounding box from {} to {}: {} ({})'.format(
            input_full_path,output_full_path,e,stack_trace))
        if options.bFailOnRenderError:
            raise

# ...def _render_sample_image_for_detection(...)


#%% Main entry point

def find_repeat_detections(input_filename, output_file_name=None, options=None):
    """
    Find detections in a MD results file that occur repeatedly and are likely to be
    rocks/sticks.

    Args:
        input_filename (str): the MD results .json file to analyze
        output_file_name (str, optional): the filename to which we should write results
            with repeat detections removed, typically set to None during the first
            part of the RDE process.
        options (RepeatDetectionOptions, optional): all the interesting options controlling
            this process; see RepeatDetectionOptions for details.

    Returns:
        RepeatDetectionResults: results of the RDE process; see RepeatDetectionResults
        for details.
    """

    ##%% Input handling

    if options is None:

        options = RepeatDetectionOptions()

    # Validate some options

    if options.customDirNameFunction is not None:
        assert options.nDirLevelsFromLeaf == 0, \
            'Cannot mix custom dir name functions with nDirLevelsFromLeaf'

    if options.nDirLevelsFromLeaf != 0:
        assert options.customDirNameFunction is None, \
            'Cannot mix custom dir name functions with nDirLevelsFromLeaf'

    if options.filterFileToLoad is not None and len(options.filterFileToLoad) > 0:

        print('Bypassing detection-finding, loading from {}'.format(options.filterFileToLoad))

        # Load the filtering file
        detection_index_file_name = options.filterFileToLoad
        with open(detection_index_file_name, 'r') as f:
            s_in = f.read()
        detection_info = jsonpickle.decode(s_in)
        filtering_base_dir = os.path.dirname(options.filterFileToLoad)
        suspicious_detections = detection_info['suspicious_detections']

        # Load the same options we used when finding repeat detections
        options = detection_info['options']

        # ...except for things that explicitly tell this function not to
        # find repeat detections.
        options.filterFileToLoad = detection_index_file_name
        options.bWriteFilteringFolder = False

    # ...if we're loading from an existing filtering file

    to_return = RepeatDetectionResults()


    # Check early to avoid problems with the output folder

    if options.bWriteFilteringFolder:
        assert options.outputBase is not None and len(options.outputBase) > 0
        os.makedirs(options.outputBase,exist_ok=True)


    # Load file to a pandas dataframe.  Also populates 'max_detection_conf', even if it's
    # not present in the .json file.
    detection_results, other_fields = load_api_results(input_filename, normalize_paths=True,
                                                       filename_replacements=options.filenameReplacements,
                                                       force_forward_slashes=True)
    to_return.detectionResults = detection_results
    to_return.otherFields = other_fields

    # Before doing any real work, make sure we can *probably* access images
    # This is just a cursory check on the first image, but it heads off most
    # problems related to incorrect mount points, etc.  Better to do this before
    # spending 20 minutes finding repeat detections.

    if options.bWriteFilteringFolder:

        if not is_sas_url(options.imageBase):

            row = detection_results.iloc[0]
            relative_path = row['file']
            if options.filenameReplacements is not None:
                for s in options.filenameReplacements.keys():
                    relative_path = relative_path.replace(s,options.filenameReplacements[s])
            absolute_path = os.path.join(options.imageBase,relative_path)
            assert os.path.isfile(absolute_path), 'Could not find file {}'.format(absolute_path)


    ##%% Separate files into locations

    # This will be a map from a directory name to smaller data frames
    rows_by_directory = {}

    # This is a mapping back into the rows of the original table
    filename_to_row = {}

    print('Separating images into locations...')

    n_custom_dir_replacements = 0

    # i_row = 0; row = detection_results.iloc[i_row]
    for i_row, row in tqdm(detection_results.iterrows(),total=len(detection_results)):

        relative_path = row['file']

        if options.customDirNameFunction is not None:
            basic_dir_name = os.path.dirname(relative_path.replace('\\','/'))
            dir_name = options.customDirNameFunction(relative_path)
            if basic_dir_name != dir_name:
                n_custom_dir_replacements += 1
        else:
            dir_name = os.path.dirname(relative_path)

        if len(dir_name) == 0:
            assert options.nDirLevelsFromLeaf == 0, \
                'Can''t use the dirLevelsFromLeaf option with flat filenames'
        else:
            if options.nDirLevelsFromLeaf > 0:
                i_level = 0
                while (i_level < options.nDirLevelsFromLeaf):
                    i_level += 1
                    dir_name = os.path.dirname(dir_name)
            assert len(dir_name) > 0

        if dir_name not in rows_by_directory:
            # Create a new DataFrame with just this row
            # rows_by_directory[dir_name] = pd.DataFrame(row)
            rows_by_directory[dir_name] = []

        rows_by_directory[dir_name].append(row)

        assert relative_path not in filename_to_row
        filename_to_row[relative_path] = i_row

    # ...for each unique detection

    if options.customDirNameFunction is not None:
        print('Custom dir name function made {} replacements (of {} images)'.format(
            n_custom_dir_replacements,len(detection_results)))

    # Convert lists of rows to proper DataFrames
    dirs = list(rows_by_directory.keys())
    for d in dirs:
        rows_by_directory[d] = pd.DataFrame(rows_by_directory[d])

    to_return.rows_by_directory = rows_by_directory
    to_return.filename_to_row = filename_to_row

    print('Finished separating {} files into {} locations'.format(len(detection_results),
                                                                  len(rows_by_directory)))

    ##% Look for repeat detections (or load them from file)

    dirs_to_search = list(rows_by_directory.keys())
    if options.debugMaxDir > 0:
        dirs_to_search = dirs_to_search[0:options.debugMaxDir]

    # Map numeric directory indices to names (we'll write this out to the detection index .json file)
    dir_index_to_name = {}
    for i_dir, dir_name in enumerate(dirs_to_search):
        dir_index_to_name[i_dir] = dir_name

    # Are we actually looking for matches, or just loading from a file?
    if len(options.filterFileToLoad) == 0:

        # length-nDirs list of lists of DetectionLocation objects
        suspicious_detections = [None] * len(dirs_to_search)

        # We're actually looking for matches...
        print('Finding similar detections...')

        dir_name_and_rows = []
        for dir_name in dirs_to_search:
            rows_this_directory = rows_by_directory[dir_name]
            dir_name_and_rows.append((dir_name,rows_this_directory))

        all_candidate_detections = [None] * len(dirs_to_search)

        # If we serialize results to intermediate files, we need to remove slashes from
        # location names; we store mappings here.
        normalized_location_name_to_location_name = None
        location_name_to_normalized_location_name = None

        if not options.bParallelizeComparisons:

            options.pbar = None
            for i_dir, dir_name in tqdm(enumerate(dirs_to_search)):
                dir_name_and_row = dir_name_and_rows[i_dir]
                assert dir_name_and_row[0] == dir_name
                print('Processing dir {} of {}: {}'.format(i_dir,len(dirs_to_search),dir_name))
                all_candidate_detections[i_dir] = \
                    _find_matches_in_directory(dir_name_and_row, options)

        else:

            n_workers = options.nWorkers
            if n_workers > len(dir_name_and_rows):
                print('Pool of {} requested, but only {} folders available, reducing pool to {}'.\
                      format(n_workers,len(dir_name_and_rows),len(dir_name_and_rows)))
                n_workers = len(dir_name_and_rows)

            pool = None

            if options.parallelizationUsesThreads:
                pool = ThreadPool(n_workers); poolstring = 'threads'
            else:
                pool = Pool(n_workers); poolstring = 'processes'

            print('Starting comparison pool with {} {}'.format(n_workers,poolstring))

            assert options.pass_detections_to_processes_method in ('file','memory'), \
                'Unrecognized IPC mechanism: {}'.format(options.pass_detections_to_processes_method)

            # ** Experimental **
            #
            # Rather than passing detections and results around in memory, write detections and
            # results for each worker to intermediate files.  May improve performance for very large
            # results sets that exceed working memory.
            if options.pass_detections_to_processes_method == 'file':

                ##%% Convert location names to normalized names we can write to files

                normalized_location_name_to_location_name = {}
                for dir_name in dirs_to_search:
                    normalized_location_name = flatten_path(dir_name)
                    assert normalized_location_name not in normalized_location_name_to_location_name, \
                        'Redundant location name {}, can\'t serialize to intermediate files'.format(
                            dir_name)
                    normalized_location_name_to_location_name[normalized_location_name] = dir_name

                location_name_to_normalized_location_name = \
                    invert_dictionary(normalized_location_name_to_location_name)


                ##%% Write results to files for each location

                print('Writing results to intermediate files')

                intermediate_json_file_folder = os.path.join(options.outputBase,'intermediate_results')
                os.makedirs(intermediate_json_file_folder,exist_ok=True)

                # i_location = 0; location_info = dir_name_and_rows[0]
                dir_name_and_intermediate_file = []

                # i_location = 0; location_info = dir_name_and_rows[i_location]
                for i_location, location_info in tqdm(enumerate(dir_name_and_rows)):

                    location_name = location_info[0]
                    assert location_name in location_name_to_normalized_location_name
                    normalized_location_name = location_name_to_normalized_location_name[location_name]
                    intermediate_results_file = os.path.join(intermediate_json_file_folder,
                                                             normalized_location_name + '.csv')
                    detections_table_this_location = location_info[1]
                    detections_table_this_location.to_csv(intermediate_results_file,header=True,index=False)
                    dir_name_and_intermediate_file.append((location_name,intermediate_results_file))


                ##%% Find detections in each directory

                options.pbar = None
                all_candidate_detection_files = list(pool.imap(
                    partial(_find_matches_in_directory,options=options), dir_name_and_intermediate_file))


                ##%% Load into a combined list of candidate detections

                all_candidate_detections = []

                # candidate_detection_file = all_candidate_detection_files[0]
                for candidate_detection_file in all_candidate_detection_files:
                    with open(candidate_detection_file, 'r') as f:
                        s = f.read()
                    candidate_detections_this_file = jsonpickle.decode(s)
                    all_candidate_detections.append(candidate_detections_this_file)


                ##%% Clean up intermediate files

                shutil.rmtree(intermediate_json_file_folder)

            # If we're passing things around in memory, rather than via intermediate files
            else:

                # We get slightly nicer progress bar behavior using threads, by passing a pbar
                # object and letting it get updated.  We can't serialize this object across
                # processes.
                if options.parallelizationUsesThreads:
                    options.pbar = tqdm(total=len(dir_name_and_rows))
                    all_candidate_detections = list(pool.imap(
                        partial(_find_matches_in_directory,options=options), dir_name_and_rows))
                else:
                    options.pbar = None
                    all_candidate_detections = list(tqdm(pool.imap(
                        partial(_find_matches_in_directory,options=options), dir_name_and_rows)))

        # ...if we're parallelizing comparisons

        if pool is not None:
            try:
                pool.close()
                pool.join()
                print("Pool closed and joined for RDE comparisons")
            except Exception as e:
                print('Warning: error closing RDE comparison pool: {}'.format(str(e)))

        print('\nFinished looking for similar detections')


        ##%% Mark suspicious locations based on match results

        print('Marking repeat detections...')

        n_images_with_suspicious_detections = 0
        n_suspicious_detections = 0

        # For each directory
        for i_dir in range(len(dirs_to_search)):

            # A list of DetectionLocation objects
            suspicious_detections_this_dir = []

            # A list of DetectionLocation objects
            candidate_detections_this_dir = all_candidate_detections[i_dir]

            for i_location, candidate_location in enumerate(candidate_detections_this_dir):

                # occurrenceList is a list of file/detection pairs
                n_occurrences = len(candidate_location.instances)

                if n_occurrences < options.occurrenceThreshold:
                    continue

                n_images_with_suspicious_detections += n_occurrences
                n_suspicious_detections += 1

                suspicious_detections_this_dir.append(candidate_location)

            suspicious_detections[i_dir] = suspicious_detections_this_dir

            # Sort the above-threshold detections for easier review
            if options.smartSort is not None:
                suspicious_detections[i_dir] = _sort_detections_for_directory(
                    suspicious_detections[i_dir],options)

            print('Found {} suspicious detections in directory {} ({})'.format(
                  len(suspicious_detections[i_dir]),i_dir,dirs_to_search[i_dir]))

        # ...for each directory

        print('Finished marking repeat detections')

        print('Found {} unique detections on {} images that are suspicious'.format(
                n_suspicious_detections, n_images_with_suspicious_detections))

    # If we're just loading detections from a file...
    else:

        assert len(suspicious_detections) == len(dirs_to_search)

        n_detections_removed = 0
        n_detections_loaded = 0

        # We're skipping detection-finding, but to see which images are actually legit false
        # positives, we may be looking for physical files or loading from a text file.
        file_list = None
        if options.filteredFileListToLoad is not None:
            with open(options.filteredFileListToLoad) as f:
                file_list = f.readlines()
                file_list = [x.strip() for x in file_list]
            n_suspicious_detections = sum([len(x) for x in suspicious_detections])
            print('Loaded false positive list from file ' + \
                  'will remove {} of {} suspicious detections'.format(
                len(file_list), n_suspicious_detections))

        # For each directory
        # i_dir = 0; detections = suspicious_detections[0]
        #
        # suspicious_detections is an array of DetectionLocation objects,
        # one per directory.
        for i_dir, detections in enumerate(suspicious_detections):

            b_valid_detection = [True] * len(detections)
            n_detections_loaded += len(detections)

            # For each detection that was present before filtering
            # i_detection = 0; detection = detections[i_detection]
            for i_detection, detection in enumerate(detections):

                # Are we checking the directory to see whether detections were actually false
                # positives, or reading from a list?
                if file_list is None:

                    # Is the image still there?
                    image_full_path = os.path.join(filtering_base_dir,
                                                   detection.sampleImageRelativeFileName)

                    # If not, remove this from the list of suspicious detections
                    if not os.path.isfile(image_full_path):
                        n_detections_removed += 1
                        b_valid_detection[i_detection] = False

                else:

                    if detection.sampleImageRelativeFileName not in file_list:
                        n_detections_removed += 1
                        b_valid_detection[i_detection] = False

            # ...for each detection

            n_removed_this_dir = len(b_valid_detection) - sum(b_valid_detection)
            if n_removed_this_dir > 0:
                print('Removed {} of {} detections from directory {}'.\
                      format(n_removed_this_dir,len(detections), i_dir))

            detections_filtered = list(compress(detections, b_valid_detection))
            suspicious_detections[i_dir] = detections_filtered

        # ...for each directory

        print('Removed {} of {} total detections via manual filtering'.\
              format(n_detections_removed, n_detections_loaded))

    # ...if we are/aren't finding detections (vs. loading from file)

    to_return.suspicious_detections = suspicious_detections

    to_return.allRowsFiltered = _update_detection_table(to_return, options, output_file_name)


    ##%% Create filtering directory

    if options.bWriteFilteringFolder:

        print('Creating filtering folder...')

        date_string = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
        filtering_dir = os.path.join(options.outputBase, 'filtering_' + date_string)
        os.makedirs(filtering_dir, exist_ok=True)

        # Take a first loop over every suspicious detection, and do the things that make
        # sense to do in a serial sampleImageDetectionsloop:
        #
        # * Generate file names (which requires an index variable)
        # * Sort instances by confidence
        # * Look up detections for each sample image in the big table (so we don't have to pass the
        #   table to workers)
        for i_dir, suspicious_detections_this_dir in enumerate(tqdm(suspicious_detections)):

            for i_detection, detection in enumerate(suspicious_detections_this_dir):

                # Sort instances in descending order by confidence
                detection.instances.sort(key=attrgetter('confidence'),reverse=True)

                if detection.clusterLabel is not None:
                    cluster_string = '_c{:0>4d}'.format(detection.clusterLabel)
                else:
                    cluster_string = ''

                # Choose the highest-confidence index
                instance = detection.instances[0]
                relative_path = instance.filename

                output_relative_path = 'dir{:0>4d}_det{:0>4d}{}_n{:0>4d}.jpg'.format(
                    i_dir, i_detection, cluster_string, len(detection.instances))
                detection.sampleImageRelativeFileName = output_relative_path

                i_row = filename_to_row[relative_path]
                row = detection_results.iloc[i_row]
                detection.sampleImageDetections = row['detections']

            # ...for each suspicious detection in this folder

        # ...for each folder

        # Collapse suspicious detections into a flat list
        all_suspicious_detections = []

        # i_dir = 0; suspicious_detections_this_dir = suspicious_detections[i_dir]
        for i_dir, suspicious_detections_this_dir in enumerate(tqdm(suspicious_detections)):
            for i_detection, detection in enumerate(suspicious_detections_this_dir):
                all_suspicious_detections.append(detection)

        # Render suspicious detections
        if options.bParallelizeRendering:

            n_workers = options.nWorkers

            pool = None

            try:
                if options.parallelizationUsesThreads:
                    pool = ThreadPool(n_workers); poolstring = 'threads'
                else:
                    pool = Pool(n_workers); poolstring = 'processes'

                print('Starting rendering pool with {} {}'.format(n_workers,poolstring))

            # We get slightly nicer progress bar behavior using threads, by passing a pbar
            # object and letting it get updated.  We can't serialize this object across
                # processes.
                if options.parallelizationUsesThreads:
                    options.pbar = tqdm(total=len(all_suspicious_detections))
                    all_candidate_detections = list(pool.imap(
                        partial(_render_sample_image_for_detection,filtering_dir=filtering_dir,
                                options=options), all_suspicious_detections))
                else:
                    options.pbar = None
                    all_candidate_detections = list(tqdm(pool.imap(
                        partial(_render_sample_image_for_detection,filtering_dir=filtering_dir,
                                options=options), all_suspicious_detections)))
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
                    print("Pool closed and joined for RDE rendering")

        else:

            # Serial loop over detections
            for detection in all_suspicious_detections:
                _render_sample_image_for_detection(detection,filtering_dir,options)

        # Delete (large) temporary data from the list of suspicious detections
        for detection in all_suspicious_detections:
            detection.sampleImageDetections = None

        # Write out the detection index
        detection_index_file_name = os.path.join(filtering_dir, detection_index_file_name_base)

        # Prepare the data we're going to write to the detection index file
        detection_info = {}

        detection_info['suspicious_detections'] = suspicious_detections
        detection_info['dir_index_to_name'] = dir_index_to_name

        # Remove the one non-serializable object from the options struct before serializing
        # to .json
        options.pbar = None
        detection_info['options'] = options

        s = jsonpickle.encode(detection_info,make_refs=False)
        with open(detection_index_file_name, 'w') as f:
            f.write(s)
        to_return.filterFile = detection_index_file_name

    # ...if we're writing filtering info

    return to_return

# ...def find_repeat_detections()
