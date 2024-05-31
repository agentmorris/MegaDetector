"""

cct_json_utils.py

Utilities for working with COCO Camera Traps .json databases:

https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md#coco-cameratraps-format

"""

#%% Constants and imports

import json
import os

from tqdm import tqdm
from collections import defaultdict, OrderedDict


#%% Classes

class CameraTrapJsonUtils:
    """
    Miscellaneous utility functions for working with COCO Camera Traps databases
    """
    
    @staticmethod
    def annotations_to_string(annotations, cat_id_to_name):
        """
        Given a list of annotations and a mapping from class IDs to names, produces
        a comma-delimited string containing a list of class names, sorted alphabetically.
        
        Args:
            annotations (list): a list of annotation dicts
            cat_id_to_name (dict): a dict mapping category IDs to category names
            
        Returns:
            str: a comma-delimited list of class names
        """
        
        class_names = CameraTrapJsonUtils.annotations_to_class_names(annotations, cat_id_to_name)
        return ','.join(class_names)


    @staticmethod
    def annotations_to_class_names(annotations, cat_id_to_name):
        """
        Given a list of annotations and a mapping from class IDs to names, produces
        a list of class names, sorted alphabetically.
        
        Args:
            annotations (list): a list of annotation dicts
            cat_id_to_name (dict): a dict mapping category IDs to category names
            
        Returns:
            list: a list of class names present in [annotations]
        """
        
        # Collect all names
        class_names = [cat_id_to_name[ann['category_id']] for ann in annotations]
        # Make names unique and sort
        class_names = sorted(set(class_names))
        return class_names


    @staticmethod
    def order_db_keys(db):
        """
        Given a dict representing a JSON database in the COCO Camera Trap
        format, returns an OrderedDict with keys in the order of 'info',
        'categories', 'annotations' and 'images'. When this OrderedDict is
        serialized with json.dump(), the order of the keys are preserved.

        Args:
            db (dict): a JSON database in the COCO Camera Trap format

        Returns:
            dict: the same content as [db] but as an OrderedDict with keys ordered for 
                readability
        """
        
        ordered = OrderedDict([
            ('info', db['info']),
            ('categories', db['categories']),
            ('annotations', db['annotations']),
            ('images', db['images'])])
        return ordered


    @staticmethod
    def group_annotations_by_image_field(db_indexed, image_field='seq_id'):
        """
        Given an instance of IndexedJsonDb, group annotation entries by a field in the
        image entry.  Typically used to find all the annotations associated with a sequence.
        
        Args:
            db_indexed (IndexedJsonDb): an initialized IndexedJsonDb, typically loaded from a 
                COCO Camera Traps .json file
            image_field (str, optional): a field by which to group annotations (defaults
                to 'seq_id')
        
        Returns:
            dict: a dict mapping objects (typically strings, in fact typically sequence IDs) to
                lists of annotations
        """
        
        image_id_to_image_field = {}
        for image_id, image_entry in db_indexed.image_id_to_image.items():
            image_id_to_image_field[image_id] = image_entry[image_field]

        res = defaultdict(list)
        for annotations in db_indexed.image_id_to_annotations.values():
            for annotation_entry in annotations:
                field_value = image_id_to_image_field[annotation_entry['image_id']]
                res[field_value].append(annotation_entry)
        return res


    @staticmethod
    def get_entries_for_locations(db, locations):
        """
        Given a dict representing a JSON database in the COCO Camera Trap format, returns a dict
        with the 'images' and 'annotations' fields in the CCT format, each is an array that only
        includes entries in the original [db] that are in the [locations] set.
        
        Args:
            db (dict): a dict representing a JSON database in the COCO Camera Trap format
            locations (set): a set or list of locations to include; each item is a string

        Returns:
            dict: a dict with the 'images' and 'annotations' fields in the CCT format
        """
        
        locations = set(locations)
        print('Original DB has {} image and {} annotation entries.'.format(
            len(db['images']), len(db['annotations'])))
        new_db = { 'images': [], 'annotations': [] }
        new_images = set()
        for i in db['images']:
            # cast location to string as the entries in locations are strings
            if str(i['location']) in locations:
                new_db['images'].append(i)
                new_images.add(i['id'])
        for a in db['annotations']:
            if a['image_id'] in new_images:
                new_db['annotations'].append(a)
        print(
            'New DB has {} image and {} annotation entries.'.format(
                len(new_db['images']), len(new_db['annotations'])))
        return new_db


class IndexedJsonDb:
    """
    Wrapper for a COCO Camera Traps database.

    Handles boilerplate dictionary creation that we do almost every time we load
    a .json database.
    """

    def __init__(self, 
                 json_filename,
                 b_normalize_paths=False,
                 filename_replacements=None,
                 b_convert_classes_to_lower=True,
                 b_force_forward_slashes=True):
        """
        Constructor for IndexedJsonDb that loads from a .json file or CCT-formatted dict.
        
        Args:
            json_filename (str): filename to load, or an already-loaded dict
            b_normalize_paths (bool, optional): whether to invoke os.path.normpath on 
                all filenames.  Not relevant if b_force_forward_slashes is True.
            filename_replacements (dict, optional): a set of string --> string mappings
                that will trigger replacements in all filenames, typically used to remove
                leading folders
            b_convert_classes_to_lower (bool, optional): whether to convert all class
                names to lowercase
            b_force_forward_slashes (bool, optional): whether to convert backslashes to
                forward slashes in all path names            
        """
        
        if isinstance(json_filename, str):
            with open(json_filename) as f:
                self.db = json.load(f)
        else:
            self.db = json_filename

        assert 'images' in self.db, (
            f'Could not find image list in file {json_filename}, are you sure '
            'this is a COCO camera traps file?')
        
        if b_convert_classes_to_lower:
            # Convert classnames to lowercase to simplify comparisons later
            for c in self.db['categories']:
                c['name'] = c['name'].lower()

        # Normalize paths to simplify comparisons later
        if b_normalize_paths:
            for im in self.db['images']:
                im['file_name'] = os.path.normpath(im['file_name'])

        if b_force_forward_slashes:
            for im in self.db['images']:
                im['file_name'] = im['file_name'].replace('\\','/')

        if filename_replacements is not None:
            for s in filename_replacements:
                # Make custom replacements in filenames, typically used to
                # accommodate changes in root paths after DB construction
                r = filename_replacements[s]
                for im in self.db['images']:
                    im['file_name'] = im['file_name'].replace(s, r)

        ### Build useful mappings to facilitate working with the DB

        # Category ID <--> name
        self.cat_id_to_name = {
            cat['id']: cat['name'] for cat in self.db['categories']}
        self.cat_name_to_id = {
            cat['name']: cat['id'] for cat in self.db['categories']}

        # Image filename --> ID
        self.filename_to_id = {
            im['file_name']: im['id'] for im in self.db['images']}

        # Image ID --> image object
        self.image_id_to_image = {im['id']: im for im in self.db['images']}

        # Image ID --> annotations
        # Each image can potentially multiple annotations, hence using lists
        self.image_id_to_annotations = {}
        self.image_id_to_annotations = defaultdict(list)
        for ann in self.db['annotations']:
            self.image_id_to_annotations[ann['image_id']].append(ann)

    # ...__init__


    def get_annotations_for_image(self, image):
        """
        Finds all the annnotations associated with the image dict [image].
        
        Args:
            image (dict): an image dict loaded from a CCT .json file.  Only the 'id' field
                is used.
            
        Returns:
            list: list of annotations associated with this image.  Returns None if the db 
                has not been loaded, or [] if no annotations are available for this image.
        """
        
        if self.db is None:
            return None

        if image['id'] not in self.image_id_to_annotations:
            return []

        image_annotations = self.image_id_to_annotations[image['id']]
        return image_annotations


    def get_classes_for_image(self, image):
        """
        Returns a list of class names associated with [image].

        Args:
            image (dict): an image dict loaded from a CCT .json file.  Only the 'id' field
                is used.
            
        Returns:
            list: list of class names associated with this image.  Returns None if the db 
                has not been loaded, or [] if no annotations are available for this image.
        """
        
        if self.db is None:
            return None

        if image['id'] not in self.image_id_to_annotations:
            return []

        class_ids = []
        image_annotations = self.image_id_to_annotations[image['id']]
        for ann in image_annotations:
            class_ids.append(ann['category_id'])
        class_ids = sorted(set(class_ids))
        class_names = [self.cat_id_to_name[x] for x in class_ids]

        return class_names

# ...class IndexedJsonDb

class SequenceOptions:
    """
    Options parameterizing the grouping of images into sequences by time.
    """
    
    def __init__(self):
        #: Images separated by <= this duration will be grouped into the same sequence.
        self.episode_interval_seconds = 60.0

    
#%% Functions

def create_sequences(image_info,options=None):
    """
    Synthesizes episodes/sequences/bursts for the images in [image_info].
    
    Modifies [image_info] in place, populating the 'seq_id', 'seq_num_frames', and 'frame_num' 
    fields for each image.
    
    Args:
        image_info (str, dict, or list): a dict in CCT format, a CCT .json file, or just the 'images' component
            of a CCT dataset (a list of dicts with  fields 'file_name' (str), 'datetime' (datetime), and
            'location' (str)).
    """
    
    if options is None:
        options = SequenceOptions()
    
    if isinstance(image_info,str):
        with open(image_info,'r') as f:
            image_info = json.load(f)
        
    if isinstance(image_info,dict):
        image_info = image_info['images']
        
    # Find all unique locations
    locations = set()
    for im in image_info:
        locations.add(im['location'])
        
    print('Found {} locations'.format(len(locations)))    
    locations = list(locations)
    locations.sort()
    
    all_sequences = set()
    
    # i_location = 0; location = locations[i_location]
    for i_location,location in tqdm(enumerate(locations),total=len(locations)):
        
        images_this_location = [im for im in image_info if im['location'] == location]    
        
        # Sorting datetimes fails when there are None's in the list.  So instead of sorting datetimes 
        # directly, sort tuples with a boolean for none-ness, then the datetime itself.
        #
        # https://stackoverflow.com/questions/18411560/sort-list-while-pushing-none-values-to-the-end
        sorted_images_this_location = sorted(images_this_location, 
                                             key = lambda im: (im['datetime'] is None,im['datetime']))
        
        sequence_id_to_images_this_location = defaultdict(list)

        current_sequence_id = None
        next_frame_number = 0
        next_sequence_number = 0
        previous_datetime = None
            
        # previous_datetime = sorted_images_this_location[0]['datetime']
        # im = sorted_images_this_location[1]
        for im in sorted_images_this_location:
            
            invalid_datetime = False
            
            if previous_datetime is None:
                delta = None
            elif im['datetime'] is None:
                invalid_datetime = True
            else:
                delta = (im['datetime'] - previous_datetime).total_seconds()
            
            # Start a new sequence if necessary, including the case where this datetime is invalid
            if delta is None or delta > options.episode_interval_seconds or invalid_datetime:
                next_frame_number = 0
                current_sequence_id = 'location_{}_sequence_index_{}'.format(
                    location,str(next_sequence_number).zfill(5))
                next_sequence_number = next_sequence_number + 1
                assert current_sequence_id not in all_sequences
                all_sequences.add(current_sequence_id)                
                
            im['seq_id'] = current_sequence_id
            im['seq_num_frames'] = None
            im['frame_num'] = next_frame_number
            sequence_id_to_images_this_location[current_sequence_id].append(im)
            next_frame_number = next_frame_number + 1
            
            # If this was an invalid datetime, this will record the previous datetime
            # as None, which will force the next image to start a new sequence.
            previous_datetime = im['datetime']
        
        # ...for each image in this location
    
        # Fill in seq_num_frames
        for seq_id in sequence_id_to_images_this_location.keys():
            assert seq_id in sequence_id_to_images_this_location
            images_this_sequence = sequence_id_to_images_this_location[seq_id]
            assert len(images_this_sequence) > 0
            for im in images_this_sequence:
                im['seq_num_frames'] = len(images_this_sequence)
                
    # ...for each location
    
    print('Created {} sequences from {} images'.format(len(all_sequences),len(image_info)))
    
# ...create_sequences()
