"""

coco_to_yolo.py

Converts a COCO-formatted dataset to a YOLO-formatted dataset, flattening
the dataset (to a single folder) in the process.

If the input and output folders are the same, writes .txt files to the input folder,
and neither moves nor modifies images.

Currently ignores segmentation masks, and errors if an annotation has a 
segmentation polygon but no bbox.

Has only been tested on a handful of COCO Camera Traps data sets; if you
use it for more general COCO conversion, YMMV.

"""

#%% Imports and constants

import json
import os
import shutil

from collections import defaultdict
from tqdm import tqdm

from megadetector.utils.path_utils import safe_create_link,find_images


#%% Support functions

def write_yolo_dataset_file(yolo_dataset_file,
                            dataset_base_dir,
                            class_list,
                            train_folder_relative=None,
                            val_folder_relative=None,
                            test_folder_relative=None):
    """
    Write a YOLOv5 dataset.yaml file to the absolute path [yolo_dataset_file] (should
    have a .yaml extension, though it's only a warning if it doesn't).  
    
    Args:
        yolo_dataset_file (str): the file, typically ending in .yaml or .yml, to write.  
            Does not have to be within dataset_base_dir.
        dataset_base_dir (str): the absolute base path of the YOLO dataset
        class_list (list or str): an ordered list of class names (the first item will be class 0, 
            etc.), or the name of a text file containing an ordered list of class names (one per 
            line, starting from class zero).
    """
    
    # Read class names
    if isinstance(class_list,str):
        with open(class_list,'r') as f:
            class_lines = f.readlines()
        class_lines = [s.strip() for s in class_lines]    
        class_list = [s for s in class_lines if len(s) > 0]

    if not (yolo_dataset_file.endswith('.yml') or yolo_dataset_file.endswith('.yaml')):
        print('Warning: writing dataset file to a non-yml/yaml extension:\n{}'.format(
            yolo_dataset_file))
        
    # Write dataset.yaml
    with open(yolo_dataset_file,'w') as f:
        
        f.write('# Train/val sets\n')
        f.write('path: {}\n'.format(dataset_base_dir))
        if train_folder_relative is not None:
            f.write('train: {}\n'.format(train_folder_relative))
        if val_folder_relative is not None:
            f.write('val: {}\n'.format(val_folder_relative))
        if test_folder_relative is not None:
            f.write('val: {}\n'.format(test_folder_relative))
            
        f.write('\n')
        
        f.write('# Classes\n')
        f.write('names:\n')
        for i_class,class_name in enumerate(class_list):
            f.write('  {}: {}\n'.format(i_class,class_name))

# ...def write_yolo_dataset_file(...)

            
def coco_to_yolo(input_image_folder,
                 output_folder,
                 input_file,
                 source_format='coco',
                 overwrite_images=False,
                 create_image_and_label_folders=False,
                 class_file_name='classes.txt',
                 allow_empty_annotations=False,
                 clip_boxes=False,
                 image_id_to_output_image_json_file=None,
                 images_to_exclude=None,
                 path_replacement_char='#',
                 category_names_to_exclude=None,
                 category_names_to_include=None,
                 write_output=True,
                 flatten_paths=True):
    """
    Converts a COCO-formatted dataset to a YOLO-formatted dataset, optionally flattening the 
    dataset to a single folder in the process.
    
    If the input and output folders are the same, writes .txt files to the input folder,
    and neither moves nor modifies images.
    
    Currently ignores segmentation masks, and errors if an annotation has a 
    segmentation polygon but no bbox.
    
    Args:
        input_image_folder (str): the folder where images live; filenames in the COCO .json
            file [input_file] should be relative to this folder
        output_folder (str): the base folder for the YOLO dataset
        input_file (str): a .json file in COCO format; can be the same as [input_image_folder], in which case
            images are left alone.
        source_format (str, optional): can be 'coco' (default) or 'coco_camera_traps'.  The only difference
            is that when source_format is 'coco_camera_traps', we treat an image with a non-bbox
            annotation with a category id of 0 as a special case, i.e. that's how an empty image
            is indicated.  The original COCO standard is a little ambiguous on this issue.  If
            source_format is 'coco', we either treat images as empty or error, depending on the value
            of [allow_empty_annotations].  [allow_empty_annotations] has no effect if source_format is
            'coco_camera_traps'.
        create_image_and_label_folder (bool, optional): whether to create separate folders called 'images' and
            'labels' in the YOLO output folder.  If create_image_and_label_folders is False, 
            a/b/c/image001.jpg will become a#b#c#image001.jpg, and the corresponding text file will 
            be a#b#c#image001.txt.  If create_image_and_label_folders is True, a/b/c/image001.jpg will become 
            images/a#b#c#image001.jpg, and the corresponding text file will be 
            labels/a#b#c#image001.txt.    
        clip_boxes (bool, optional): whether to clip bounding box coordinates to the range [0,1] before
            converting to YOLO xywh format
        image_id_to_output_image_json_file (str, optional): an optional *output* file, to which we will write
            a mapping from image IDs to output file names
        images_to_exclude (list, optional): a list of image files (relative paths in the input folder) that we 
            should ignore
        path_replacement_char (str, optional): only relevant if [flatten_paths] is True; this is used to replace
            path separators, e.g. if [path_replacement_char] is '#' and [flatten_paths] is True, a/b/c/d.jpg
            becomes a#b#c#d.jpg
        category_names_to_exclude (str, optional): category names that should not be represented in the
            YOLO output; only impacts annotations, does not prevent copying images.  There's almost no reason
            you would want to specify this and [category_names_to_include]. 
        category_names_to_include (str, optional): allow-list of category names that should be represented in the
            YOLO output; only impacts annotations, does not prevent copying images.  There's almost no reason
            you would want to specify this and [category_names_to_exclude]. 
        write_output (bool, optional): determines whether we actually copy images and write annotations;
            setting this to False mostly puts this function in "dry run" "mode.  The class list
            file is written regardless of the value of write_output.
    
    Returns:
        dict: information about the coco --> yolo mapping, containing at least the fields:
        
        - class_list_filename: the filename to which we wrote the flat list of class names required 
          by the YOLO format.
        - source_image_to_dest_image: a dict mapping source images to destination images
        - coco_id_to_yolo_id: a dict mapping COCO category IDs to YOLO category IDs        
    """
        
    ## Validate input
    
    if category_names_to_include is not None and category_names_to_exclude is not None:
        raise ValueError('category_names_to_include and category_names_to_exclude are mutually exclusive')
        
    if output_folder is None:
        output_folder = input_image_folder
    
    if images_to_exclude is not None:
        images_to_exclude = set(images_to_exclude)
            
    if category_names_to_exclude is None:
        category_names_to_exclude = {}
            
    assert os.path.isdir(input_image_folder)
    assert os.path.isfile(input_file)
    os.makedirs(output_folder,exist_ok=True)
    
    if (output_folder == input_image_folder) and (overwrite_images) and \
        (not create_image_and_label_folders) and (not flatten_paths):
            print('Warning: output folder and input folder are the same, disabling overwrite_images')
            overwrite_images = False
            
    ## Read input data
    
    with open(input_file,'r') as f:
        data = json.load(f)
        
        
    ## Parse annotations
  
    image_id_to_annotations = defaultdict(list)
    
    # i_ann = 0; ann = data['annotations'][0]
    for i_ann,ann in enumerate(data['annotations']):
        
        # Make sure no annotations have *only* segmentation data 
        if ( \
            ('segmentation' in ann.keys()) and \
            (ann['segmentation'] is not None) and \
            (len(ann['segmentation']) > 0) ) \
            and \
            (('bbox' not in ann.keys()) or (ann['bbox'] is None) or (len(ann['bbox'])==0)):
                raise ValueError('Oops: segmentation data present without bbox information, ' + \
                                 'this script isn\'t ready for this dataset')
        
        image_id_to_annotations[ann['image_id']].append(ann)
        
    print('Parsed annotations for {} images'.format(len(image_id_to_annotations)))
        
    # Re-map class IDs to make sure they run from 0...n-classes-1
    #
    # Note: this allows unused categories in the output data set.  This is OK for
    # some training pipelines, not for others.
    next_category_id = 0
    coco_id_to_yolo_id = {}    
    coco_id_to_name = {}
    yolo_id_to_name = {}
    coco_category_ids_to_exclude = set()
    
    for category in data['categories']:
        coco_id_to_name[category['id']] = category['name']
        if (category_names_to_include is not None) and \
            (category['name'] not in category_names_to_include):
            coco_category_ids_to_exclude.add(category['id'])
            continue
        elif (category['name'] in category_names_to_exclude):
            coco_category_ids_to_exclude.add(category['id'])
            continue               
        assert category['id'] not in coco_id_to_yolo_id
        coco_id_to_yolo_id[category['id']] = next_category_id
        yolo_id_to_name[next_category_id] = category['name']
        next_category_id += 1
        
    
    ## Process images (everything but I/O)
    
    # List of dictionaries with keys 'source_image','dest_image','bboxes','dest_txt'
    images_to_copy = []
    
    missing_images = []
    excluded_images = []
    
    image_names = set()
    
    typical_image_extensions = set(['.jpg','.jpeg','.png','.gif','.tif','.bmp'])
    
    printed_empty_annotation_warning = False
    
    image_id_to_output_image_name = {}
    
    print('Processing annotations')
    
    n_clipped_boxes = 0
    n_total_boxes = 0
    
    # i_image = 0; im = data['images'][i_image]
    for i_image,im in tqdm(enumerate(data['images']),total=len(data['images'])):
                
        output_info = {}
        source_image = os.path.join(input_image_folder,im['file_name'])        
        output_info['source_image'] = source_image
        
        if images_to_exclude is not None and im['file_name'] in images_to_exclude:
            excluded_images.append(im['file_name'])
            continue
        
        tokens = os.path.splitext(im['file_name'])
        if tokens[1].lower() not in typical_image_extensions:
            print('Warning: unusual image file name {}'.format(im['file_name']))
                
        if flatten_paths:
            image_name = tokens[0].replace('\\','/').replace('/',path_replacement_char) + \
                '_' + str(i_image).zfill(6)            
        else:
            image_name = tokens[0]

        assert image_name not in image_names, 'Image name collision for {}'.format(image_name)
        image_names.add(image_name)
        
        assert im['id'] not in image_id_to_output_image_name
        image_id_to_output_image_name[im['id']] = image_name
        
        dest_image_relative = image_name + tokens[1]
        output_info['dest_image_relative'] = dest_image_relative
        dest_txt_relative = image_name + '.txt'
        output_info['dest_txt_relative'] = dest_txt_relative
        output_info['bboxes'] = []
        
        # assert os.path.isfile(source_image), 'Could not find image {}'.format(source_image)
        if not os.path.isfile(source_image):
            print('Warning: could not find image {}'.format(source_image))
            missing_images.append(im['file_name'])
            continue
        
        image_id = im['id']
        
        image_bboxes = []
            
        if image_id in image_id_to_annotations:
                        
            for ann in image_id_to_annotations[image_id]:
                
                # If this annotation has no bounding boxes...
                if 'bbox' not in ann or ann['bbox'] is None or len(ann['bbox']) == 0:
                    
                    if source_format == 'coco':
                    
                        if not allow_empty_annotations:
                            # This is not entirely clear from the COCO spec, but it seems to be consensus
                            # that if you want to specify an image with no objects, you don't include any
                            # annotations for that image.
                            raise ValueError('If an annotation exists, it should have content')
                        else:
                            continue
                    
                    elif source_format == 'coco_camera_traps':
                        
                        # We allow empty bbox lists in COCO camera traps; this is typically a negative
                        # example in a dataset that has bounding boxes, and 0 is typically the empty 
                        # category.
                        if ann['category_id'] != 0:
                            if not printed_empty_annotation_warning:
                                printed_empty_annotation_warning = True
                                print('Warning: non-bbox annotation found with category {}'.format(
                                    ann['category_id']))
                        continue
                    
                    else:
                        
                        raise ValueError('Unrecognized COCO variant: {}'.format(source_format))
                    
                # ...if this is an empty annotation
                
                coco_bbox = ann['bbox']
                
                # This category isn't in our category list.  This typically corresponds to whole sets
                # of images that were excluded from the YOLO set.
                if ann['category_id'] in coco_category_ids_to_exclude:
                    continue
                
                yolo_category_id = coco_id_to_yolo_id[ann['category_id']]
                
                # COCO: [x_min, y_min, width, height] in absolute coordinates
                # YOLO: [class, x_center, y_center, width, height] in normalized coordinates
                
                # Convert from COCO coordinates to YOLO coordinates
                img_w = im['width']
                img_h = im['height']
                                
                if source_format in ('coco','coco_camera_traps'):
                    
                    x_min_absolute = coco_bbox[0]
                    y_min_absolute = coco_bbox[1]
                    box_w_absolute = coco_bbox[2]
                    box_h_absolute = coco_bbox[3]
                    
                    x_center_absolute = (x_min_absolute + (x_min_absolute + box_w_absolute)) / 2
                    y_center_absolute = (y_min_absolute + (y_min_absolute + box_h_absolute)) / 2
                    
                    x_center_relative = x_center_absolute / img_w
                    y_center_relative = y_center_absolute / img_h
                    
                    box_w_relative = box_w_absolute / img_w
                    box_h_relative = box_h_absolute / img_h
                    
                else:
                    
                    raise ValueError('Unrecognized source format {}'.format(source_format))
                
                if clip_boxes:
                    
                    clipped_box = False
                    
                    box_right = x_center_relative + (box_w_relative / 2.0)                    
                    if box_right > 1.0:
                        clipped_box = True
                        overhang = box_right - 1.0
                        box_w_relative -= overhang
                        x_center_relative -= (overhang / 2.0)

                    box_bottom = y_center_relative + (box_h_relative / 2.0)                                        
                    if box_bottom > 1.0:
                        clipped_box = True
                        overhang = box_bottom - 1.0
                        box_h_relative -= overhang
                        y_center_relative -= (overhang / 2.0)
                    
                    box_left = x_center_relative - (box_w_relative / 2.0)
                    if box_left < 0.0:
                        clipped_box = True
                        overhang = abs(box_left)
                        box_w_relative -= overhang
                        x_center_relative += (overhang / 2.0)
                        
                    box_top = y_center_relative - (box_h_relative / 2.0)
                    if box_top < 0.0:
                        clipped_box = True
                        overhang = abs(box_top)
                        box_h_relative -= overhang
                        y_center_relative += (overhang / 2.0)
                        
                    if clipped_box:
                        n_clipped_boxes += 1
                
                yolo_box = [yolo_category_id,
                            x_center_relative, y_center_relative, 
                            box_w_relative, box_h_relative]
                
                image_bboxes.append(yolo_box)
                n_total_boxes += 1
                
            # ...for each annotation 
            
        # ...if this image has annotations
        
        output_info['bboxes'] = image_bboxes
        
        images_to_copy.append(output_info)        
    
    # ...for each image
        
    print('\nWriting {} boxes ({} clipped) for {} images'.format(n_total_boxes,
                                                               n_clipped_boxes,len(images_to_copy)))
    print('{} missing images (of {})'.format(len(missing_images),len(data['images'])))
    
    if images_to_exclude is not None:
        print('{} excluded images (of {})'.format(len(excluded_images),len(data['images'])))
            
    
    ## Write output
    
    print('Generating class list')
    
    class_list_filename = os.path.join(output_folder,class_file_name)
    with open(class_list_filename, 'w') as f:
        print('Writing class list to {}'.format(class_list_filename))
        for i_class in range(0,len(yolo_id_to_name)):
            # Category IDs should range from 0..N-1
            assert i_class in yolo_id_to_name
            f.write(yolo_id_to_name[i_class] + '\n')
    
    if image_id_to_output_image_json_file is not None:
        print('Writing image ID mapping to {}'.format(image_id_to_output_image_json_file))
        with open(image_id_to_output_image_json_file,'w') as f:
            json.dump(image_id_to_output_image_name,f,indent=1)
    

    if (output_folder == input_image_folder) and (not create_image_and_label_folders):
        print('Creating annotation files (not copying images, input and output folder are the same)')
    else:
        print('Copying images and creating annotation files')

    if create_image_and_label_folders:
        dest_image_folder = os.path.join(output_folder,'images')
        dest_txt_folder = os.path.join(output_folder,'labels')
    else:
        dest_image_folder = output_folder
        dest_txt_folder = output_folder
        
    source_image_to_dest_image = {}
    
    # TODO: parallelize this loop
    #
    # output_info = images_to_copy[0]
    for output_info in tqdm(images_to_copy):

        source_image = output_info['source_image']
        dest_image_relative = output_info['dest_image_relative']
        dest_txt_relative = output_info['dest_txt_relative']
        
        dest_image = os.path.join(dest_image_folder,dest_image_relative)
        dest_txt = os.path.join(dest_txt_folder,dest_txt_relative)
        
        source_image_to_dest_image[source_image] = dest_image
        
        if write_output:
            
            os.makedirs(os.path.dirname(dest_image),exist_ok=True)        
            os.makedirs(os.path.dirname(dest_txt),exist_ok=True)
            
            if not create_image_and_label_folders:
                assert os.path.dirname(dest_image) == os.path.dirname(dest_txt)
            
            if (not os.path.isfile(dest_image)) or (overwrite_images):
                shutil.copyfile(source_image,dest_image)
        
            bboxes = output_info['bboxes']        
            
            # Only write an annotation file if there are bounding boxes.  Images with 
            # no .txt files are treated as hard negatives, at least by YOLOv5:
            #
            # https://github.com/ultralytics/yolov5/issues/3218
            #
            # I think this is also true for images with empty .txt files, but 
            # I'm using the convention suggested on that issue, i.e. hard 
            # negatives are expressed as images without .txt files.
            if len(bboxes) > 0:
                
                with open(dest_txt,'w') as f:
                    
                    # bbox = bboxes[0]
                    for bbox in bboxes:
                        assert len(bbox) == 5
                        s = '{} {} {} {} {}'.format(bbox[0],bbox[1],bbox[2],bbox[3],bbox[4])
                        f.write(s + '\n')
                        
        # ...if we're actually writing output
        
    # ...for each image
    
    coco_to_yolo_info = {}
    coco_to_yolo_info['class_list_filename'] = class_list_filename
    coco_to_yolo_info['source_image_to_dest_image'] = source_image_to_dest_image
    coco_to_yolo_info['coco_id_to_yolo_id'] = coco_id_to_yolo_id
    
    return coco_to_yolo_info

# ...def coco_to_yolo(...)


def create_yolo_symlinks(source_folder,images_folder,labels_folder,
                         class_list_file=None,
                         class_list_output_name='object.data',
                         force_lowercase_image_extension=False):
    """
    Given a YOLO-formatted folder of images and .txt files, creates a folder
    of symlinks to all the images, and a folder of symlinks to all the labels. 
    Used to support preview/editing tools that assume images and labels are in separate 
    folders.
    
    :meta private:
    """    
    
    assert source_folder != images_folder and source_folder != labels_folder
    
    os.makedirs(images_folder,exist_ok=True)
    os.makedirs(labels_folder,exist_ok=True)
    
    image_files_relative = find_images(source_folder,recursive=True,return_relative_paths=True)
    
    # image_fn_relative = image_files_relative[0]=    
    for image_fn_relative in tqdm(image_files_relative):
        
        source_file_abs = os.path.join(source_folder,image_fn_relative)
        target_file_abs = os.path.join(images_folder,image_fn_relative)
        
        if force_lowercase_image_extension:
            tokens = os.path.splitext(target_file_abs)
            target_file_abs = tokens[0] + tokens[1].lower()
            
        os.makedirs(os.path.dirname(target_file_abs),exist_ok=True)
        safe_create_link(source_file_abs,target_file_abs)
        source_annotation_file_abs = os.path.splitext(source_file_abs)[0] + '.txt'
        if os.path.isfile(source_annotation_file_abs):
            target_annotation_file_abs = \
                os.path.splitext(os.path.join(labels_folder,image_fn_relative))[0] + '.txt'
            os.makedirs(os.path.dirname(target_annotation_file_abs),exist_ok=True)
            safe_create_link(source_annotation_file_abs,target_annotation_file_abs)

    # ...for each image  

    if class_list_file is not None:
        target_class_list_file = os.path.join(labels_folder,class_list_output_name)
        safe_create_link(class_list_file,target_class_list_file)

# ...def create_yolo_symlinks(...)


#%% Interactive driver

if False:
    
    pass

    #%% Options
    
    input_file = os.path.expanduser('~/data/md-test-coco.json')
    image_folder = os.path.expanduser('~/data/md-test')    
    output_folder = os.path.expanduser('~/data/md-test-yolo')    
    create_image_and_label_folders=False
    class_file_name='classes.txt'
    allow_empty_annotations=False
    clip_boxes=False
    image_id_to_output_image_json_file=None
    images_to_exclude=None
    path_replacement_char='#'
    category_names_to_exclude=None
                                  
                                               
    #%% Programmatic execution
    
    coco_to_yolo_results = coco_to_yolo(image_folder,output_folder,input_file,
                     source_format='coco',
                     overwrite_images=False,
                     create_image_and_label_folders=create_image_and_label_folders,
                     class_file_name=class_file_name,
                     allow_empty_annotations=allow_empty_annotations,
                     clip_boxes=clip_boxes)                     
    
    create_yolo_symlinks(source_folder=output_folder,
                         images_folder=output_folder + '/images',
                         labels_folder=output_folder + '/labels',
                         class_list_file=coco_to_yolo_results['class_list_filename'],
                         class_list_output_name='object.data',
                         force_lowercase_image_extension=True)


    #%% Prepare command-line example

    s = 'python coco_to_yolo.py {} {} {} --create_bounding_box_editor_symlinks'.format(
        image_folder,output_folder,input_file)
    print(s)
    import clipboard; clipboard.copy(s)    


#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(
        description='Convert COCO-formatted data to YOLO format, flattening the image structure')
    
    # input_image_folder,output_folder,input_file
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to input images')
    
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to flat, YOLO-formatted dataset')
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to COCO dataset file (.json)')
    
    parser.add_argument(
        '--create_bounding_box_editor_symlinks',
        action='store_true',
        help='Prepare symlinks so the whole folder appears to contain "images" and "labels" folderss')        
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    coco_to_yolo_results = coco_to_yolo(args.input_folder,args.output_folder,args.input_file)
    
    if args.create_bounding_box_editor_symlinks:
        create_yolo_symlinks(source_folder=args.output_folder,
                             images_folder=args.output_folder + '/images',
                             labels_folder=args.output_folder + '/labels',
                             class_list_file=coco_to_yolo_results['class_list_filename'],
                             class_list_output_name='object.data',
                             force_lowercase_image_extension=True)
    
if __name__ == '__main__':
    main()
