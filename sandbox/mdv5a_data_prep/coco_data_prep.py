"""

coco_data_prep.py

After downloading COCO data with coco_download.py, filter to categories relevant for 
MDv5a training.  Merges the train/val/test COCO data into a single COCO .json file.
To convert to YOLO format, use coco_to_yolo.py.

"""

#%% Imports and constants

import os
import json

from tqdm import tqdm

from pycocotools.coco import COCO

from megadetector.utils.path_utils import parallel_copy_files

# All image filenames in the COCO annotation files start with this URL
coco_base_url = 'http://images.cocodataset.org/'
coco_base_url_replacement = 'images/'

relative_annotation_files = [
    'annotations/instances_train2017.json',
    'annotations/instances_val2017.json'
]

categories_to_keep = ['person', 'bicycle', 'car', 'airplane', 'motorcycle', 'bus', 'train', 'boat', 'truck']


#%% Functions

def filter_coco_data(input_folder,output_folder):
    """
    Filter COCO train/val data to categories relevant for MDv5a training, merging into
    a single .json file, with relative filenames in the file_name field.
    
    Args:
        input_folder (str): the folder to which COCO train/val data was downloaded and
            unzipped
        output_folder (str): the folder to which we should copy the filtered subeset of
            images and write the filtered .json file
            
    Returns:
        str: the absolute path of the filtered .json file
    """
    
    ##%% Prep
    
    os.makedirs(output_folder,exist_ok=True)
    
    category_id_to_category = {}
    
    
    ##%% Filter images and annotations
    
    image_id_to_image = {}
    annotation_id_to_annotation = {}
    
    # Number of images before filtering
    n_images_total = 0
    n_annotations_total = 0
    
    
    # annotation_file_relative = relative_annotation_files[0]
    for annotation_file_relative in relative_annotation_files:
    
        print('Processing annotation file {}'.format(annotation_file_relative))
        
        annotation_file_abs = os.path.join(input_folder,annotation_file_relative)
        assert os.path.isfile(annotation_file_abs)
        
        coco=COCO(annotation_file_abs)

        n_images_total += len(coco.getImgIds())
        n_annotations_total += len(coco.getAnnIds())
        
        # category_name = categories_to_keep[0]
        for category_name in tqdm(categories_to_keep):
            
            category_id = coco.getCatIds(catNms=[category_name])
            assert len(category_id) == 1
            category_id = category_id[0]
            
            img_ids = coco.getImgIds(catIds=category_id)
    
            # img_id = img_ids[0]
            for img_id in img_ids:
                
                img = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=category_id, iscrowd=None)
                anns = coco.loadAnns(ann_ids)
    
                # In the COCO .json file, 'file_name' is a somewhat-unhelpful base name, we want the
                # relative path.
                assert img['coco_url'].startswith(coco_base_url)
                image_fn_relative = img['coco_url'].replace(coco_base_url,coco_base_url_replacement)
                image_fn_abs_input = os.path.join(input_folder,image_fn_relative)
                assert os.path.isfile(image_fn_abs_input)
                img['file_name'] = image_fn_relative
                
                # Only add each image once
                if img['id'] in image_id_to_image:
                    assert image_id_to_image[img['id']]['file_name'] == img['file_name']
                else:
                    image_id_to_image[img['id']] = img
                
                # Each annotation should be unique
                for ann in anns:
                    assert ann['id'] not in annotation_id_to_annotation
                    annotation_id_to_annotation[ann['id']] = ann                    
                    
            cat = coco.loadCats(category_id)
            assert len(cat) == 1
            cat = cat[0]
            
            # Only add each category once
            if category_id in category_id_to_category:
                assert category_id_to_category[category_id]['name'] == cat['name']
            else:
                category_id_to_category[category_id] = cat            

        # ...for each category
    
    # ...for each annotation file (train, val)
    
    print('\nPreserved {} of {} images, {} of {} annotations'.format(
        len(image_id_to_image),n_images_total,
        len(annotation_id_to_annotation),n_annotations_total))
    
    
    ##%% Write filtered annotations
    
    filtered_annotations = {}
    filtered_annotations['images'] = list(image_id_to_image.values())
    filtered_annotations['annotations'] = list(annotation_id_to_annotation.values())
    filtered_annotations['info'] = {'categories_to_keep':categories_to_keep}
    filtered_annotations['categories'] = list(category_id_to_category.values())
    filtered_annotation_file_abs = os.path.join(output_folder,'coco_filtered_annotations.json')
    
    with open(filtered_annotation_file_abs, 'w') as f:
        json.dump(filtered_annotations, f, indent=1)


    ##%% Copy files to the output folder
    
    input_image_to_output_image = {}
    
    # img = filtered_annotations['images'][0]
    for img in filtered_annotations['images']:
        image_fn_relative = img['file_name']
        image_fn_abs_input = os.path.join(input_folder,image_fn_relative)
        assert os.path.isfile(image_fn_abs_input)
        assert image_fn_abs_input not in input_image_to_output_image
        image_fn_abs_output = os.path.join(output_folder,image_fn_relative)
        input_image_to_output_image[image_fn_abs_input] = image_fn_abs_output
        
    parallel_copy_files(input_image_to_output_image)
    
    
    ##%% Validate the output file
    
    from megadetector.data_management.databases.integrity_check_json_db import \
        integrity_check_json_db,IntegrityCheckOptions
    
    options = IntegrityCheckOptions()
    options.baseDir = output_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False
    options.allowIntIDs = True
    
    sortedCategories, data, errorInfo = integrity_check_json_db(filtered_annotation_file_abs,options) # noqa
    

    ##%%
    
    return filtered_annotation_file_abs

# ...filter_coco_data(...)


#%% Interactive driver

if False:
    
    pass

    #%%    
    
    input_folder = 'g:/temp/coco'; assert os.path.isdir(input_folder)
    output_folder = 'g:/temp/coco_filtered'
    filtered_annotation_file_abs = filter_coco_data(input_folder, output_folder)

    #%%
    
    preview_folder = 'g:/temp/coco_filtered_preview'
    
    from megadetector.visualization.visualize_db import DbVizOptions,visualize_db
    
    options = DbVizOptions()
    options.num_to_visualize = 1000
    options.viz_size = None
    options.parallelize_rendering = True
    
    html_preview_file,_ = visualize_db(db_path=filtered_annotation_file_abs, 
                 output_dir=preview_folder,
                 image_base_dir=output_folder,
                 options=options)
    
    from megadetector.utils.path_utils import open_file
    
    open_file(html_preview_file)
    
    
#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(description='Process COCO dataset')
    parser.add_argument('input_folder', type=str, help='Folder to which COCO data was downloaded')
    parser.add_argument('output_folder', type=str, help='Folder to write filtered COCO image set and annotations')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    
    filter_coco_data(args.input_folder,args.output_folder)

if __name__ == '__main__':
    main()
