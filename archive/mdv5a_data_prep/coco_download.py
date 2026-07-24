"""

coco_data_prep.py

Downloads and unzips COCO 2017 data, see coco_data_prep.py for subsequent filtering.

Preserves COCO file structure; within [download_folder], you'll end up with:
    
* images
  * train2017
    * 000000000009.jpg
  * val2017
    * 000000000139.jpg
* annotations
  * instances_train2017.json
  * instances_val2017.json

"""

#%% Imports and constants

import os
import zipfile

from megadetector.utils.url_utils import download_url

download_folder = 'g:/temp/coco'
delete_zipfiles = True
overwrite_zipfiles = False

image_urls = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip',
    # Don't download test data
    # 'http://images.cocodataset.org/zips/test2017.zip'
]

annotation_urls = [
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
]


#%% Folder prep

image_folder = os.path.join(download_folder,'images')
os.makedirs(image_folder,exist_ok=True)


#%% Download

# url = image_urls[0]
for url in image_urls:
    
    filename = url.split('/')[-1]
    filepath = os.path.join(image_folder, filename)
    download_url(url=url,destination_filename=filepath,progress_updater=True,
                 force_download=overwrite_zipfiles,verbose=True)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(filepath))
        
    if delete_zipfiles:
        os.remove(filepath)

# url = annotation_urls[0]
for url in annotation_urls:
    
    filename = url.split('/')[-1]
    filepath = os.path.join(download_folder, filename)
    download_url(url=url,destination_filename=filepath,progress_updater=True,
                 force_download=overwrite_zipfiles)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(download_folder)
        
    if delete_zipfiles:
        os.remove(filepath)
