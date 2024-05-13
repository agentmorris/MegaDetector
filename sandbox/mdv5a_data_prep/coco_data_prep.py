import os
import shutil
import json
from pycocotools.coco import COCO

coco=COCO('coco/annotations/instances_train2017.json')

# Filter out people and vehicles categories one at a time
categories = ['person', 'bicycle', 'car', 'airplane', 'motorcycle', 'bus', 'train', 'boat', 'truck']
filtered_annotations = {'images': [], 'annotations': [], 'categories': []}

# Create a directory for filtered images
if not os.path.exists('coco_filtered_images'):
    os.makedirs('coco_filtered_images')

for category in categories:
    catId = coco.getCatIds(catNms=[category])
    imgIds = coco.getImgIds(catIds=catId)

    # Get the images and annotations for the selected category
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catId, iscrowd=None)
        anns = coco.loadAnns(annIds)

        filtered_annotations['images'].append(img)
        filtered_annotations['annotations'].extend(anns)

        shutil.copy('coco/images/train2017/' + img['file_name'], 'coco_filtered_images/' + img['file_name'], follow_symlinks=True)

    cat = coco.loadCats(catId)
    filtered_annotations['categories'].extend(cat)

with open('filtered_annotations.json', 'w') as f:
    json.dump(filtered_annotations, f)
