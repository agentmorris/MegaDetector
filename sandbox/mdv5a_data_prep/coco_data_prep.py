import os
import shutil
import json
from pycocotools.coco import COCO
import argparse

arg_parser = argparse.ArgumentParser(description='Process COCO dataset')

arg_parser.add_argument('datasetYear',
                       metavar='dataset_year',
                       type=str,
                       help='Pass downloaded data directory e.g. train2017')

args = arg_parser.parse_args()

coco=COCO('coco/annotations/instances_' + args.datasetYear + '.json')

# Filter out people and vehicles categories one at a time
categories = ['person', 'bicycle', 'car', 'airplane', 'motorcycle', 'bus', 'train', 'boat', 'truck']
filtered_annotations = {'images': [], 'annotations': [], 'categories': []}
yolov5_annotations = []

if not os.path.exists('coco_filtered_images'):
    os.makedirs('coco_filtered_images')

for category in categories:
    catId = coco.getCatIds(catNms=[category])
    imgIds = coco.getImgIds(catIds=catId)

    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catId, iscrowd=None)
        anns = coco.loadAnns(annIds)

        filtered_annotations['images'].append(img)
        filtered_annotations['annotations'].extend(anns)

        # Convert COCO format to YOLOv5 format
        for ann in anns:
            bbox = ann['bbox']
            yolov5_annotation = [categories.index(category), bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]
            yolov5_annotations.append(yolov5_annotation)

        shutil.copy('coco/images/' + args.datasetYear + '/' + img['file_name'], 'coco_filtered_images/' + img['file_name'], follow_symlinks=True)

    cat = coco.loadCats(catId)
    filtered_annotations['categories'].extend(cat)

with open('coco_filtered_annotations.json', 'w') as f:
    json.dump(filtered_annotations, f)

with open('yolov5_annotations.txt', 'w') as f:
    for annotation in yolov5_annotations:
        f.write(' '.join(map(str, annotation)) + '\n')
