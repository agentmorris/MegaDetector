# Overview

This folder contains scripts and configuration files for training and evaluating [MegaDetector](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md).  If you are looking to <b>use</b> MegaDetector, you probably don't want to start with this page; instead, start with the [MegaDetector README](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md).  If you are looking to fine-tune MegaDetector on new data, you also don't want to start with this page; instead, start with the [YOLOv5 training guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

# Contents of this folder

* [megadetector_colab.ipynb](megadetector_colab.ipynb) is a Colab notebook that demonstrates the use of MegaDetector
* [process_video.py](process_video.py) is a (still pretty beta) script for running a video or folder of videos through MegaDetector.  YMMV; if you are working with video, you may want to [email us](mailto:cameratraps@lila.science).
* [pytorch_detector.py](pytorch_detector.py) is a wrapper for YOLOv5; users generally won't have to interact with this script directly.
* [run_detector.py](run_detector.py) is a simple script used to run MegaDetector on a small number of images, it's mostly used for environment validation.
* [run_detector_batch.py](run_detector_batch.py) is probably the most important tool in this repo; this is how one runs MegaDetector on lots of images.  See the "[using MegaDetector](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#using-the-model)" section for more information.
* [run_inference_with_yolov5_val.py](run_inference_with_yolov5_val.py) is a slightly-beta replacement for run_detector_batch.py that uses the native YOLOv5 tools to run MDv5 on lots of images.  The main reason to do this is that it allows us to benefit from YOLOv5's [test-time augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) tools.  If you have a dataset where MegaDetector works pretty well, and if you could just squeeze a *tiny* bit more recall out of it, you would be good to go, this may be useful to you.  If you're thinking of trying this out, you may want to [email us](mailto:cameratraps@lila.science).
* [tf_detector.py](tf_detector.py) is a wrapper for the TensorFlow Object Detection API; users generally won't have to interact with this script directly.
* [video_utils.py](video_utils.py) contains utilities for working with videos (extracting frames, rendering boxes on frames and assembling those into videos, etc.).
* [detector_training](detector_training) contains configuration files used at the time MD was trained.


# Format notes

Most users don't need to know anything about any of these formats; if you run MegaDetector and load the results into, e.g., Timelapse, all of this happens under the hood.  But if you are writing code to work with MegaDetector or any of the file formats described in this repo, some handy tips that might avoid confusion later:

* The [MegaDetector results format](../api/batch_processing#megadetector-batch-output-format) (the one all of our tools eventually produce, and the one downstream applications know how to read) uses normalized coordinates, as `[xmin, ymin, width_of_box, height_of_box]`
* Bounding boxes predicted by the underlying MegaDetector model are in normalized coordinates, as `[ymin, xmin, ymax, xmax]`
* The [COCO Camera Traps format](../data_management#coco-camera-traps-format) uses absolute coordinates, as `[xmin, ymin, width_of_box, height_of_box]`

In all cases, the origin is in the upper-left of the image.


# Training MegaDetector

## Assemble the training data set

These steps document the steps taken to assemble the training data set when MDv5 was trained at Microsoft; this section is not meaningful outside of Microsoft.

### Query MegaDB for the images of interest

Use `data_management/megadb/query_script.py` to query for all desired image entries. Write the query to use or select one from the examples at the top of the script. You may need to adjust the code parsing the query's output if you are using a new query. Fill in the output directory and other parameters also near the top of the script. 

To get labels for training  MegaDetector, use the query `query_bbox`. Note that to include images that were sent for annotation and were confirmed to be empty, make sure to specify `ARRAY_LENGTH(im.bbox) >= 0` to include the ones whose `bbox` field is an empty list. 

Running this query will take about 10+ minutes; this is a relatively small query so no need to increase the throughput of the database. The output is a JSON file containing a list, where each entry is the label for an image:
 
```json
{
 "bbox": [
  {
   "category": "person",
   "bbox": [
    0.3023,
    0.487,
    0.5894,
    0.4792
   ]
  }
 ],
 "file": "Day/1/IMAG0773 (4).JPG",
 "dataset": "dataset_name",
 "location": "location_designation"
}
```

### Assign each image a `download_id`

To avoid creating nested directories for downloaded images, we give each image a `download_id` to use as the file name to save the image at.

In any script or notebook, give each entry a unique ID (`<dataset>.seq<seq_id>.frame<frame_num>`).

If you are preparing data to add to an existing, already downloaded collection, add a field `new_entry` to the entry.

Save this version of the JSON list:

```json
{
 "bbox": [
  {
   "category": "person",
   "bbox": [
    0.3023,
    0.487,
    0.5894,
    0.4792
   ]
  }
 ],
 "file": "Day/1/IMAG0773 (4).JPG",
 "dataset": "dataset_name",
 "location": "location_designation",
 "download_id": "tnc_islands.seqab350628-ff22-2a29-8efa-boa24db24b57.frame0",
 "new_entry": true
}
```

### Download the images

Use `data_management/megadb/download_images.py` to download the new images, probably to an attached disk. Use the flag `--only_new_images` if in the above step you added the `new_entry` field to images that still need to be downloaded. 


### Split the images into train/val/test set

Use `data_management/megadb/split_images.py` to move the images to new folders `train`, `val`, and `test`. It will look up the splits in the Splits table in MegaDB, and any entries that do not have a location field will be placed in the training set.


## Train with YOLOv5

This section documents the environment in which MegaDetector v5 was trained; for more information about these parameters, see the [YOLOv5 training guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

With image size 1280px, starting with pre-trained weights (automatically downloaded from latest release) of the largest model (yolov5x6.pt). Saving checkpoint every epoch. Example:

```
export WANDB_CACHE_DIR=/camtraps/wandb_cache

docker pull nvidia/cuda:11.4.2-runtime-ubuntu20.04

(or yasiyu.azurecr.io/yolov5_training with the YOLOv5 repo dependencies installed)

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -d -it -v /marmot_disk_0/camtraps:/camtraps nvcr.io/nvidia/pytorch:21.10-py3 /bin/bash 

torchrun --standalone --nnodes=1 --nproc_per_node 2 train.py --project megadetectorv5 --name camonly_mosaic_xlarge_dist_5 --noval --save-period 1 --device 0,1 --batch 8 --imgsz 1280 --epochs 10 --weights yolov5x6.pt --data /home/ilipika/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5_yolo/data_camtrap_images_only.yml --hyp /home/ilipika/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5_yolo/hyp_mosaic.yml
```
