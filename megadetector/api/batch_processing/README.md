# MegaDetector batch processing format

Tools or scripts that run MegaDetector (particularly [run_detector_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector_batch.py)), or tools that want to produce MegaDetector-compatible output, produce results in the format described on this page.  This folder contains modules for working with files in this format.


## Post-processing tools

The [postprocessing](postprocessing) folder contains tools for working with MegaDetector output.  In particular, [postprocess_batch_results.py](postprocessing/postprocess_batch_results.py) provides visualization and accuracy assessment tools; a sample output report is available [here](https://lila.science/public/snapshot_safari_public/snapshot-safari-kar-2022-00-00-v5a.0.0_0.200/index.html) for the case where ground truth is not available (the typical case).


## Integration with other tools

The [integration](integration) folder contains guidelines and postprocessing scripts for using the output of our API in other applications, particularly [Timelapse](https://saul.cpsc.ucalgary.ca/timelapse/).


## MegaDetector batch output format

Permanent link to this section: <https://lila.science/megadetector-output-format>

A validator for this format is available [here](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/validate_batch_results.py).

Example output with both detection and classification results:

```jsonc
{
    "info": {

        // Required
        "format_version": "1.4",
        
        // All other "info" fields are optional
        "detector": "md_v4.1.0.pb",
        "detection_completion_time": "2019-05-22 02:12:19",
        "classifier": "ecosystem1_v2",
        "classification_completion_time": "2019-05-26 01:52:08",
        "detector_metadata": {
           "megadetector_version":"v4.1.0",
           // These fields make recommendations to downstream tools about 
           // reasonable default confidence thresholds.
           "typical_detection_threshold":0.8,
           "conservative_detection_threshold":0.6
        },
        "classifier_metadata": {
           "typical_classification_threshold":0.75
        },
        "summary_report": "<p>Summary report</p><br/><p>...in which HTML is allowed.</p>"
    },
    // detection_categories is required; category IDs must be string-formatted ints.
    //
    // Category names can be arbitrary, but downstream tools may take a particular dependency 
    // on the name "animal", so using "animal" (rather than, e.g., "animals" or "wildlife")
    // is recommended.
    "detection_categories": {
        "1": "animal",
        "2": "person",
        "3": "vehicle"
    },
    // classification_categories is optional; if present, category IDs must be 
    // string-formatted ints.
    "classification_categories": {
        "0": "fox",
        "1": "elk",
        "2": "wolf",
        "3": "bear",
        "4": "moose"
    },
    // classification_category_descriptions is optional; if present, category IDs must be 
    // string-formatted ints.  This is typically used to provide searchable taxonomic names
    // for categories.
    "classification_category_descriptions": {
        "0": "animalia;chordata;mammalia;carnivora;canidae;vulpesvulpes",
        "1": "animalia;chordata;mammalia;artiodactyla;cervidae;cervuscanadensis",
        "2": "animalia;chordata;mammalia;carnivora;canidae;canislupus",
        "3": "animalia;chordata;mammalia;carnivora;ursidae;ursusamericanus",
        "4": "animalia;chordata;mammalia;artiodactyla;cervidae;alcesalces"     
    },
    // The "images" array is required, but can be empty.
    "images": [
        {
            "file": "path/from/base/dir/image_with_animal.jpg",
            // "detections" should be present for any image that was 
            // successfully processed.  I.e., a lack of detections
            // should be communicated with an empty "detections" array,
            // not by omitting the "detections" field.
            "detections": [
                {
                    "category": "1",
                    "conf": 0.926,
                    // See below for more details on bounding box format
                    "bbox": [0.0, 0.2762, 0.1539, 0.2825], 
                    "classifications": [
                        ["3", 0.901],
                        ["1", 0.071],
                        ["4", 0.025]
                    ]
                },
                {
                    "category": "1",
                    "conf": 0.061,
                    "bbox": [0.0451, 0.1849, 0.3642, 0.4636]
                }
            ]
        },
        // Videos appear in the same format as images, with the addition of the
        // "frame_rate" field (for the file) and the "frame_number" field (for each 
        // detection).  For videos, "frame_rate" and "frame_number" are required fields.
        //
        // frame_rate should be greater than zero, and can be int- or float-valued.  
        //
        // frame_number should be int-valued, and greater than or equal to zero.
        //
        // Detections are typically included for just one representative
        // frame for each detection category, but detections may also be reported for
        // multiple frames for a single detection category, as in this example.
        {
            "file": "path/from/base/dir/video_with_person.mp4",
            "frame_rate": 20,
            "detections": [
                {
                    "category": "2",
                    "conf": 0.871,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "frame_number": 0
                },
                {
                    "category": "2",
                    "conf": 0.774,
                    "bbox": [0.11, 0.21, 0.31, 0.41],
                    "frame_number": 20
                }                               
            ]
        },         
        {
            // This file was processed correctly, but had no detections
            "file": "/path/from/base/dir/empty_image.jpg",
            "detections": []
        },
        {
            "file": "/path/from/base/dir/another_image.jpg",
            "detections": [],
            // The "failure" field is optional; for successfully-processed images, it can be 
            // omitted  or can be null.  This file was processed correctly.
            "failure": null        
        },
        {
            // This file was not processed.  "failure" should be a string in this case, indicating 
            // the reason for failure.  "detections" can be null or omitted in this case.
            "file": "/path/from/base/dir2/corrupted_image_0.jpg",
            "failure": "Failure image access",
            "detections": null
        }
    ]
}
```

### Model metadata

The 'detector' field (within the 'info' field) specifies the filename of the detector model that produced this results file.  It was omitted in old files generated with run_detector_batch.py, so with extremely high probability, if this field is not present, you can assume the file was generated with MegaDetector v4.

In newer files, this should contain the filename (base name only) of the model file, which typically will be one of:

* megadetector_v4.1 (MegaDetector v4, run via the batch API) 
* md_v4.1.0.pb (MegaDetector v4, run locally) 
* md_v5a.0.0.pt (MegaDetector v5a) 
* md_v5b.0.0.pt (MegaDetector v5b) 

This string is used by some tools to choose appropriate default confidence values, which depend on the model version.  If you change the name of the MegaDetector file, you will break this convention, and YMMV.
 
The "detector_metadata" and "classifier_metadata" fields are also optionally added as of format version 1.2.  These currently contain useful default confidence values for downstream tools (particularly Timelapse), but we strongly recommend against blindly trusting these defaults; always explore your data before choosing a confidence threshold, as the optimal value can vary widely.


### Detector outputs

The bounding box in the `bbox` field is represented as

```
[x_min, y_min, width_of_box, height_of_box]
```

where `(x_min, y_min)` is the upper-left corner of the detection bounding box, with the origin in the upper-left corner of the image. The coordinates and box width and height are *relative* (i.e., normalized) to the width and height of the image. Note that this is different from the coordinate format used in the [COCO Camera Traps](data_management/README.md) databases, which are in absolute coordinates. 

The detection category `category` can be interpreted using the `detection_categories` dictionary. 

Detection categories not listed here are allowed by this format specification, but should be treated as "no detection".

When the detector model detects no animal (or person or vehicle), the confidence `conf` is shown as 0.0 (not confident that there is an object of interest) and the `detections` field is an empty list.


### Classifier outputs

After a classifier is applied, each tuple in a `classifications` list represents `[species, confidence]`. They must be listed in descending order by confidence. The species categories should be interpreted using the `classification_categories` dictionary.  Keys in `classification_categories` will always be non-negative integers formatted as strings.
