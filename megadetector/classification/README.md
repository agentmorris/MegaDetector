# Table of Contents

[Overview](#overview)  
[Training classifiers](#training-classifiers-that-play-nicely-with-megadetector)  
[Running MegaClassifier](#running-megaclassifier)  
&nbsp;&nbsp;&nbsp;&nbsp;[MegaClassifier overview](#megaclassifier-overview-and-caveats)  
&nbsp;&nbsp;&nbsp;&nbsp;[MegaClassifier instructions](#megaclassifier-instructions)  
&nbsp;&nbsp;&nbsp;&nbsp;[MegaClassifier label specification syntax](#megaclassifier-label-specification-syntax)  

# Overview

This README originally documented the training and inference processes for species classifiers that play nicely with MegaDetector, particularly "MegaClassifier".  The training process used to create MegaClassifier is obsolete at this point, so this page is now about <i>running</i> MegaClassifier.  In fact, <i>running</i> MegaClassifier is now obsolete too; MegaClassifier is now strictly less useful than [SpeciesNet](http://github.com/google/cameratrapai), so, if you're here looking for MegaClassifier, you probably want to try SpeciesNet.

But regardless of MegaClassifier, SpeciesNet, etc., first, some tips about tools you might use to train your own classifiers.

## No, first, are you sure you want to train your own classifier?

My two cents...

I would only advise training a custom model as a last resort.  Remember that an AI classifier's ability to make you more efficient isn't necessarily related to whether it knows about all of your species or your geography: if a classifier from an unrelated geography makes _consistent_ predictions on your data, it doesn't really matter whether it knows about your species.  For example, if I am interested in coyotes in Canada, and I run a classifier that was trained in Kenya, and all of my coyotes are consistently classified as jackals, that's just as good as having a coyote classifier.

With that in mind, here are the things I would try before training your own classifier:

1. Consider trying [SpeciesNet](https://github.com/google/cameratrapai), a global classifier that works pretty well in a variety of ecosystems.  If your species are not covered, but you can do a suitable remapping of the outputs, that might be a very efficient solution.  (Full disclosure: I work on both MegaDetector and SpeciesNet.)
2. I keep a list of publicly-available species classifiers [here](https://agentmorris.github.io/camera-trap-ml-survey/#publicly-available-ml-models-for-camera-traps); if one appears to have species that are visually similar to yours, give that a try.  Remember that the "right" classifier for you doesn't need to be specific to your geography, it just needs to make consistent predictions on the species you care about.

But if you still want to either train a classifier, or try MegaClassifier, or just learn about classifier training, read on...

# Training classifiers that play nicely with MegaDetector

This repo is focused on MegaDetector, but we love talking about species classifiers too.  Feel free to <a href="mailto:cameratraps@lila.science">email us</a> if you want to chat classifiers, or - better yet - post questions to the <a href="https://aiforconservation.slack.com">AI for Conservation Slack</a>.

That said, some thoughts on training classifiers...

## Infrastructure for training classifiers

We recommend checking out the [Mega Efficient Wildlife Classifier](https://github.com/zaandahl/mewc) (MEWC) repo, which makes it easy to train and run species classifiers on MegaDetector crops.

## Other classifiers that exist in the universe

The "[publicly-available ML models for camera traps](https://agentmorris.github.io/camera-trap-ml-survey/#publicly-available-ml-models-for-camera-traps)" section of the [camera trap ML survey](https://agentmorris.github.io/camera-trap-ml-survey) lists all the classifiers I'm aware of that one can download and run.  Many of them operate alongside MegaDetector, though that list also includes whole-image classifiers.

## Data for training your own classifiers

There are &gt;10M [camera trap images available on LILA](https://lila.science/category/camera-traps/); for many classifiers, you may find sufficient data there, or you may be able to complement your own data with LILA data.  We don't recommend messing around with individual LILA datasets; instead, check out the [taxonomy mapping for camera trap datasets](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/) page, which treats all the camera trap data on LILA as one big dataset, represented in a single .csv file, with a consistent taxonomy.  This combined dataset has also been [imported into Hugging Face](https://huggingface.co/datasets/society-ethics/lila_camera_traps).  If you are training a classifier on MegaDetector crops from LILA data, you don't have to run MD yourself: [we have already run MegaDetector on all camera trap data on LILA](https://lila.science/megadetector-results-for-camera-trap-datasets/).

# Running MegaClassifier

## MegaClassifier overview (and caveats)

<i>MegaClassifier</i> is a species classification model that runs on MegaDetector crops, i.e. it assumes you have already run MegaDetector, so it only has to look at the images (and in fact only the pixels) that are actually animals.

First, let me give you a zillion reasons that MegaClassifier may not be what you want.  MegaClassifier is trained on data from a variety of ecosystems, but it was never intended as a broad/global classifier; the idea - which I still think was a good idea, even though we never really got to finish the vision! - is that you would always use this classifier with a mapping from raw output classes to classes you care about.  For example, you might map all the canids to a "canid" class, or if you only have one cervid species in your ecosystem, you might map all the cervids to a specific species, so a prediction of "European red deer" might get translated into "black-tailed deer" if you're working in California.

But we only ever really developed one such mapping file (for the Western US/Canada), so we've only really ever used it with this one mapping file, whose output categories are (bear, bird, canid, cat, cow, pronghorn, deer, elk, goat/sheep, moose, horse, rabbit/hare, small mammal, other).  That said, it's been pretty useful with that mapping file, particularly in cases where deer/elk/cattle are the most common classes, and/or when domestic dogs are very common (e.g. for urban surveys).  The way it's typically used is to quickly filter the common classes, so even if someone is doing a bobcat survey, we generally encourage them to ignore all classes other than deer/cattle/elk.  

Most MegaClassifier users work with MegaClassifier (and MegaDetector) output in [Timelapse](https://saul.cpsc.ucalgary.ca/timelapse/); the [Timelapse Image Recognition Guide](https://saul.cpsc.ucalgary.ca/timelapse/uploads/Guides/TimelapseImageRecognitionGuide.pdf) does a good job laying out an AI-accelerated image review workflow.

It's also been integrated into WildTrax ([release notes](https://abmi.ca/home/news-events/events/New-WildTrax-Release)) ([documentation](https://wildtrax.ca/resources/user-guide/#image-data)), whose users are often squarely in that "lots of deer/elk/cattle" category.

Also, although I'm just calling this section "MegaClassifier" to keep things simple, there are actually two classifiers I'll talk about in these instructions: "MegaClassifier" (which has a few hundred output classes from all over the world) and the "IDFG classifier" ("IDFG" is "Idaho Fish and Game").  The steps for running them are almost the same, but with MegaClassifier there's an extra step to map the output classes down to classes that an individual organization cares about.  It would almost never make sense to use MegaClassifier without an output mapping, and as per above, the only mapping we've built and tested is also for IDFG.  Neither model is obviously better or worse on data from the Western US/Canada, though they are... different in interesting ways.  90% of the time I end up just using MegaClassifier, even for IDFG data, so you can mostly ignore the "IDFG classifier".

## MegaClassifier instructions

### Files you should download before following these instructions

#### Files you need to run MegaClassifier

Model file<br/>
[megaclassifier/v0.1/megaclassifier_v0.1_efficientnet-b3_compiled.pt](https://lila.science/public/models/megaclassifier/v0.1/megaclassifier_v0.1_efficientnet-b3_compiled.pt)

Mapping of model outputs to class names<br/>
[megaclassifier/v0.1/megaclassifier_v0.1_index_to_name.json](https://lila.science/public/models/megaclassifier/v0.1/megaclassifier_v0.1_index_to_name.json)

Mapping file to get MegaClassifier classes to Idaho-relevant classes<br/>
[idfg_to_megaclassifier_labels.json](https://lila.science/public/models/megaclassifier/v0.1/idfg_to_megaclassifier_labels.json)

#### Files you need to run the IDFG classifier

Model file<br/>
[idfg_classifier_ckpt_14_compiled.pt](https://lila.science/public/models/idfg_classifier/idfg_classifier_20200905_042558/idfg_classifier_ckpt_14_compiled.pt)

Mapping of model outputs to class names<br/>
[idfg_classifier_20200905_042558/label_index.json](https://lila.science/public/models/idfg_classifier/idfg_classifier_20200905_042558/label_index.json)


### Environment setup

Install Anaconda, Miniforge, or Mambaforge.  The [MegaDetector User Guide](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md) recommends Mambaforge, but if you're reading this classification README, you're likely already pretty Python-savvy and choose your own environment.  We will use "mamba" in these instructions, but if you're using Anaconda, just replace "mamba" with "conda".

These instructions assume you are running in a Miniforge/Mambaforge/Anaconda command prompt, and that your current working folder is the folder where you cloned the MegaDetector repo when you followed the [MegaDetector setup instructions](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#using-the-model).  If you used the same folders we recommend in those instructions, on Windows, before running these commands, run `cd c:\git\MegaDetector`.

Then create the Python environment for classification:

```bash
mamba env create -f envs/environment-classifier.yml
```

...and activate this environment:

```bash
mamba activate megaclassifier
```

Subsequent steps will assume your command prompt is in the "classification" folder in this repo.

This environment replicates the training environment, which - all other things being equal - is good practice, but that environment is no longer always compatible with current GPUs.  If you are using an RTX 4090 or newer, you may experience [this issue](https://github.com/pytorch/pytorch/issues/87595) ("nvrtx: invalid value for --gpu-architecture").  Although this has not been extensively tested, if you hit this error, we recommend creating the environment this way instead:

```bash
mamba env create -f envs/environment-classifier-unpinned.yml
```


### Running MegaClassifier

#### Run MegaDetector

First, you need to run MegaDetector on your new images to get an output JSON file, typically using [run_detector_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector_batch.py), though it's also fine to use a third-party tool like [AddaxAI](https://addaxdatascience.com/addaxai/) (formerly EcoAssist).  Instructions for running MegaDetector are [here](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#using-the-model).

#### A note on all the other steps in this section

This section is going to document a bunch of Python scripts required to run MegaClassifier; in practice, I never run all these scripts directly.  This all happens within [manage_local_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.py) (or its Jupyter Notebook cousin, [manage_local_batch.ipynb](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.ipynb), a notebook that I use to run all the steps involved in an image processing job, including classification.  See the cell called "run MegaClassifier", somewhere around [here](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.py#L937).  manage_local_batch is maintained, up-to-date, and cross-platform.

There are also two somewhat-out-of-date Python scripts – one for MegaClassifier, and one for the IDFG classifier – where I set just the relevant variables (e.g. the name of the input file from MegaDetector, the link to where the images live, etc.), and those scripts generate the actual commands I'll run.  These are out of date, Linux-only, and not documented for external consumption, but they may be useful as a reference:

* [Script for generating MegaClassifier commands](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/classification/prepare_classification_script_mc.py)
* [Script for generating IDFG classifier commands](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/classification/prepare_classification_script.py)

These instructions also assume you are running in a Miniforge/Mambaforge/Anaconda command prompt, and that your current working folder is the "classification" folder within your MegaDetector folder.  If you used the same folders we recommend in the MegaDetector instructions, for example, on Windows, before running these commands, run `cd c:\git\MegaDetector\classification`.

#### Crop images

Run `crop_detections.py` to crop the bounding boxes according to the MegaDetector results.  Unless you have a good reason not to, use the `--square-crops` flag, which crops the tightest square enclosing each bounding box (which may have an arbitrary aspect ratio).

Linux-flavored example:

```bash
python crop_detections.py \
    detections.json \
    /path/to/crops \
    --images-dir /path/to/images \
    --threshold 0.15 \
    --save-full-images --square-crops \
    --threads 20 \
    --logdir "."
```

Windows-flavored example:

```bash
python crop_detections.py ^
    "c:\folder\where\you\store\your\results\detections.json" ^
    "c:\folder\for\storing\cropped\images" ^
    --images-dir "c:\folder\where\your\images\live" ^
    --threshold 0.15 ^
    --save-full-images --square-crops ^
    --threads 20 ^
    --logdir "c:\folder\where\you\store\your\results"
```

#### Run classifier

Load the model file for the classifier (the .pt file you downloaded above).

The following script will output a CSV file (optionally gzipped) whose columns are:

* `path`: path to image crop, relative to the cropped images directory
* category names: one column per classifier output category. The values are the confidence of the classifier on each category.

On a GPU, this should run at ~200 crops per second.

Linux-flavored example:

```bash
python run_classifier.py \
    /path/to/classifier-training/megaclassifier/megaclassifier_v0.1_efficientnet-b3_compiled.pt \
    /path/to/crops \
    classifier_output.csv.gz \
    --detections-json detections.json \
    --classifier-categories /path/to/classifier-training/megaclassifier/megaclassifier_v0.1_index_to_name.json \
    --image-size 300 --batch-size 64 --num-workers 8
```

Windows-flavored example:

```bash
python run_classifier.py ^
    "c:\folder\where\you\downloaded\megaclassifier\megaclassifier_v0.1_efficientnet-b3_compiled.pt" ^
    "c:\folder\for\storing\cropped\images" ^
    "c:\folder\where\you\store\your\results\megclassifier_output.csv.gz" ^
    --detections-json "c:\folder\where\you\store\your\results\detections.json" ^
    --classifier-categories "c:\folder\where\you\downloaded\megaclassifier\megaclassifier_v0.1_index_to_name.json" ^
    --image-size 300 --batch-size 64 --num-workers 8
```

#### (Optional) Map classifier categories to desired categories

<i>This part is only relevant to MegaClassifier, not the IDFG classifier.</i>

MegaClassifier outputs 100+ categories, but we usually don't care about all of them. Instead, we can group the classifier labels into desired "target" categories. This process involves three sub-steps:

* Specify the target categories that we care about.
* Build a mapping from desired target categories to MegaClassifier labels.
* Aggregate probabilities from the classifier's outputs according to the mapping.

##### Specify the target categories that we care about

Use the [label specification syntax](#megaclassifier-label-specification-syntax) to specify the taxa and/or dataset classes that constitute each target category. If using the CSV format, convert it to the JSON specification syntax using `python csv_to_json.py`.

##### Build a mapping from desired target categories to MegaClassifier labels

<i>This step build a new mapping.  If you are happy running the mapping we provide (to Western-US species), skip this step.</i>

Run the `map_classification_categories.py` script with the target label specification JSON to create a mapping from target categories to MegaClassifier labels. The output file is another JSON file representing a dictionary whose keys are target categories and whose values are lists of MegaClassifier labels. MegaClassifier labels who are not explicitly assigned a target are assigned to a target named "other". Each MegaClassifier label is assigned to exactly one target category.

Linux-flavored example:

```bash
python map_classification_categories.py \
    target_label_spec.json \
    /path/to/classifier-training/megaclassifier/megaclassifier_v0.1_label_spec.json \
    /path/to/camera-traps-private/camera_trap_taxonomy_mapping.csv \
    --output target_to_classifier_labels.json \
    --classifier-label-index /path/to/classifier-training/megaclassifier/megaclassifier_v0.1_index_to_name.json
```

##### Aggregate probabilities from the classifier's outputs according to the mapping

Using the mapping, create a new version of the classifier output CSV with probabilities summed within each target category. Also output a new "index-to-name" JSON file which identifies the sequential order of the target categories.

Linux-flavored example:

```bash
python aggregate_classifier_probs.py \
    classifier_output.csv.gz \
    --target-mapping target_to_classifier_labels.json \
    --output-csv classifier_output_remapped.csv.gz \
    --output-label-index label_index_remapped.json
```

Windows-flavored example:

```bash
python aggregate_classifier_probs.py ^
    "c:\folder\where\you\store\your\results\megclassifier_output.csv.gz"  ^
    --target-mapping "c:\folder\where\you\downloaded\megaclassifier\idfg_to_megaclassifier_labels.json" ^
    --output-csv "c:\folder\where\you\store\your\results\megclassifier_output_remapped.csv.gz"  ^
    --output-label-index "c:\folder\where\you\store\your\results\megclassifier_label_index_remapped.json"
```

#### Merge classification results with detection results

Finally, merge the classification results CSV with the original detection JSON file. Use the `--threshold` argument to exclude predicted categories from the JSON file if their confidence is below a certain threshold. The output JSON file path is specified by the `--output-json` argument. If desired, this file can then be opened in Timelapse.

Linux-flavored example:

```bash
python merge_classification_detection_output.py \
    classifier_output_remapped.csv.gz \
    label_index_remapped.json \
    --output-json detections_with_classifications.json \
    --classifier-name megaclassifier_v0.1_efficientnet-b3 \
    --threshold 0.05 \
    --detection-json detections.json
```

Windows-flavored example:

```bash
python merge_classification_detection_output.py ^
    "c:\folder\where\you\store\your\results\megclassifier_output_remapped.csv.gz" ^
    "c:\folder\where\you\store\your\results\megclassifier_label_index_remapped.json" ^
    --output-json "c:\folder\where\you\store\your\results\detections_with_classifications.json" ^
    --classifier-name megaclassifier_v0.1_efficientnet-b3 ^
    --threshold 0.05 ^
    --detection-json "c:\folder\where\you\store\your\results\detections.json"
```


### Use MegaClassifier as a feature extractor

To use MegaClassifier to extract features from images, use the following code:

```python
import efficientnet
import torch

model = efficientnet.EfficientNet.from_name('efficientnet-b3', num_classes=169)
ckpt = torch.load('megaclassifier_v0.1_efficientnet-b3.pt')
model.load_state_dict(ckpt['model'])

x = torch.rand((1, 3, 224, 224))  # random image, batch size 1
conv_features = model.extract_features(x)
print(conv_features.shape)  # torch.Size([1, 1536, 7, 7])
features = model._avg_pooling(conv_features).flatten(start_dim=1)
print(features.shape)  # torch.Size([1, 1536])
```

## MegaClassifier label specification syntax

### CSV

```
output_label,type,content

# select a specific row from the primary taxonomy CSV
<label>,row,<dataset_name>|<dataset_label>

# select all animals in a taxon from a particular dataset
<label>,datasettaxon,<dataset_name>|<taxon_level>|<taxon_name>

# select all animals in a taxon across all datasets
<label>,<taxon_level>,<taxon_name>

# exclude certain rows or taxa
!<label>,...

# set a limit on the number of images to sample for this class
<label>,max_count,<int>

# when sampling images, prioritize certain datasets over others
# is they Python syntax for List[List[str]], i.e., a list of lists of strings
<label>,prioritize,"[['<dataset_name1>', '<dataset_name2>'], ['<dataset_name3>']]"
```

A CSV label specification file can be converted to the [JSON label specification syntax](#json) via the Python script `csv_to_json.py`.

### JSON

```javascript
{
    // name of classification label
    "cervid": {

        // select animals to include based on hierarchical taxonomy,
        // optionally restricting to a subset of datasets
        "taxa": [
            {
                "level": "family",
                "name": "cervidae",
                "datasets": ["idfg", "idfg_swwlf_2019"]
                // include all datasets if no "datasets" key given
            }
        ],

        // select animals to include based on dataset labels
        "dataset_labels": {
            "idfg": ["deer", "elk", "prong"],
            "idfg_swwlf_2019": ["elk", "muledeer", "whitetaileddeer"]
        },

        "max_count": 50000,  // only include up to this many images (not crops)

        // prioritize images from certain datasets over others,
        // only used if "max_count" is given
        "prioritize": [
            ["idfg_swwlf_2019"],  // give 1st priority to images from this list of datasets
            ["idfg"]  // give 2nd priority to images from this list of datasets
            // give remaining priority to images from all other datasets
        ]

    },

    // name of another classification label
    "bird": {
        "taxa": [
            {
                "level": "class",
                "name": "aves"
            }
        ],
        "dataset_labels": {
            "idfg_swwlf_2019": ["bird"]
        },

        // exclude animals using the same format
        "exclude": {
            // same format as "taxa" above
            "taxa": [
                {
                    "level": "genus",
                    "name": "meleagris"
                }
            ],

            // same format as "dataset_labels" above
            "dataset_labels": {
                "idfg_swwlf_2019": ["turkey"]
            }
        }
    }
}
```

