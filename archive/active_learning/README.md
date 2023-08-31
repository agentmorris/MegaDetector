# Active learning for camera traps

## Overview

This repository contains the code, models, and instructions to run active deep learning animal identification as described in:

Norouzzadeh MS, Morris D, Beery S, Joshi N, Jojic N, Clune J. [A deep active learning system for species identification and counting in camera trap images](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13504). Methods in ecology and evolution. 2021 Jan;12(1):150-61.

## Contents

The remainder of this README summarizes the main folders in this repo.

### . (root)

#### train_embedding.py

This script learns an embedding model for a given dataset or set of datasets using either triplet, constrastive, or cross entropy loss.
Please refer to the command line arguments and the comments inside the code for further details.

#### run_active_learning.py

This script runs the active learning model on a given dataset. The algorithm starts with 1,000 randomly selected queries and then actively choose the other labels to be labeled by the oracle. This scripts does not consider an oracle in the loop for labeling. Instead, it simulates an oracle by using pre-labeled samples. For further details, please refer to the command line arguments and the comments inside the code.

### data_preprocessing

Produces crops either from detector_output.csv produced by run_tf_detector_batch.py (via crop_images_from_batch_api_detections.py) or from bboxes stored in COCO .json file. Either way, produces a .json file in the format expected by Database/initialize_*.py.

### Database

* initialize_*.py populates a database (target or pretrain).
* DB_models.py defines the data representation that's built; this is what's used by  the rest of the code.
* add_oracle_to_db.py adds ground truth to a db for offline experiments.

### experiments

One-off scripts and notebooks.

### labeling_tool

See [labeling_tool/README.md](labeling_tool/README.md).

### DL

#### Engine.py

ML utility functions like "train()" and "validate()"

#### losses.py

Loss functions: focal, center, triplet, contrastive

#### networks.py

* EmbeddingNet loads pre-trained embedding models
* SoftMaxNet is a wrapper for EmbeddingNet adding softmax loss 
* ClassificationNet was the hand-created classification network, but was replaced by a network built into scikit learn

### sqlite_data_loader

Loads a data set from an existing SQLite DB.

### sampling_methods

Query tools for image selection for active learning.

Mostly third-party code from:

https://github.com/google/active-learning/tree/master/sampling_methods

New stuff:

* entropy
* confidence
* uniform_sampling (modified)
* constants.py (modified)

Also miscellaneous fixes related to database assumptions.
