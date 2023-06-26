## Overview

This folder contains scripts for preparing data for upload to [LILA](https://lila.science), and working with LILA index files.

## Files in this folder

<b>[lila_common.py](lila_common.py)</b>

Shared utilities for working with LILA data, including hard-coded URLs where important metadata files live.

<b>[create_lila_test_set.py](create_lila_test_set.py)</b>

Download a test set of camera trap images, containing N empty and N non-empty images from each LILA data set.  Also a useful test driver for the most important parts of lila_common.py.

<b>[download_lila_subset.py](download_lila_subset.py)</b>

Example of how to download a list of files from LILA, e.g. all the files in all data sets matching a common name.

<b>[generate_lila_per_image_labels.py](generate_lila_per_image_labels.py)</b>

You don't want to run this file if you're not a LILA maintainer; this is used to generate the [giant .csv file](https://lila.science/public/lila_image_urls_and_labels.csv.zip) with one row per annotation, in a common taxononomic universe.

<b>[get_lila_annotation_counts.py](get_lila_annotation_counts.py)</b>

Generates a .json-formatted dictionary mapping each LILA dataset to all categories that exist for that dataset, with counts for the number of annotations for each category.

<b>[get_lila_image_counts.py](get_lila_image_counts.py)</b>

Generates a .json-formatted dictionary mapping each LILA dataset to all categories that exist for that dataset, with counts for the number of images for each category.

