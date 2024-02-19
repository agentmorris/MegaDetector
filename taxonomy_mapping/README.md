## Overview

This folder contains scripts and some reference files used for mapping data on LILA to the iNaturalist taxonomy.  The results of this process are published [here](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/).

This folder is only for generating and maintaining this mapping If you want to <i>use</i> data from LILA, don't worry about this folder; instead, see the [lila](https://github.com/agentmorris/MegaDetector/tree/main/data_management/lila) folder.


## To add a new dataset come to the harmonized LILA taxonomy

### Update the index file

* Edit the [LILA camera trap datasets index file](http://lila.science/wp-content/uploads/2023/06/lila_camera_trap_datasets.csv) to include the new dataset name, metadata URLs, and MD results URLs.  The copy on LILA is the source of truth.


### Update the taxonomy mapping

* Use [get_lila_annotation_counts.py](https://github.com/agentmorris/MegaDetector/blob/main/data_management/lila/get_lila_annotation_counts.py) to download the .json files for every LILA dataset, and list all the category names prsent in each .json file.  This will produced a .json-formatted dictionary mapping each dataset to all of the categories it contains (lila_dataset_to_categories.json).

* Use [map_new_lila_datasets.py](map_new_lila_datasets.py) to create a .csv file mapping each category in the dataset of interest to a scientific name and taxonomy.  This will eventually become a subset of rows in the "primary" .csv file.  This is a semi-automated process; a first pass will automatically look up common names against the iNat and GBIF taxonomies, with some heuristics to avoid simple problems (like making sure that "greater_kudu" matches "greater kudu", or that "black backed jackal" matches "black-backed jackal"), but you will need to fill in a few gaps manually.  Specifically:

  * Set "datasets_to_map" and "output_file" appropriately
  * Run the whole script; this will create the .csv file you'll be working with
  * Open that .csv file, and use the "manual lookup" cell to fix things that matched incorrectly or didn't match at all.  I do this with three windows open: a .csv editor, Spyder (with the cell called "manual lookup" from this script open), and a browser.  Leave all versions of "empty" as empty rows, other than the dataset name and label.

* Use [preview_lila_taxonomy.py](preview_lila_taxonomy.py) to produce an HTML file full of images that you can use to make sure that the matches were sensible; be particularly suspicious of anything that doesn't look like a mammal, bird, or reptile.  Go back and fix things in the .csv file.  This script/notebook also does a bunch of other consistency checking, e.g. making sure that if the "taxonomy_level" column says "species", the "taxonomy_string" column is actually a species.

* When you are satisfied with that .csv file, manually append it to the "primary" .csv file (lila-taxonomy-mapping.csv), which is currently in a private repository (note to self: private-repo/lila-taxonomy/lila-taxonomy-mapping.csv).  [preview_lila_taxonomy.py](preview_lila_taxonomy.py) can also be run against the primary file.


### Stuff we do once we've updated the source taxonomy file

* Check for errors (one more time) (this should be redundant with what's now included in [preview_lila_taxonomy.py](preview_lila_taxonomy.py), but it can't hurt) by running:

    ```bash
    python taxonomy_mapping/taxonomy_csv_checker.py private-repo/lila-taxonomy/lila-taxonomy-mapping.csv
    ```
    
* Prepare the "release" taxonomy file (which removes a couple columns and removes unused rows) using [prepare_lila_taxonomy_release.py](prepare_lila_taxonomy_release.py).  This will create "lila-taxonomy-mapping_release.csv" in the local LILA base folder.  Run the taxonomy checker against this file too, just to be safe.

* Upload to <https://lila.science/public/lila-taxonomy-mapping_release.csv>.  This is a small file that does not get zipped.


### Test the metadata file, index file, and MD results files

* Use 'test_lila_metadata_urls.py' to verify that the metadata .csv files and MegaDetector results files exist, and that their contents point to base URLs that actually exist.  I.e., make sure that all the metadata URLs and MD results files are programmatically usable.


### Update the One True CSV file

* Use 'generate_lila_per_image_labels.py' to generate the main .csv table (lila_image_urls_and_labels.csv).  This takes a few hours.  At the end of this process, you'll have the zipped version as well, as "lila_image_urls_and_labels.csv" and "lila_image_urls_and_labels.csv.zip" in the local LILA base folder.

* Upload to <https://lila.science/public/lila_image_urls_and_labels.csv.zip>

* Test the new .csv file by running create_lila_test_set.py and download_lila_subset.py


## To make small changes to the taxonomy mapping

* If you're likely to need help from the semi-automated matching tools, open map_new_lila_datasets.py,  run the initialization cells, and head directly to the "manual lookup" cell at the end.

* Open the source taxonomy (private-repo/lila-taxonomy/lila-taxonomy-mapping.csv) in your favorite .csv editor.

* Make your changes.

* Run all the steps in the [stuff we do once we've updated the source taxonomy file](#stuff-we-do-once-weve-updated-the-source-taxonomy-file) section.


## Files in this folder

<b>[map_lila_taxonomy_to_wi_taxonomy.py](map_lila_taxonomy_to_wi_taxonomy.py)</b>

Loads the [Wildlife Insights taxonomy](https://www.wildlifeinsights.org/get-started/taxonomy), does some internal consistency checking, and maps all scientific names to the iNat taxonomy.  A few categories are mapped manually, from the file [lila_to_wi_supplementary_mapping_file.csv](lila_to_wi_supplementary_mapping_file.csv).

<b>[map_new_lila_datasets.py](map_new_lila_datasets.py)</b>

Interactive notebook used to map the categories in a new dataset into the common taxonomy.  E.g., given a .json file in which we have a category called "bear", semi-automatically work out the species to which we want to map that dataset/category pair.

<b>[prepare_lila_taxonomy_release.py](prepare_lila_taxonomy_release.py)</b>

Given the private intermediate taxonomy mapping (produced by map_new_lila_datasets.py), prepare the public (release) taxonomy mapping file (lila-taxonomy-mapping.csv).

<b>[preview_lila_taxonomy.py](preview_lila_taxonomy.py)</b>

Does some consistency-checking on the LILA taxonomy file, and generates an HTML preview page that we can use to determine whether the mappings make sense.

<b>[retrieve_sample_image.py](retrieve_sample_image.py)</b>

Wrapper for simple_image_download.py that makes it easier for me to toggle download backends.  Used for downloading images from the Web to make sure that taxonomy mappings look reasonable.

<b>[simple_image_download.py](simple_image_download.py)</b>

Slightly modified version of [simple_image_download](https://github.com/RiddlerQ/simple_image_download). Used for downloading images from the Web to make sure that taxonomy mappings look reasonable.

<b>[species_lookup.py](species_lookup.py)</b>

Functions for looking up species names in the GBIF and iNat taxonomies.  Called from map_new_lila_datasets and preview_lila_taxonomy.

<b>[taxonomy_csv_checker.py](taxonomy_csv_checker.py)</b>

Checks the taxonomy CSV file for internal consistency.

<b>[taxonomy_graph.py](taxonomy_graph.py)</b>

Methods for transforming taxonomy CSV into a graph structure backed by [NetworkX](https://networkx.org/).

<b>[validate_lila_category_mappings.py](validate_lila_category_mappings.py)</b>

Confirm that all category names on LILA have mappings in the taxonomy file.  Not strictly required as part of the mapping process, just a post-hoc consistency check.

<b>[visualize_taxonomy.ipynb](visualize_taxonomy.ipynb)</b>

Notebook for visualizing the taxonomy .csv file.

<b>[lila_to_wi_supplementary_mapping_file.csv](lila_to_wi_supplementary_mapping_file.csv)</b>

Manual mappings used to resolve ambiguity in the Wildlife Insight &rarr; iNat taxonomy mapping.
