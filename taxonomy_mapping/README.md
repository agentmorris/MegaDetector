## Overview

This folder contains scripts and some reference files used for mapping data on LILA to the iNaturalist taxonomy.  The results of this process are published [here](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/).

This folder is only for generating and maintaining this mapping If you want to <i>use</i> data from LILA, don't worry about this folder; instead, see the [lila](https://github.com/agentmorris/MegaDetector/tree/main/data_management/lila) folder.


## Mapping a new dataset into the standard taxonomy

When a new .json file comes in (usually for a new LILA dataset) and needs to be mapped to scientific names...

* Assuming this is a LILA dataset, edit the [LILA metadata file](http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt) to include the new .json and dataset name.

* Assuming this is a LILA dataset, use [get_lila_category_list.py](https://github.com/agentmorris/MegaDetector/blob/main/data_management/lila/get_lila_category_list.py) to download the .json files for every LILA dataset.  This will produced a .json-formatted dictionary mapping each dataset to all of the categories it contains.

* Use [map_new_lila_datasets.py](map_new_lila_datasets.py) to create a .csv file mapping each of those categories to a scientific name and taxonomy.  This will eventually become a subset of rows in the "primary" .csv file.  This is a semi-automated process; it will look up common names against the iNat and GBIF taxonomies, with some heuristics to avoid simple problems (like making sure that "greater_kudu" matches "greater kudu", or that "black backed jackal" matches "black-backed jackal"), but you will need to fill in a few gaps manually.  I do this with three windows open: a .csv editor, Spyder (with the cell called "manual lookup" from this script open), and a browser.  Once you generate this .csv file, it's considered permanent, i.e., the cell that wrote it won't re-write it, so manually edit to your heart's content.

* Use [preview_lila_taxonomy.py](preview_lila_taxonomy.py) to produce an HTML file full of images that you can use to make sure that the matches were sensible; be particularly suspicious of anything that doesn't look like a mammal, bird, or reptile.  Go back and fix things in the .csv file.  This script/notebook also does a bunch of other consistency checking.

* When you are totally satisfied with that .csv file, manually append it to the "primary" .csv file (lila-taxonomy-mapping.csv), which is currently in a private repository.  [preview_lila_taxonomy.py](preview_lila_taxonomy.py) can also be run against the primary file.

* Check for errors (one more time) (this should be redundant with what's now included in [preview_lila_taxonomy.py](preview_lila_taxonomy.py), but it can't hurt) by running:

    ```bash
    python taxonomy_mapping/taxonomy_csv_checker.py /path/to/taxonomy.csv
    ```
    
* Prepare the "release" taxonomy file (which removes a couple columns and removes unused rows) using [prepare_lila_taxonomy_release.py](prepare_lila_taxonomy_release.py) .

* Use [map_lila_categories.py](map_lila_categories.py) to get a mapping of every LILA data set to the common taxonomy.

* The [visualize_taxonomy.ipynb](visualize_taxonomy.ipynb) notebook demonstrates how to visualize the taxonomy hierarchy. It requires the *networkx* and *graphviz* Python packages.


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
