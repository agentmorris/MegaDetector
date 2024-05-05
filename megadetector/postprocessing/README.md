# Postprocessing tools for MegaDetector results

This folder contains scripts for working with MegaDetector results, i.e. results in the [MegaDetector results format](https://github.com/agentmorris/MegaDetector/tree/main/api/batch_processing#megadetector-batch-output-format).

Modules in this folder include...

## postprocess_batch_results.py

Renders an HTML page that previews the results in a MegaDetector results file. We use this to help new users get a sense of whether MegaDetector worked well on their images, and in particular to quickly assess whether MegaDetector is missing anything.  You can see a sample of the output of this script [here](https://lila.science/public/snapshot_safari_public/snapshot-safari-kar-2022-00-00-v5a.0.0_0.200/).  If ground truth is not provided, the sampled images are divided into links called `non-detections`, `animal`, `person`, etc.; if ground truth (in CCT format) is provided, results are divided into true/false positives/negatives, and a precision-recall curve is plotted. 


## separate_detections_into_folders.py

Given a MegaDetector results file and the original image folder, creates a new image folder in which images have been separated according to MegaDetector results.  Typically used by MegaDetector users who are not using Timelapse to review their results, e.g. folks who want to upload their animals to a citizen-science platform, but don't want to upload the blanks or the people.  Optionally also renders boxes on detected objects.

This script preserves relative paths within each of the output folders; for example, if your .json file has these images:

* a/b/c/1.jpg
* a/b/d/2.jpg
* a/b/e/3.jpg
* a/b/f/4.jpg
* a/x/y/5.jpg

And let's say:

* The results say that the first three images are empty/person/vehicle, respectively
* The fourth image is above threshold for "animal" and "person"
* The fifth image contains an animal
* You specify an output base folder of c:\out

You will get the following files:

* c:\out\empty\a\b\c\1.jpg
* c:\out\people\a\b\d\2.jpg
* c:\out\vehicles\a\b\e\3.jpg
* c:\out\animal_person\a\b\f\4.jpg
* c:\out\animals\a\x\y\5.jpg


## convert_output_format.py

Converts MegaDetector output from .json to .csv.


## combine_api_outputs.py

Merges two or more MegaDetector .json files into one.


## compare_batch_results.py

Compares two or more MegaDetector results files, often used to compare MegaDetector v5a to v5b, MegaDetector v4 to v5, or the raw output of MegaDetector with a processed version.  Comparing the literal boxes is generally not interesting; this script tries to answer the question "which images did each file consider an animal that the other one didn't?"  You can see an example comparison file [here](https://lila.science/public/ena24/ena24-comparison-2022-06-08/).


## merge_detections.py

Merges the high-confidence detections from one file into another results file that was based on the same images.  Used in rare scenarios where we run both MDv5a and MDv5b, or MDv4 and MDv5, and we want a "best of both worlds" output file.


## categorize_detections_by_size.py

Though this isn't exactly the stuff you write machine learning papers about, it's often useful to categorize detections into "big" and "small".  This is particularly useful in snowy ecosystems where (a) animals tend to be big and furry and often smush their fur against the camera and (b) snow often obscures the lens and creates false detections.  It's remarkable how much a snowy lens can look like elk fur.  So it's often faster to review the very large detections as a single category.


## subset_json_detector_output.py

Splits .json files into smaller .json files (e.g. one results file per image folder), and allows a few other filename manipulations within the .json file (e.g. string replacement operations).  Note that if you're thinking of using this, it *might* be easier to achieve what you want to achieve in a simple text editor.  More generally, if you're having trouble lining your filenames up with your Timelapse project, feel free to [email us](mailto:cameratraps@lila.science).


## repeat_detection_elimination

Scripts and tools for semi-automated removal of detections on pesky rocks and sticks that are detected thousands of times in a row.


## CameraTrapJsonFileProcessingApp

This contains a GUI application that does the same thing as "subset_json_detector_output.py": various filename manipulation operations to adjust your .json file, typically to match your Timelapse folder structure.  Again, if you're thinking of using this, it *might* be easier to achieve what you want to achieve in a simple text editor.  More generally, if you're having trouble lining your filenames up with your Timelapse project, feel free to [email us](mailto:cameratraps@lila.science).

<img src="images/CameraTrapJsonManagerApp.jpg" width="500">
