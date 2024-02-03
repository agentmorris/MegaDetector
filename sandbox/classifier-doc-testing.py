########
#
# classifier-doc-testing.py
#
# Script to validate the MegaClassifier sample instructions, including URLs.
#
########

#%% Imports and constants

from md_utils.url_utils import download_url
import os

# That's right, I want these paths to literally work
detections_file = r"c:\folder\where\you\store\your\results\detections.json"
crops_folder = r"c:\folder\for\storing\cropped\images"
image_folder = r"c:\folder\where\your\images\live"
results_folder = r"c:\folder\where\you\store\your\results"
model_folder = r"c:\folder\where\you\downloaded\megaclassifier"

megaclassifier_urls = [
    'https://lila.science/public/models/megaclassifier/v0.1/megaclassifier_v0.1_label_spec.json',
    'https://lila.science/public/models/megaclassifier/v0.1/megaclassifier_v0.1_index_to_name.json',
    'https://lila.science/public/models/megaclassifier/v0.1/megaclassifier_v0.1_efficientnet-b3_compiled.pt',
    'https://lila.science/public/models/megaclassifier/v0.1/megaclassifier_v0.1_efficientnet-b3.pt',
    'https://lila.science/public/models/megaclassifier/v0.1/idfg_to_megaclassifier_labels.json'
]

# We won't actually use these here, but I want to validate the URLs
idfg_classifier_urls = [
    'https://lila.science/public/models/idfg_classifier/idfg_classifier_20200905_042558/prepare_classification_script_megaclassifier.py',
    'https://lila.science/public/models/idfg_classifier/idfg_classifier_20200905_042558/prepare_classification_script_idfg.py',
    'https://lila.science/public/models/idfg_classifier/idfg_classifier_20200905_042558/label_index.json',
    'https://lila.science/public/models/idfg_classifier/idfg_classifier_20200905_042558/idfg_classifier_ckpt_14_compiled.pt'
]


#%% Create folders

for folder_name in [os.path.dirname(detections_file),
                    crops_folder,
                    image_folder,
                    results_folder,
                    model_folder]:
    os.makedirs(folder_name,exist_ok=True)    


#%% Download models

urls = megaclassifier_urls + idfg_classifier_urls

# url = urls[0]
for url in urls:
    bn = url.split('/')[-1]
    download_url(url,os.path.join(model_folder,bn))


#%% Download test images

from md_utils.md_tests import MDTestOptions, download_test_data
options = MDTestOptions()
options.scratch_dir = image_folder
download_test_data(options)

from md_utils.path_utils import find_images
image_file_names = find_images(image_folder, recursive=True, return_relative_paths=False)


#%% Run MegaDetector

from detection.run_detector_batch import load_and_run_detector_batch
md_results = load_and_run_detector_batch('MDV5A', image_file_names)

# Save results as relative filenames
from detection.run_detector_batch import write_results_to_file
_ = write_results_to_file(md_results, detections_file, relative_path_base=image_folder, 
                          detector_file='MDV5A')


#%% Paste the instructions from the MegaClassifier page, they should work now
