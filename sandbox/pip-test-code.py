### Run MegaDetector on one image

from md_utils import url_utils
from md_visualization import visualization_utils as vis_utils
from detection import run_detector

# This is the image at the bottom of this page, it has one animal in it
image_url = 'https://github.com/agentmorris/MegaDetector/raw/main/images/orinoquia-thumb-web.jpg'
temporary_filename = url_utils.download_url(image_url)

image = vis_utils.load_image(temporary_filename)

# This will automatically download MDv5a to the system temp folder;
# you can also specify a filename explicitly, or set the $MDV5A
# environment variable to point to the model file.
model = run_detector.load_detector('MDV5A')

result = model.generate_detections_one_image(image)

detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
print('Found {} detection above threshold'.format(len(detections_above_threshold)))


### Run MegaDetector on a folder of images

from detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file
from md_utils import path_utils
import os

# Pick a folder to run MD on recursively, and an output file
image_folder = os.path.expanduser('~/megadetector_test_images')
output_file = os.path.expanduser('~/megadetector_output_test.json')

# Recursively find images
image_file_names = path_utils.find_images(image_folder,recursive=True)

# This will automatically download MDv5a to the system temp folder;
# you can also specify a filename explicitly, or set the $MDV5A
# environment variable to point to the model file.
detector_filename = 'MDV5A'

results = load_and_run_detector_batch(detector_filename, image_file_names, quiet=True)

# Write results as relative filenames; this is what Timelapse
# and other downstream tools expect.  The detector file is being
# passed here only so the version information can get written to the 
# output file. 
write_results_to_file(results,output_file,relative_path_base=image_folder,
    detector_file=detector_filename)

