"""

 md-pip-test.py

 Basic test driver to validate a pip install of the MD PyPI package.  

 The main test driver - which tests both Python and CLI invocation of
 most modules - is in md_utils/md_tests.py.

"""

### Run MegaDetector on one image

from md_utils import url_utils
from md_visualization import visualization_utils as vis_utils
from detection import run_detector

# This is the image at the bottom of this page, it has one animal in it.
image_url = 'https://github.com/agentmorris/MegaDetector/raw/main/images/orinoquia-thumb-web.jpg'
temporary_filename = url_utils.download_url(image_url)

image = vis_utils.load_image(temporary_filename)

# This will automatically download MDv5a; you can also specify a filename.
model = run_detector.load_detector('MDV5A')

result = model.generate_detections_one_image(image)

detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
print('Found {} detections above threshold'.format(len(detections_above_threshold)))


### Run MegaDetector on a folder of images

from detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file
from md_utils import path_utils
import os

# Pick a folder to run MD on recursively, and an output file
image_folder = '/tmp/md-tests/md-test-images'
output_file = os.path.expanduser('~/megadetector_output_test.json')

# Recursively find images
image_file_names = path_utils.find_images(image_folder,recursive=True)
print('Processing {} images from {}'.format(len(image_file_names),image_folder))

# This will automatically download MDv5a; you can also specify a filename.
detector_filename = 'MDV5A'

results = load_and_run_detector_batch(detector_filename, image_file_names, quiet=True)

# Write results to a format that Timelapse and other downstream tools like.
write_results_to_file(results,
                      output_file,
                      relative_path_base=image_folder,
                      detector_file=detector_filename)

