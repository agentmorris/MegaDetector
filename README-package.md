# MegaDetector

This package is a pip-installable version of the support/inference code for [MegaDetector](https://github.com/agentmorris/MegaDetector/?tab=readme-ov-file#megadetector), an object detection model that helps conservation biologists spend less time doing boring things with camera trap images.  Complete documentation for this Python package is available at [megadetector.readthedocs.io](https://megadetector.readthedocs.io).

If you aren't looking for the Python package specifically, and you just want to learn more about what MegaDetector is all about, head over to the [MegaDetector repo](https://github.com/agentmorris/MegaDetector/?tab=readme-ov-file#megadetector).

If you don't want to run MegaDetector, and you just want to use the utilities in this package - postprocessing, manipulating large volumes of camera trap images, etc. - you may want to check out the [megadetector-utils](https://pypi.org/project/megadetector-utils/) package, which is identical to this one, but excludes all of the PyTorch/YOLO dependencies, and is thus approximately one zillion times smaller.

## Installation

Install with:

`pip install megadetector`

MegaDetector model weights aren't downloaded at the time you install the package, but they will be (optionally) automatically downloaded the first time you run the model.

## Package reference

See [megadetector.readthedocs.io](https://megadetector.readthedocs.io).


## Examples of things you can do with this package

### Run MegaDetector on one image and count the number of detections

```
from megadetector.utils import url_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector

# This is the image at the bottom of this page, it has one animal in it
image_url = 'https://github.com/agentmorris/MegaDetector/raw/main/images/orinoquia-thumb-web.jpg'
temporary_filename = url_utils.download_url(image_url)

image = vis_utils.load_image(temporary_filename)

# This will automatically download MDv5a; you can also specify a filename.
model = run_detector.load_detector('MDV5A')

result = model.generate_detections_one_image(image)

detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
print('Found {} detections above threshold'.format(len(detections_above_threshold)))
```

### Run MegaDetector on a folder of images

```
from megadetector.detection.run_detector_batch import \
    load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils
import os

# Pick a folder to run MD on recursively, and an output file
image_folder = os.path.expanduser('~/megadetector_test_images')
output_file = os.path.expanduser('~/megadetector_output_test.json')

# Recursively find images
image_file_names = path_utils.find_images(image_folder,recursive=True)

# This will automatically download MDv5a; you can also specify a filename.
results = load_and_run_detector_batch('MDV5A', image_file_names)

# Write results to a format that Timelapse and other downstream tools like.
write_results_to_file(results,
                      output_file,
                      relative_path_base=image_folder,
                      detector_file=detector_filename)
```

## Contact

Contact <a href="cameratraps@lila.science">cameratraps@lila.science</a> with questions.

## Gratuitous animal picture

<img src="https://github.com/agentmorris/MegaDetector/raw/main/images/orinoquia-thumb-web_detections.jpg"><br/>Image credit University of Minnesota, from the [Orinoquía Camera Traps](http://lila.science/datasets/orinoquia-camera-traps/) data set.
