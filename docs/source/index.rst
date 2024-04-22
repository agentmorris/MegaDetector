##########################################
Welcome to the MegaDetector Python package
##########################################

You've found the documentation for the `MegaDetector Python package <https://pypi.org/project/megadetector/>`_.  If you are an ecologist looking to run MegaDetector on your camera trap images, you don't need to write (or even run) any Python code, so you probably don't want this Python package.  Instead, head over to the "`Getting started with MegaDetector <https://github.com/agentmorris/MegaDetector/blob/main/getting-started.md>`_" page.

If you're a programmer-type looking to (a) run MegaDetector from Python code or (b) use other tools from the MegaDetector repo, e.g. for manipulating common camera trap data formats, you're in the right place!

************
Installation
************

You likely already guessed it: ::

  pip install megadetector


********
Examples
********

=============================
Run MegaDetector on one image
=============================

::

  from md_utils import url_utils
  from md_visualization import visualization_utils as vis_utils
  from detection import run_detector

  # This is the image at the bottom of this page, it has one animal in it
  image_url = 'https://github.com/agentmorris/MegaDetector/raw/main/images/orinoquia-thumb-web.jpg'
  temporary_filename = url_utils.download_url(image_url)

  image = vis_utils.load_image(temporary_filename)

  # This will automatically download MDv5a; you can also specify a filename.
  model = run_detector.load_detector('MDV5A')

  result = model.generate_detections_one_image(image)

  detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
  print('Found {} detections above threshold'.format(len(detections_above_threshold)))

======================================
Run MegaDetector on a folder of images
======================================

::

  from detection.run_detector_batch import load_and_run_detector_batch,write_results_to_file
  from md_utils import path_utils
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


*********************
Package documentation
*********************

.. toctree::
   :maxdepth: 2

   modules
   

****************************
Gratuitous camera trap image
****************************

.. image:: https://github.com/agentmorris/MegaDetector/raw/main/images/orinoquia-thumb-web_detections.jpg

Image credit University of Minnesota, from the `Orinoquía Camera Traps <http://lila.science/datasets/orinoquia-camera-traps/>`_ data set.


*******
Indices
*******

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
