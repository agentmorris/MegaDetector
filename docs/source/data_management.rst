data\_management package
========================

This package contains tools for:

-  Converting frequently-used metadata formats to `COCO Camera
   Traps <https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md#coco-cameratraps-format>`_
   format
-  Converting the output of AI models (especially
   `YOLOv5 <https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/convert_output_format.py>`_)
   to the format used for AI results throughout this repo
-  Creating, visualizing, and editing COCO Camera Traps .json databases


Subpackages
-----------

.. toctree::
   :maxdepth: 4

   data_management.databases
   data_management.lila

Submodules
----------

data\_management.cct\_json\_utils module
----------------------------------------

.. automodule:: megadetector.data_management.cct_json_utils
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.cct\_to\_md module
-----------------------------------

.. automodule:: megadetector.data_management.cct_to_md
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.cct\_to\_wi module
-----------------------------------

.. automodule:: megadetector.data_management.cct_to_wi
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.coco\_to\_labelme module
-----------------------------------------

.. automodule:: megadetector.data_management.coco_to_labelme
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.coco_to_labelme
   :func: main
   :hook:
   :prog: coco_to_labelme
   :no_default_values:

data\_management.coco\_to\_yolo module
--------------------------------------

.. automodule:: megadetector.data_management.coco_to_yolo
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.coco_to_yolo
   :func: main
   :hook:
   :prog: coco_to_yolo
   :no_default_values:

data\_management.generate\_crops\_from\_cct module
--------------------------------------------------

.. automodule:: megadetector.data_management.generate_crops_from_cct
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.get\_image\_sizes module
-----------------------------------------

.. automodule:: megadetector.data_management.get_image_sizes
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.get_image_sizes
   :func: main
   :hook:
   :prog: get_image_sizes
   :no_default_values:

data\_management.labelme\_to\_coco module
-----------------------------------------

.. automodule:: megadetector.data_management.labelme_to_coco
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.labelme_to_coco
   :func: main
   :hook:
   :prog: labelme_to_coco
   :no_default_values:

data\_management.labelme\_to\_yolo module
-----------------------------------------

.. automodule:: megadetector.data_management.labelme_to_yolo
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.rename\_images module
-----------------------------------------

.. automodule:: megadetector.data_management.rename_images
   :members:
   :undoc-members:
   :show-inheritance:

.. sphinx_argparse_cli::
   :module: megadetector.data_management.rename_images
   :func: main
   :hook:
   :prog: rename_images
   :no_default_values:

data\_management.ocr\_tools module
----------------------------------

.. automodule:: megadetector.data_management.ocr_tools
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.remove\_exif module
------------------------------------

.. automodule:: megadetector.data_management.remove_exif
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

data\_management.read\_exif module
----------------------------------

.. automodule:: megadetector.data_management.read_exif
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.read_exif
   :func: main
   :hook:
   :prog: read_exif
   :no_default_values:


data\_management.resize\_coco\_dataset module
---------------------------------------------

.. automodule:: megadetector.data_management.resize_coco_dataset
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.wi\_download\_csv\_to\_coco module
---------------------------------------------------

.. automodule:: megadetector.data_management.wi_download_csv_to_coco
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.speciesnet\_to\_md module
------------------------------------------

.. automodule:: megadetector.data_management.speciesnet_to_md
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.speciesnet_to_md
   :func: main
   :hook:
   :prog: speciesnet_to_md
   :no_default_values:

data\_management.yolo\_output\_to\_md\_output module
----------------------------------------------------

.. automodule:: megadetector.data_management.yolo_output_to_md_output
   :members:
   :undoc-members:
   :show-inheritance:

data\_management.yolo\_to\_coco module
--------------------------------------

.. automodule:: megadetector.data_management.yolo_to_coco
   :members:
   :undoc-members:
   :show-inheritance:
   
data\_management.mewc\_to\_md module
-----------------------------------------

.. automodule:: megadetector.data_management.mewc_to_md
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main

.. sphinx_argparse_cli::
   :module: megadetector.data_management.mewc_to_md
   :func: main
   :hook:
   :prog: mewc_to_md
   :no_default_values:
   
