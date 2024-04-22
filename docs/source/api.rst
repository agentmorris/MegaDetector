api package
===========

This package contains code for hosting our models as an API, either for synchronous operation (i.e., for real-time inference) or as a batch process (for large biodiversity surveys).  

Common operations one might do after running MegaDetector are found in the :doc:`api.batch_processing.postprocessing <./api.batch_processing.postprocessing>` submodule, for example, generating preview pages to summarize your results, separating images into different folders based on MegaDetecotr results, or converting results to different formats.

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   api.batch_processing

Module contents
---------------

.. automodule:: api
   :members:
   :undoc-members:
   :show-inheritance:
