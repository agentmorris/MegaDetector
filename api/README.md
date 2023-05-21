# MegaDetector APIs

Though many users run MegaDetector locally, we also package MegaDetector and associated inference scripts into two APIs that can process camera trap images in a variety of scenarios. This folder contains the source code of the APIs, documentation on how to set them up, and tools for working with the output they produce.  We don't currently operate any public API instances.


### Synchronous API

This API is intended for real-time scenarios where a small number of images are processed at a time and latency is a priority.  See documentation [here](synchronous).


### Batch processing API

This API runs MegaDetector on lots of images (typically millions) and distributes the work over potentially many nodes using [Azure Batch](https://azure.microsoft.com/en-us/services/batch/). See documentation [here](batch_processing).

The [batch_processing](batch_processing) folder also contains lots of postprocessing scripts (results preview, format conversion, etc.) that were originally used with batch API output, but are now used almost exclusively with local inference output (from [run_detector_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/detection/run_detector_batch.py)), which uses the same output format.


### Gratuitous camera trap picture

![cat in camera trap](../images/orinoquia-thumb-web.jpg)<br/>Image credit University of Minnesota, from the [Orinoquía Camera Traps](http://lila.science/datasets/orinoquia-camera-traps/) data set.


