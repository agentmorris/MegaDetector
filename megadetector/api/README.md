# MegaDetector APIs

Though many users run MegaDetector locally, we have packaged MegaDetector and associated inference scripts into two APIs that can process camera trap images in a variety of scenarios.  This folder used to contain the source code of the APIs, and documentation on how to set them up.  All of the code has been moved to the [archive](../archive/api) folder; what's left here is just documentation.


### Synchronous API

This API is intended for real-time scenarios where a small number of images are processed at a time and latency is a priority.  See documentation [here](../archive/synchronous).


### Batch processing API

This API runs MegaDetector on lots of images (typically millions) and distributes the work over potentially many nodes using [Azure Batch](https://azure.microsoft.com/en-us/services/batch/). See documentation [here](batch_processing).


### Gratuitous camera trap picture

![cat in camera trap](../../images/orinoquia-thumb-web.jpg)<br/>Image credit University of Minnesota, from the [Orinoquía Camera Traps](http://lila.science/datasets/orinoquia-camera-traps/) data set.


