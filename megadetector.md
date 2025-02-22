# MegaDetector User Guide

## Table of contents

1. [MegaDetector overview](#megadetector-overview)
2. [Our ask to MegaDetector users](#our-ask-to-megadetector-users)
3. [Who is using MegaDetector?](#who-is-using-megadetector)
4. [How fast is MegaDetector, and can I run it on my giant/small computer?](#how-fast-is-megadetector-and-can-i-run-it-on-my-giantsmall-computer)
5. [Downloading the model (optional)](#downloading-the-model-optional)
6. [Using the model](#using-the-model)
7. [OK, but is that how the MD devs run the model?](#ok-but-is-that-how-the-md-devs-run-the-model)
8. [Is there a GUI?](#is-there-a-gui)
9. [How do I use the results?](#how-do-i-use-the-results)
10. [Have you evaluated MegaDetector's accuracy?](#have-you-evaluated-megadetectors-accuracy)
11. [What is MegaDetector bad at?](#what-is-megadetector-bad-at)
12. [Pro tips for coaxing every bit of accuracy out of MegaDetector](#pro-tips-for-coaxing-every-bit-of-accuracy-out-of-megadetector)
13. [Citing MegaDetector](#citing-megadetector)
14. [Tell me more about why detectors are a good first step for camera trap images](#tell-me-more-about-why-detectors-are-a-good-first-step-for-camera-trap-images)
15. [Pretty picture](#pretty-picture)
16. [Mesmerizing video](#mesmerizing-video)
17. [Can you share the training data?](#can-you-share-the-training-data)
18. [What if I just want to run non-MD scripts from this repo?](#what-if-i-just-want-to-run-non-md-scripts-from-this-repo)
19. [What if I want to use MD without all the baggage of your very specific package versions?](#what-if-i-want-to-use-md-without-all-the-baggage-of-your-very-specific-package-versions)


## MegaDetector overview

Conservation biologists invest a huge amount of time reviewing camera trap images, and a huge fraction of that time is spent reviewing images they aren't interested in.  This primarily includes empty images, but for many projects, images of people and vehicles are also "noise", or at least need to be handled separately from animals.

*Machine learning can accelerate this process, letting biologists spend their time on the images that matter.*

To this end, this page hosts a model we've trained - called "MegaDetector" - to detect animals, people, and vehicles in camera trap images.  It does not identify animals to the species level, it just finds them.  

Before you read the rest of this page...

* If you are looking for a convenient tool to run MegaDetector, you don't need anything from this page: check out [AddaxAI](https://addaxdatascience.com/addaxai/) (formerly EcoAssist).
* If you're just <i>considering</i> the use of AI in your workflow, and you aren't even sure yet whether MegaDetector would be useful to you, we recommend reading the much shorter "[getting started with MegaDetector](getting-started.md)" page.
* If you're a programmer-type looking to use tools from this repo, check out the [Python package](https://pypi.org/project/megadetector/) that provides access to everything in this repo (yes, you guessed it, "pip install megadetector").
* If you're already familiar with MegaDetector and you're ready to run it on your data, and you're looking for instructions on running MegaDetector, read on!
* If you have any questions, or you want to tell us that MegaDetector was amazing/terrible on your images, <a href="mailto:cameratraps@lila.science">email us</a>!

MegaDetector is just one of many tools that aims to make conservation biologists more efficient with AI.  If you want to learn about other ways to use AI to accelerate camera trap workflows, check out our of the field, affectionately titled &ldquo;[Everything I know about machine learning and camera traps](https://agentmorris.github.io/camera-trap-ml-survey/)&rdquo;.


## Our ask to MegaDetector users

MegaDetector is free, and it makes us super-happy when people use it, so we put it out there as a downloadable model that is easy to use in a variety of conservation scenarios.  That means we don't know who's using it unless you contact us (or we happen to run into you), so please please pretty-please email us at [cameratraps@lila.science](mailto:cameratraps@lila.science) if you find it useful!


## How fast is MegaDetector, and can I run it on my giant/small computer?

We often run MegaDetector on behalf of users as a free service; see our ["Getting started with MegaDetector"](getting-started.md) page for more information.  But there are many reasons to run MegaDetector on your own; how practical this is will depend in part on how many images you need to process and what kind of computer hardware you have available.  MegaDetector is designed to favor accuracy over speed, and we typically run it on <a href="https://en.wikipedia.org/wiki/Graphics_processing_unit">GPU</a>-enabled computers.  That said, you can run anything on anything if you have enough time, and we're happy to support users who run MegaDetector on their own GPUs (in the cloud or on their own PCs), on their own CPUs, or even on embedded devices.  If you only need to process a few thousand images per week, for example, a typical laptop will be just fine.  If you want to crunch through 20 million images as fast as possible, you'll want at least one GPU.

Here are some rules of thumb to help you estimate how fast you can run MegaDetector on different types of hardware.

* On a decent laptop (without a fancy deep learning GPU) that is neither the fastest nor slowest laptop you can buy in 2023, MegaDetector v5 can process somewhere between 25,000 and 50,000 images per day.  This might be totally fine for scenarios where you have even hundreds of thousands of images, as long as you can wait a few days.
* On a dedicated deep learning GPU that is neither the fastest nor slowest GPU you can buy in 2023, MegaDetector v5 can process between 300,000 and 1,000,000 images per day.  We include a few <a href="#benchmark-timings">benchmark timings</a> below on some specific GPUs.

We don't typically recommend running MegaDetector on embedded devices, although <a href="https://www.electromaker.io/project/view/whats-destroying-my-yard-pest-detection-with-raspberry-pi">some folks have done it</a>!  More commonly, for embedded scenarios, it probably makes sense to use MegaDetector to generate bounding boxes on lots of images from your specific ecosystem, then use those boxes to train a smaller model that fits your embedded device's compute budget.

### Benchmark timings

These results are based on a test batch of around 13,000 images from the public <a href="https://lila.science/datasets/snapshot-karoo">Snapshot Karoo</a> and <a href="http://lila.science/datasets/idaho-camera-traps/">Idaho Camera Traps</a> datasets.  These were chosen to be "typical", and anecdotally they are, though FWIW we have seen very high-resolution images that run around 30% slower than these, and very low-resolution images (typically video frames) that run around 100% faster than these.</i>

Some of these results were measured by "team MegaDetector", and some are user-reported; YMMV.  
  
#### Timing results for MDv5

* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/">RTX 4090</a> processes around 17.6 images per second, or around 1,500,000 images per day (for MDv5)
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/">RTX 3090</a> processes around 11.4 images per second, or around 985,000 images per day (for MDv5)
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/">RTX 3080</a> processes around 9.5 images per second, or around 820,800 images per day (for MDv5)
* A desktop <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/">RTX 3050</a> processes around 4.2 images per second, or around 363,000 images per day (for MDv5)
* A laptop  <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/">RTX 3050</a> processes around 3.0 images per second, or around 250,000 images per day (for MDv5)
* A <a href="https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-pascal-p2000-data-sheet-us-nvidia-704443-r2-web.pdf">Quadro P2000</a> processes around 2.1 images per second, or around 180,000 images per day (for MDv5)
* A 2024 M3 MacBook Pro (18 GPU cores) averages around 4.61 images per second, or around 398,000 images per day (for MDv5)
* A 2020 M1 MacBook Pro (8 GPU cores) averages around 1.85 images per second, or around 160,000 images per day (for MDv5)
* An Intel Core i7-12700 CPU processes around 0.5 images per second on a single core (43,000 images per day) (multi-core performance is... complicated) (for MDv5)
* An Intel Core i7-13700K CPU processes around 0.8 images per second on a single core (69,000 images per day) (multi-core performance is... complicated) (for MDv5)

#### Timing results for MDv4

FWIW, MDv5 is consistently 3x-4x faster than MDv4, so if you see a device listed here and want to estimate MDv5 performance, assume 3x-4x speedup.

* An <a href="https://www.nvidia.com/en-us/data-center/v100/">NVIDIA V100</a> processes around 2.79 images per second, or around 240,000 images per day (for MDv4)
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/">NVIDIA RTX 3090</a> processes ~3.24 images per second, or ~280,000 images per day (for MDv4)
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/">NVIDIA RTX 2080 Ti</a> processes ~2.48 images per second, or ~214,000 images per day (for MDv4)
* An <a href="https://www.nvidia.com/en-us/geforce/20-series/">NVIDIA RTX 2080</a> processes ~2.0 images per second, or ~171,000 images per day (for MDv4)
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2060-super/">NVIDIA RTX 2060 SUPER</a> processes ~1.64 images per second, or ~141,000 images per day (for MDv4)
* An <a href="https://www.nvidia.com/en-us/titan/titan-v/">NVIDIA Titan V</a> processes ~1.9 images per second, or ~167,000 images per day (for MDv4)
* An <a href="https://www.notebookcheck.net/NVIDIA-Quadro-T2000-Laptop-Graphics-Card.423971.0.html">NVIDIA Titan Quadro T2000</a> processes ~0.64 images per second, or ~55,200 images per day (for MDv4)

 #### Contributing to this benchmark list
 
If you want to run this benchmark on your own, here are <a href="https://github.com/agentmorris/MegaDetector/blob/main/sandbox/download_megadetector_timing_benchmark_set.bat">azcopy commands</a> to download those 13,226 images, and we're happy to help you get MegaDetector running on your setup.  Or if you're using MegaDetector on other images with other GPUs, we'd love to include that data here as well.  <a href="mailto:cameratraps@lila.science">Email us</a>!

### User-reported timings on other data

Speed can vary widely based on image size, hard drive speed, etc., and in these numbers we're just taking what users report without asking what the deal was with the data, so... YMMV.

* A GTX 1080 processed 699,530 images in just over 44 hours through MDv5 (4.37 images per second, or ~378,000 images per day)
* An RTX 3050 processes ~4.6 images per second, or ~397,000 images per day through MDv5
* An RTX 3090 processes ~11 images per second, or ~950,000 images per day through MDv5

## Who is using MegaDetector?

See <a href="https://github.com/agentmorris/MegaDetector/#who-is-using-megadetector">this list</a> on the repo's main page.


## Downloading the model (optional)

In previous versions of these instructions, you had to download MegaDetector to your PC before running it.  The scripts we use to run MegaDetector can now automatically download MegaDetector, so <b>this whole download step is optional now</b>, and if you're going to follow the instructions on this page, <b>you can probably ignore this section and skip to the "[Using the model](#using-the-model)" section</b>.

That said, in this section, we provide download links for lots of MegaDetector versions.  Unless you have a very esoteric scenario, you want MegaDetector v5, and you can ignore all the other MegaDetector versions.  The rest of this section, after the MDv5 download links, is more like a mini-MegaDetector-museum than part of the User Guide.

### MegaDetector v5.0, 2022.06.15

#### Release notes

This release incorporates additional training data, specifically aiming to improve our coverage of:

* Boats and trains in the "vehicle" class
* Artificial objects (e.g. bait stations, traps, lures) that frequently overlap with animals
* Rodents, particularly at close range
* Reptiles and small birds

This release also represents a change in MegaDetector's architecture, from Faster-RCNN to [YOLOv5](https://github.com/ultralytics/yolov5).  Our inference scripts have been updated to support both architectures, so the transition should be <i>mostly</i> seamless.

MDv5 is actually two models (MDv5a and MDv5b), differing only in their training data (see the [training data](#can-you-share-the-training-data) section for details).  Both appear to be more accurate than MDv4, and both are 3x-4x faster than MDv4, but each MDv5 model can outperform the other slightly, depending on your data.  When in doubt, for now, try them both.  If you really twist our arms to recommend one... we recommend MDv5a.  But try them both and tell us which works better for you!  The [pro tips](#pro-tips-for-coaxing-every-bit-of-accuracy-out-of-megadetector) section contains some additional thoughts on when to try multiple versions of MD.

See the [release page](https://github.com/agentmorris/MegaDetector/releases/tag/v5.0) for more details, and in particular, be aware that the range of confidence values produced by MDv5 is very different from the range of confidence values produced by MDv4!  <i>Don't use your MDv4 confidence thresholds with MDv5!</i>

#### Download links

* [MegaDetector v5a (.pt)](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)
* [MegaDetector v5b (.pt)](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt)


### MegaDetector v4.1, 2020.04.27

#### Release notes

This release incorporates additional training data from Borneo, Australia and the [WCS Camera Traps](http://lila.science/datasets/wcscameratraps) dataset, as well as images of humans in both daytime and nighttime. We also have a preliminary "vehicle" class for cars, trucks, and bicycles.

#### Download links

* [Frozen model (.pb)](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb)
* [TFODAPI config file](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.config)
* [Last checkpoint (for resuming training)](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_checkpoint.zip)
* [TensorFlow SavedModel for TFServing](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_saved_model.zip) (inputs in uint8 format, `serving_default` output signature)

If you're not sure which format to use, you want the "frozen model" file (the first link).


### MegaDetector v3, 2019.05.30

#### Release notes

In addition to incorporating additional data, this release adds a preliminary "human" class.  Our animal training data is still far more comprehensive than our humans-in-camera-traps data, so if you're interested in using our detector but find that it works better on animals than people, stay tuned.

#### Download links

- [Frozen model (.pb)](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/megadetector_v3.pb)
- [TFODAPI config file](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/megadetector_v3.config)
- [Last checkpoint (for resuming training)](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/megadetector_v3_checkpoint.zip)
- [TensorFlow SavedModel](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/saved_model_normalized_megadetector_v3_tf19.tar.gz) (inputs in TF [common image format](https://www.tensorflow.org/hub/common_signatures/images#image_input), `default` output signature)
- [TensorFlow SavedModel for TFServing](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip) (inputs in uint8 format, `serving_default` output signature)


### MegaDetector v2, 2018

#### Release notes

First MegaDetector release!  Yes, that's right, v2 was the first release.  If there was a v1, we don't remember it.

#### Download links

- [Frozen model (.pb)](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/megadetector_v2.pb)
- [TFODAPI config file](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/megadetector_v2.config)
- [Last checkpoint (for resuming training)](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/megadetector_v2_checkpoint.zip)


## Using the model

### If you already know about how Python environments work...

You may want to skip the rest of this page and use the [MegaDetector Python package](https://pypi.org/project/megadetector/) (pip install megadetector).  There are examples on the package home page, and the package is documented [here](https://megadetector.readthedocs.io/).

If you are new to Python, you are in the right place, read on...

### Overview

We provide two ways to run MegaDetector on your images:

1. A simple test script that makes neat pictures with bounding boxes, but doesn't produce a useful output file ([run_detector.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector.py))

2. A script for running large batches of images on a local GPU ([run_detector_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector_batch.py))

Also see the <a href="#is-there-a-gui">&ldquo;Is there a GUI?&rdquo;</a> section for graphical options and other ways of running MD, including real-time APIs, Docker environments, and other goodies.

The remainder of this section provides instructions for running our "official" scripts, including installing all the necessary Python dependencies.

### 1. Install prerequisites: Miniforge, Git, and NVIDIA stuff

#### Install Miniforge

All of the instructions that follow assume you have installed [Miniforge](https://github.com/conda-forge/miniforge).  Miniforge is an environment for installing and running Python stuff.

If you know what you're doing, or you already have Anaconda installed, you can use either Anaconda or Miniforge; the environment files work with both.  But our experiences have been best with Miniforge, so, if you just want to get up and running, start by installing Miniforge.  If you're using Anaconda and you're staring at a "solving environment" prompt that's been running for like a day, consider switching to Miniforge.


##### Install Miniforge on Windows

To install Miniforge on Windows, just download and run the [Miniforge installer](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe).  If you get a "Windows protected your PC" warning, you might have to click "More info" and "run anyway". You can leave everything at the default value during installation.

All the instructions below will assume you are running at the Miniforge command prompt, which is basically just like a regular command prompt, but it has access to all the Python stuff.  On Windows, once you've installed Miniforge, you can start your Miniforge command prompt by launching the shortcut called "Miniforge prompt".

You will know you are at a Miniforge prompt (as opposed to run-of-the-mill command prompt) if you see an environment name in parentheses before your current directory, like this:

<img src="images/anaconda-prompt-base.jpg" style="margin-left:25px;">

...or this:

<img src="images/anaconda-prompt-ct.jpg" style="margin-left:25px;">


##### Install Miniforge on Linux/Mac

The [list of Miniforge installers](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3) has links for Linux and OSX.  If you're installing on a Mac, be sure to download the right installer: "x86_64" if you are on an Intel Mac, "arm64 (Apple Silicon)" if you are on an M1/M2 Mac with Apple silicon.  In all of these cases, you will be downloading a .sh file; after you run it to install Miniforge, you should see an environment name in parentheses just like in the images above.


#### Install git (if you're on Windows, otherwise you probably already have it)

The instructions will also assume you have git installed.  If you're not familiar with git, and you are on a Windows machine, we recommend installing [Git for Windows](https://git-scm.com/download/win).  If you're on a Linux machine or a Mac, there's like a 99.9% chance you already have git installed.


#### Install Nvidia stuff (if you have an Nvidia GPU, otherwise you don't need it)

If you have a deep-learning-friendly GPU, you will also need to have a recent [NVIDIA driver](https://www.nvidia.com/download/index.aspx) installed.  If you don't have an Nvidia GPU, it's OK,  you can still run MegaDetector on your CPU, and you don't need to install any special drivers.


### 2. Optional: download the MegaDetector model(s)

<b>This step is optional; in fact, the only reason to run this step is if you will not have an Internet connection later when you need to run MegaDetector</b>.

Otherwise, when you run MegaDetector later in these instructions, the model file will get downloaded automatically.

That said, if you want to save MegaDetector to a particular folder, download one or more MegaDetector model files (typically [MDv5a](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt), but you can also download [MDv5b](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt) and/or [MDv4](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb)) to your computer.  You can put them anywhere, later in these instructions you'll tell the relevant scripts where to find the model file.


### 3. Clone the relevant git repos and add them to your path, and set up your Python environment

You will need the contents of two git repos to make everything work: this repo and the [YOLOv5](https://github.com/ultralytics/yolov5) repo (more specifically, a fork of that repo).  You will also need to set up a Python environment with all the Python packages that our code depends on.  In this section, we provide <a href="#windows-instructions-for-gitpython-stuff">Windows</a>, <a href="#linux-instructions-for-gitpython-stuff">Linux</a>, and <a href="#mac-instructions-for-gitpython-stuff">Mac</a> instructions for doing all of this stuff.


#### Windows instructions for git/Python stuff

The first time you set all of this up, open your Miniforge prompt, and run:

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/agentmorris/MegaDetector
git clone https://github.com/ecologize/yolov5
cd c:\git\MegaDetector
mamba env create --file envs\environment-detector.yml
mamba activate megadetector
set PYTHONPATH=c:\git\MegaDetector;c:\git\yolov5
```

<a name="windows-new-shell"></a>
Your environment is set up now!  In the future, when you open your Miniforge prompt, you only need to run:

```batch
cd c:\git\MegaDetector
mamba activate megadetector
set PYTHONPATH=c:\git\MegaDetector;c:\git\yolov5
```

Pro tip: if you have administrative access to your machine, rather than using the "set PYTHONPATH" steps, you can also create a permanent PYTHONPATH environment variable.  Here's a [good page](https://www.computerhope.com/issues/ch000549.htm) about editing environment variables in Windows.  But if you just want to "stick to the script" and do it exactly the way we recommend above, that's fine.


#### Linux instructions for git/Python stuff

If you have installed Miniforge on Linux, you are probably always at an Miniforge prompt; i.e., you should see "(base)" at your command prompt.  Assuming you see that, the first time you set all of this up, and run:

```batch
mkdir ~/git
cd ~/git
git clone https://github.com/ecologize/yolov5
git clone https://github.com/agentmorris/MegaDetector
cd ~/git/MegaDetector
mamba env create --file envs/environment-detector.yml
mamba activate megadetector
export PYTHONPATH="$HOME/git/MegaDetector:$HOME/git/yolov5"
```

If you want to use MDv4 (which you probably don't, unless you have a really good reason to), there's one extra setup step (this will not break your MDv5 setup, you can run both in the same environment):

```batch
mamba activate megadetector
pip install tensorflow
```

<a name="linux-new-shell"></a>
Your environment is set up now!  In the future, whenever you start a new shell, you just need to do:

```batch
cd ~/git/MegaDetector
mamba activate megadetector
export PYTHONPATH="$HOME/git/MegaDetector:$HOME/git/yolov5"
```

Pro tip: rather than updating your PYTHONPATH every time you start a new shell, you can add the "export" line to your .bashrc file.


#### Mac instructions for git/Python stuff

If you have installed Miniforge on Mac, you are probably always at an Miniforge prompt; i.e., you should see "(base)" at your command prompt.  Assuming you see that, the first time you set all of this up, and run:

```batch
mkdir ~/git
cd ~/git
git clone https://github.com/ecologize/yolov5
git clone https://github.com/agentmorris/MegaDetector
cd ~/git/MegaDetector
./envs/md-mac-env-setup.sh
mamba activate megadetector
export PYTHONPATH="$HOME/git/MegaDetector:$HOME/git/yolov5"
```

If you want to use MDv4 (which you probably don't, unless you have a really good reason to), there's one extra setup step (this will not break your MDv5 setup, you can run both in the same environment):

```batch
mamba activate megadetector
pip install tensorflow
```

<a name="linux-new-shell"></a>
Your environment is set up now!  In the future, whenever you start a new shell, you just need to do:

```batch
cd ~/git/MegaDetector
mamba activate megadetector
export PYTHONPATH="$HOME/git/MegaDetector:$HOME/git/yolov5"
```

Pro tip: rather than updating your PYTHONPATH every time you start a new shell, you can add the "export" line to your .bashrc file.


### 4. Hooray, we finally get to run MegaDetector!

#### run_detector.py

To test MegaDetector out on small sets of images and get super-satisfying visual output, we provide [run_detector.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector.py), an example script for invoking this detector on new images.  This isn't how we recommend running lots of images through MegaDetector (see [run_detector_batch.py](#2-run_detector_batchpy) below for "real" usage), but it's a quick way to test things out.  [Let us know](mailto:cameratraps@lila.science) how it works on your images!

The following examples assume you have your Miniforge prompt open, and have put things in the same directories we put things in the above instructions.  If you put things in different places, adjust these examples to match your folders, and most importantly, adjust these examples to point to your images.

To use run_detector.py on Windows, when you open a new Miniforge prompt, don't forget to do this:

```batch
cd c:\git\MegaDetector
mamba activate megadetector
set PYTHONPATH=c:\git\MegaDetector;c:\git\yolov5
```

Then you can run the script like this:

```batch
python megadetector\detection\run_detector.py MDV5A --image_file "some_image_file.jpg" --threshold 0.1
```

"MDV5A" tells this script to automatically download MegaDetector v5a; if you already downloaded it, you can replace this with the full path to your MegaDetector model file (e.g. "c:\megadetector\md_v5a.0.0.pt").

Change "some_image_file.jpg" to point to a real image on your computer.

If you ran this script on "some_image_file.jpg", it will produce a file called "some_image_file_detections.jpg", which - if everything worked right - has boxes on objects of interest.

If you have a GPU, and it's being utilized correctly, near the beginning of the output, you should see:

`GPU available: True`

If you have an Nvidia GPU, and it's being utilized correctly, near the beginning of the output, you should see:

`GPU available: True`

If you have an Nvidia GPU and you see "GPU available: False", your GPU environment may not be set up correctly.  95% of the time, this is fixed by <a href="https://www.nvidia.com/en-us/geforce/drivers/">updating your Nvidia driver"</a> and rebooting.  If you have an Nvidia GPU, and you've installed the latest driver, and you've rebooted, and you're still seeing "GPU available: False", <a href="mailto:cameratraps@lila.science">email us</a>.


<b>This is really just a test script, you will mostly only use this to make sure your environment is set up correctly</b>.  run_detector_batch.py (see <a href="#run_detector_batchpy">below</a>) is where the interesting stuff happens.

You can see all the options for this script by running:

```batch
python megadetector\detection\run_detector.py
```

To use this script on Linux/Mac, when you open a new Miniforge prompt, don't forget to do this:
 
```batch
cd ~/git/MegaDetector
mamba activate megadetector
export PYTHONPATH="$HOME/git/MegaDetector:$HOME/git/yolov5"
```

Then you can run the script like this:

```batch
python megadetector/detection/run_detector.py MDV5A --image_file "some_image_file.jpg" --threshold 0.1
```

Don't forget to change "some_image_file.jpg" to point to a real image on your computer.

#### run_detector_batch.py

To apply this model to larger image sets on a single machine, we recommend a different script, [run_detector_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector_batch.py).  This outputs data in the [MegaDetector results format](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing#megadetector-batch-output-format), so you can work with the results in tools like [Timelapse](https://saul.cpsc.ucalgary.ca/timelapse/).

To use run_detector_batch.py on Windows, when you open a new Miniforge prompt, don't forget to do this:

```batch
cd c:\git\MegaDetector
mamba activate megadetector
set PYTHONPATH=c:\git\MegaDetector;c:\git\yolov5
```

Then you can run the script like this:

```batch
python megadetector\detection\run_detector_batch.py MDV5A "c:\some_image_folder" "c:\megadetector\test_output.json" --output_relative_filenames --recursive --checkpoint_frequency 10000 --quiet
```

"MDV5A" tells this script to automatically download MegaDetector v5a; if you already downloaded it, you can replace this with the full path to your MegaDetector model file (e.g. "c:\megadetector\md_v5a.0.0.pt").

Change "c:\some_image_folder" to point to the real folder on your computer where your images live.

This will produce a file called "c:\megadetector\test_output.json", which - if everything worked right - contains information about where objects of interest are in your images.  You can use that file with any of our [postprocessing](megadetector/postprocessing) scripts, but most users will read this file into [Timelapse](https://saul.cpsc.ucalgary.ca/timelapse/).

You can see all the options for this script by running:

```batch
python megadetector\detection\run_detector_batch.py
```

#### Saving and restoring run_detector_batch.py checkpoints

If you are running very large batches, we strongly recommend adding the `--checkpoint_frequency` option to save checkpoints every N images (you don't want to lose all the work your PC has done if your computer crashes!).  10000 is a good value for checkpoint frequency; that will save the results every 10000 images.  This is what we've used in the example above.  When you include this option, you'll see a line in the output that looks like this:

`The checkpoint file will be written to c:\megadetector\md_checkpoint_20230305232323.json`

The default checkpoint file will be in the same folder as your output file; in this case, because we told the script to write the final output to c:\megadetector\test_output.json, the checkpoint will be written in the c:\megadetector folder.  If everything goes smoothly, the checkpoint file will be deleted when the script finishes.  If your computer crashes/reboots/etc. while the script is running, you can pick up where you left off by running exactly the same command you ran the first time, but adding the "--resume_from_checkpoint" option, with the checkpoint file you want to resume from, or you can just say "auto" to use the most recent checkpoint.  So, in this case, you would run:

```batch
python megadetector\detection\run_detector_batch.py MDV5A "c:\some_image_folder" "c:\megadetector\test_output.json" --output_relative_filenames --recursive --checkpoint_frequency 10000 --quiet --resume_from_checkpoint auto
```

You will see something like this at the beginning of the output:

`Restored 80000 entries from the checkpoint`

If the extremely unlikely event that your computer happens to crash *while* a checkpoint is getting written... don't worry, you're still safe, but it's a bit outside the scope of this tutorial, so just <a href="mailto:cameratraps@lila.science">email us</a> in that case.


#### If your GPU isn't recognized by run_detector_batch.py

If you have an Nvidia GPU, and it's being utilized correctly, near the beginning of the output, you should see:

`GPU available: True`

If you have an Nvidia GPU and you see "GPU available: False", your GPU environment may not be set up correctly.  95% of the time, this is fixed by <a href="https://www.nvidia.com/en-us/geforce/drivers/">updating your Nvidia driver</a> and rebooting.  If you have an Nvidia GPU, and you've installed the latest driver, and you've rebooted, and you're still seeing "GPU available: False", <a href="mailto:cameratraps@lila.science">email us</a>.

#### Slightly modified run_detector_batch.py instructions for Linux/Mac

To use this script on Linux/Mac, when you open a new Miniforge prompt, don't forget to do this:
 
```batch
cd ~/git/MegaDetector
mamba activate megadetector
export PYTHONPATH="$HOME/git/MegaDetector:$HOME/git/yolov5"
```

Then you can run the script like this:

```batch
python megadetector/detection/run_detector_batch.py MDV5A "/some/image/folder" "$HOME/megadetector/test_output.json" --output_relative_filenames --recursive --checkpoint_frequency 10000
```


## OK, but is that how the MD devs run the model?

Almost... we run a lot of MegaDetector on a lot of images, and in addition to running the main "run_detector_batch" script described in the previous section, running a large batch of images also usually includes:

* Dividing images into chunks for running on multiple GPUs
* Making sure that the number of failed/corrupted images was reasonable
* Eliminating frequent false detections using the [repeat detection elimination](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination) process
* Visualizing the results using [postprocess_batch_results.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py) to make "results preview" pages like [this one](https://lila.science/public/snapshot_safari_public/snapshot-safari-kar-2022-00-00-v5a.0.0_0.200/)

...and, less frequently:

* Running a species classifier on the MD crops
* Moving images into folders based on MD output
* Various manipulation of the output files, e.g. splitting .json files into smaller .json files for subfolders
* Running and comparing multiple versions of MegaDetector

There are separate scripts to do all of these things, but things would get chaotic if we ran each of these steps separately.  So in practice we almost always run MegaDetector using [manage_local_batch.py](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.py), a script broken into cells for each of those steps.  We run this in an interactive console in [Spyder](https://github.com/spyder-ide/spyder), but we also periodically export this script to a [notebook](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.ipynb) that does exactly the same thing.

So, if you find yourself keeping track of lots of steps like this to manage large MD jobs, try the notebook out!  And let us know if it's useful/broken/wonderful/terrible.

 
## Is there a GUI?

Many of our users either use our Python tools to run MegaDetector or have us run MegaDetector for them (see [this page](getting-started.md) for more information about that), then most of those users use [Timelapse](https://saul.cpsc.ucalgary.ca/timelapse/) to use their MegaDetector results in an image review workflow.

But we recognize that Python tools can be a bit daunting, so we're excited that a variety of tools allow you to run MegaDetector in a GUI have emerged from the community.  We're interested in users' perspectives on all of these tools, so if you find them useful - or if you know of others - [let us know](mailto:cameratraps@lila.science), and thank those developers!

### GUI tools for running MegaDetector locally

* [AddaxAI](https://addaxdatascience.com/addaxai/) is a GUI-based tool for running MegaDetector and a variety of species classifiers; also supports a number of postprocessing functions (e.g. folder separation)
* [CamTrap Detector](https://github.com/bencevans/camtrap-detector) is a GUI-based tool for running MegaDetector (supports MDv5)
* [MegaDetector-GUI](https://github.com/petargyurov/megadetector-gui) is a GUI-based tool for running MegaDetector in Windows environments (MDv4 only as far as we know)

 
### Interactive demos/APIs

* [Hendry Lydecker](https://github.com/hlydecker) set up a [Hugging Face app](https://huggingface.co/spaces/hlydecker/MegaDetector_v5) for running MDv5
* [Ben Evans](https://bencevans.io/) set up a [Web-based MegaDetector demo](https://replicate.com/bencevans/megadetector) at [replicate.com](https://replicate.com)


### Thick-client tools that leverage MegaDetector

* [DeepFaune](https://www.deepfaune.cnrs.fr/en/)


### Cloud-based platforms that leverage MegaDetector

It's not quite as simple as "these platforms all run MegaDetector on your images", but to varying degrees, all of the following online platforms use MegaDetector:

* [Wildlife Insights](https://wildlifeinsights.org/)
* [TrapTagger](https://wildeyeconservation.org/trap-tagger-about/)
* [WildTrax](https://www.wildtrax.ca/)
* [Agouti](https://agouti.eu/)
* [Trapper](https://trapper-project.readthedocs.io/en/latest/overview.html)
* [Camelot](https://camelotproject.org/)
* [WildePod](https://wildepod.org/)
* [wpsWatch](https://wildlifeprotectionsolutions.org/wpswatch/)
* [TNC Animl](https://animl.camera/) ([code](https://github.com/tnc-ca-geo/animl-frontend))
* [Cam-WON](https://wildlifeobserver.net/)
* [Zooniverse ML Subject Assistant](https://subject-assistant.zooniverse.org/#/intro)
* [Dudek AI Image Toolkit](https://ait.dudek.com)
* [Zamba Cloud](https://github.com/drivendataorg/zamba)
* [OCAPI](https://www.terroiko.fr/ocapi/)


### Other ways of running MegaDetector that don't fit easily into one of those categories

#### Third-party repos that use MegaDetector

<!-- Sync'd with the last chunk of the list of repos on the camera trap ML survey -->
* Mega-Efficient Wildlife Classifier (MEWC) (tools for training classifiers on MD crops) ([github.com/zaandahl/mewc](https://github.com/zaandahl/mewc))
* MegaDetectorLite (ONNX/TensorRT conversions for MD) ([github.com/timmh/MegaDetectorLite](https://github.com/timmh/MegaDetectorLite))
* MegaDetector-FastAPI (MD serving via FastAPI/Streamlit) ([github.com/abhayolo/megadetector-fastapi](https://github.com/abhayolo/megadetector-fastapi))
* MegaDetector UI (tools for server-side invocation of MegaDetector) ([github.com/NINAnor/megadetector_ui](https://github.com/NINAnor/megadetector_ui)
* MegaDetector Container (Docker image for running MD) ([github.com/bencevans/megadetector-contained](https://github.com/bencevans/megadetector-contained))
* MegaDetector V5 - ONNX (tools for exporting MDv5 to ONNX) ([github.com/parlaynu/megadetector-v5-onnx](https://github.com/parlaynu/megadetector-v5-onnx))
* MEWC (Mega Efficient Wildlife Classifier) ([github.com/zaandahl/mewc](https://github.com/zaandahl/mewc))
* CamTrapML (Python library for camera trap ML) ([github.com/bencevans/camtrapml](https://github.com/bencevans/camtrapml))
* WildCo-Faceblur (MD-based human blurring tool for camera traps) ([github.com/WildCoLab/WildCo_Face_Blur](https://github.com/WildCoLab/WildCo_Face_Blur))
* CamTrap Detector (MDv5 GUI) ([github.com/bencevans/camtrap-detector](https://github.com/bencevans/camtrap-detector))
* SDZG Animl (package for running MD and other models via R) ([github.com/conservationtechlab/animl](https://github.com/conservationtechlab/animl))
* SpSeg (WII Species Segregator) ([github.com/bhlab/SpSeg](https://github.com/bhlab/SpSeg))
* Wildlife ML (detector/classifier training with active learning) ([github.com/slds-lmu/wildlife-ml](https://github.com/slds-lmu/wildlife-ml))
* BayDetect (GUI and automation pipeline for running MD) ([github.com/enguy-hub/BayDetect](https://github.com/enguy-hub/BayDetect))
* Automated Camera Trapping Identification and Organization Network (ACTION) ([github.com/humphrem/action](https://github.com/humphrem/action))
* TigerVid (animal frame/clip extraction from videos) ([github.com/sheneman/tigervid](https://github.com/sheneman/tigervid))
* Trapper AI (AI backend for the TRAPPER platform) ([gitlab.com/trapper-project/trapper-ai](https://gitlab.com/trapper-project/trapper-ai))
* video-processor (MD workflow for security camera footage) ([github.com/evz/video-processor](https://github.com/evz/video-processor))
* Declas (client-side tool for running MD and classifiers) ([github.com/stangandaho/declas](https://github.com/stangandaho/declas))
* AI for Wildlife Monitoring (real-time alerts using 4G camera traps) ([github.com/ratsakatika/camera-traps])(https://github.com/ratsakatika/camera-traps/)

#### Third-party things that aren't repos

* [Kaggle notebook](https://www.kaggle.com/code/evmans/train-megadetector-tutorial) for fine-tuning MegaDetector to add additional classes

#### Maintained within this repo

* [Colab notebook](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/megadetector_colab.ipynb) ([open in Colab](https://colab.research.google.com/github/agentmorris/MegaDetector/blob/main/detection/megadetector_colab.ipynb)) for running MDv5 on images stored in Google Drive.
* [Real-time MegaDetector API using Flask](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/synchronous).  This is deployed via Docker, so the Dockerfile provided for the real-time API may be a good starting point for other Docker-based MegaDetector deployments as well.
* [Batch processing API](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing) that runs images on many GPUs at once on Azure.  There is no public instance of this API, but the code allows you to stand up your own endpoint.
 

## How do I use the results?

See the ["How do people use MegaDetector results?"](https://github.com/agentmorris/MegaDetector/blob/main/getting-started.md#how-people-use-megadetector-results) section of our "getting started" page.


## Have you evaluated MegaDetector's accuracy?

Internally, we track metrics on a validation set when we train MegaDetector, but we can't stress enough how much performance of any AI system can vary in new environments, so if we told you "99.9999% accurate" or "50% accurate", etc., we would immediately follow that up with "but don't believe us: try it in your environment!"

Consequently, when we work with new users, we always start with a "test batch" to get a sense for how well MegaDetector works for <i>your</i> images.  We make this as quick and painless as possible, so that in the (hopefully rare) cases where MegaDetector will not help you, we find that out quickly.

All of those caveats aside, we are aware of some external validation studies... and we'll list them here... but still, try MegaDetector on your images before you assume any performance numbers!

These are not necessarily papers specifically about evaluating MegaDetector, but they at least include a standalone MD evaluation.

### MDv5 evaluations

* WildEye. [MegaDetector Version 5 evaluation](https://wildeyeconservation.org/megadetector-version-5/).
* Clarfeld LA, Sirén AP, Mulhall BM, Wilson TL, Bernier E, Farrell J, Lunde G, Hardy N, Gieder KD, Abrams R, Staats S. [Evaluating a tandem human-machine approach to labelling of wildlife in remote camera monitoring](https://www.sciencedirect.com/science/article/pii/S1574954123002868). Ecological Informatics. 2023 Aug 10:102257.
* Aguirre I, Hood GA, Westbrook CJ. [Short-term dynamics of beaver dam flow states](https://www.sciencedirect.com/science/article/pii/S0048969724009641). Science of The Total Environment. 2024 Feb 9:170825.

### MDv4 evaluations

* Mitterwallner V, Peters A, Edelhoff H, Mathes G, Nguyen H, Peters W, Heurich M, Steinbauer MJ. [Automated visitor and wildlife monitoring with camera traps and machine learning](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.367). Remote Sensing in Ecology and Conservation. 2023.
* Fennell M, Beirne C, Burton AC. [Use of object detection in camera trap image identification: assessing a method to rapidly and accurately classify human and animal detections for research and application in recreation ecology](https://www.sciencedirect.com/science/article/pii/S2351989422001068?via%3Dihub). Global Ecology and Conservation. 2022 Mar 25:e02104.
* Vélez J, Castiblanco-Camacho PJ, Tabak MA, Chalmers C, Fergus P, Fieberg J.  [Choosing an Appropriate Platform and Workflow for Processing Camera Trap Data using Artificial Intelligence](https://arxiv.org/abs/2202.02283). arXiv. 2022 Feb 4.
* [github.com/FFI-Vietnam/camtrap-tools](https://github.com/FFI-Vietnam/camtrap-tools) (includes an evaluation of MegaDetector)


Bonus... this paper is not a formal review, but includes a thorough case study around MegaDetector:

* Tuia D, Kellenberger B, Beery S, Costelloe BR, Zuffi S, Risse B, Mathis A, Mathis MW, van Langevelde F, Burghardt T, Kays R. [Perspectives in machine learning for wildlife conservation](https://www.nature.com/articles/s41467-022-27980-y). Nature Communications. 2022 Feb 9;13(1):1-5.

If you know of other validation studies that have been published, [let us know](mailto:cameratraps@lila.science)!

### One more reminder about evaluations on someone else's data

Really, don't trust results from one ecosystem and assume they will hold in another. [This paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html) is about just how catastrophically bad AI models for camera trap images <i>can</i> fail to generalize to new locations.  We hope that's not the case with MegaDetector!  But don't assume.


## What is MegaDetector bad at?

While MegaDetector works well in a variety of terrestrial ecosystems, it's not perfect, and we can't stress enough how important it is to test MegaDetector on your own data before trusting it.  We can help you do that; [email us](mailto:cameratraps@lila.science) if you have questions about how to evaluate MegaDetector on your own data, even if you don't have images you've already reviewed.

But really, we'll answer the question... MegaDetector v5's biggest challenges are with reptiles.  This is an area where accuracy has dramatically improved since MDv4, but it's still the case that reptiles are under-represented in camera trap data, and an AI model is only as good as its training data.  That doesn't mean MDv5 doesn't support reptiles; sometimes it does amazing on reptile-heavy datasets.  But sometimes it drives you bonkers by missing obvious reptiles.

If you want to read more about our favorite MD failure cases, check out the [MegaDetector challenges](megadetector-challenges.md) page.

tl;dr: always test on your own data!
 

## Pro tips for coaxing every bit of accuracy out of MegaDetector

As per the [training data](#can-you-share-the-training-data) section, MDv5 is actually two models (MDv5a and MDv5b), differing only in their training data.  In fact, MDv5a's training data is a superset of MDv5b's training data.  So, when should you use each?  What should you do if MegaDetector is working, but not <i>quite</i> well enough for a difficult scenario, like the ones on our [MegaDetector challenges](megadetector-challenges.md) page?  Or what if MegaDetector is working great, but you're a perfectionist who wants to push the envelope on precision?  This section is a very rough flowchart for how the MegaDetector developers choose MegaDetector versions/enhancements when presented with a new dataset.

1. The first thing we always run is MDv5a... <b>95% of the time, the flowchart stops here</b>.  That's in bold because we want to stress that this whole section is about the unusual case, not the typical case.  There are enough complicated things in life, don't make choosing MegaDetector versions more complicated than it needs to be.<br/></br>Though FWIW, we're not usually trying to squeeze every bit of precision out of a particular dataset, we're almost always focused on recall (i.e., not missing animals).  So if MDv5a is finding all the animals and the number of false positives is "fine", we don't usually run MDv5b, for example, just to see whether it would slightly further reduce the number of false positives.

2. If things are working great, but you're going to be using MegaDetector a lot and you want to add a step to your process that has a bit of a learning curve, but can eliminate a bunch of false positives once you get used to it, consider the [repeat detection elimination](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination) process.

3. If anything looks off, specifically if you're missing animals that you think MegaDetector should be getting, or if you just want to see if you can squeeze a little more precision out, try MDv5b.  Usually, we've found that 
MDv5a works at least as well as MDv5b, but every dataset is different.<br/><br/>For example, [WildEye](https://wildeyeconservation.org/) did a thorough [MegaDetector v5 evaluation](https://wildeyeconservation.org/megadetector-version-5/) and found slightly better precision with MDv5b.  MDv5a is trained on everything MDv5b was trained on, plus some non-camera-trap data, so as a general rule, MDv5a may do <i>slightly</i> better on reptiles, birds, and distant vehicles.  MDv5b may do <i>slightly</i> better on very dark or low-contrast images.

4. If you're still missing animals, but one or both models look close, try again using YOLOv5's [test-time augmentation tools](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) via this [alternative MegaDetector inference script](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_inference_with_yolov5_val.py), which produces output in the same format as the standard inference script, but uses YOLOv5's native inference tools.  It will run a little more slowly, and still lacks some of the bells and whistles of the standard inference script, but sometimes augmentation helps.

5. If something still looks off, try MDv4.

6. If none of the above are quite working well enough, but two or three of the above are close, try using [merge_detections.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/merge_detections.py) to get the best of both worlds, i.e. to take the high-confidence detections from multiple MegaDetector results files.

7. If things are still not good enough, we have a case where MD just seems not to work; that's what the [MegaDetector challenges](megadetector-challenges.md) page is all about.  Now we're in DIY territory.

And please please please, <b>if you find you need to do anything other than step 1 (simple MDv5a), please [let us know](mailto:cameratraps@lila.science)!</b>  It's really helpful for us to hear about cases where MegaDetector either doesn't work well or requires extra tinkering.


## Citing MegaDetector

If you use MegaDetector in a publication, please cite:

Beery S, Morris D, Yang S. Efficient pipeline for camera trap image review. arXiv preprint arXiv:1907.06772. 2019 Jul 15.

Please include the version of MegaDetector you used.  If you are including any analysis of false positives/negatives, please be sure to specify the confidence threshold you used as well.

The same citation, in BibTex format:

```BibTeX
@article{beery2019efficient,
  title={Efficient Pipeline for Camera Trap Image Review},
  author={Beery, Sara and Morris, Dan and Yang, Siyu},
  journal={arXiv preprint arXiv:1907.06772},
  year={2019}
}
```


## Tell me more about why detectors are a good first step for camera trap images

Can do!  See these [slides](http://dmorris.net/misc/cameratraps/ai4e_camera_traps_overview).


## Pretty picture

Here's a "teaser" image of what detector output looks like:

![alt text](images/detector_example.jpg "Red bounding box on fox")<br/>Image credit University of Washington.


## Mesmerizing video

Here's a neat [video](http://dmorris.net/video/detector_video.html) of MDv2 running in a variety of ecosystems, on locations unseen during training.

<a href="http://dmorris.net/video/detector_video.html">
<img width=600 src="http://dmorris.net/video/detector_video_thumbnail.png">
</a>

Image credit [eMammal](https://emammal.si.edu/).  Video created by [Sara Beery](https://beerys.github.io/).


## Can you share the training data?

This model is trained on bounding boxes from a variety of ecosystems, and many of the images we use in training can't be shared publicly.  But in addition to the private training data we use, we also use many of the bounding boxes available on lila.science:

<https://lila.science/category/camera-traps/>

Each version of MegaDetector uses all the training data from the previous version, plus a bunch of new stuff.  Specifically...

MegaDetector v2 was trained on... actually, we don't remember, that was before the dawn of time.

MegaDetector v3 was trained on private data, plus public data from:

* [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)
* [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti)
* [Idaho Camera Traps](https://lila.science/datasets/idaho-camera-traps/)

MegaDetector v4 was trained on all MDv3 training data, plus new private data, and new public data from:

* [WCS Camera Traps](https://lila.science/datasets/wcscameratraps)
* [NACTI (North American Camera Trap Images)](https://lila.science/datasets/nacti)
* [Island Conservation Camera Traps](https://lila.science/datasets/island-conservation-camera-traps)

MegaDetector v5b was trained on all MDv4 training data, plus new private data, and new public data from:

* [Orinoquía Camera Traps](https://lila.science/orinoquia-camera-traps/)
* [SWG Camera Traps](https://lila.science/datasets/swg-camera-traps)
* [ENA24](https://lila.science/datasets/ena24detection)
* [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps)
* [Several datasets from Snapshot Safari](https://lila.science/category/camera-traps/snapshot-safari/)

The total dataset for MDv5b (including train/val/test) was around ~2.3M boxes on ~2.7M images, all of which are camera trap images.

MegaDetector v5a was trained on all MDv5b training data, and new (non-camera-trap) public data from:

* The [iNaturalist Dataset 2017](https://github.com/visipedia/inat_comp/tree/master/2017)
* [COCO](https://cocodataset.org/#home)

So if MegaDetector performs really well on any of the above data sets, that's great, but it's a little bit cheating, because we haven't published the set of locations from those data sets that we use during training.


## What if I just want to run non-MD scripts from this repo?

If you want to run scripts from this repo, but you won't actually be running MegaDetector, you can install a lighter-weight version of the same environment by doing the following:

1. Install [Miniforge](https://github.com/conda-forge/miniforge), an environment for installing and running Python stuff.  If you already have Anaconda installed, you can use that instead.

2. Install git. If you're not familiar with git, we recommend installing git from git-scm ([Windows link](https://git-scm.com/download/win)) ([Mac link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)).

The remaining steps will assume you are running at a Miniforge prompt.  You will know you are at a Miniforge prompt (as opposed to run-of-the-mill command prompt) if you see an environment name in parentheses before your current directory, like this:

<img src="images/anaconda-prompt-base.jpg" style="margin-left:25px;">

...or this:

<img src="images/anaconda-prompt-ct.jpg" style="margin-left:25px;">

3. In your Miniforge prompt, run the following to create your environment (on Windows):

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/agentmorris/MegaDetector
cd c:\git\MegaDetector
mamba env create --file envs\environment.yml
mamba activate cameratraps
set PYTHONPATH=c:\git\MegaDetector
```

...or the following (on MacOS):

```batch
mkdir ~/git
cd ~/git
git clone https://github.com/agentmorris/MegaDetector
cd ~/git/MegaDetector
mamba env create --file envs/environment.yml
mamba activate cameratraps
export PYTHONPATH="$HOME/git/MegaDetector"
```

4. Whenever you want to start this environment again, run the following (on Windows):

```batch
cd c:\git\MegaDetector
mamba activate cameratraps
set PYTHONPATH=c:\git\MegaDetector
```

...or the following (on MacOS):

```batch
cd ~/git/MegaDetector
mamba activate cameratraps
export PYTHONPATH="$HOME/git/MegaDetector"
```

Also, the environment file we're referring to in this section ([envs/environment.yml](environment.yml), the one without all the MegaDetector stuff) doesn't get quite the same level of TLC that our MegaDetector environment does, so if anyone tries to run scripts that don't directly involve MegaDetector using this environment, and packages are missing, [let us know](mailto:cameratraps@lila.science).


## What if I want to use MD without all the baggage of your very specific package versions?

We've historically gone a little bonkers making sure that MegaDetector results are absolutely repeatable, so have been very wary of changing PyTorch/YOLOv5 versions, or even Pillow versions.  On top of that, various combinations of YOLOv5 and PyTorch versions were unable to load models trained with the specific versions that existed when MDv5 was created.  The result of this is that our recommended environment uses older versions of PyTorch (1.10) and YOLOv5.

But... all of those incompatibilities have worked themselves out with only minimal changes to MegaDetector-related code, so as of 2023.09, you can run MegaDetector in the newest versions of Python (3.11.5), PyTorch (2.0.1), and YOLOv5, without having to clone the YOLOv5 repo separately.  Results are <i>very slightly</i> different than they are in the recommended environment, typically around the third decimal place in both confidence values and box coordinates.  But if you are OK living on the cutting edge with us, you can now set up MegaDetector like this, using a requirements.txt file that doesn't pin any package versions:

### On Windows

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/agentmorris/MegaDetector
cd c:\git\MegaDetector
mamba create -n megadetector-pip python=3.11 pip -y
mamba activate megadetector-pip
pip install -r envs\requirements.txt
set PYTHONPATH=c:\git\MegaDetector
```

### On Linux

```batch
mkdir ~/git
cd ~git
git clone https://github.com/agentmorris/MegaDetector
cd ~/git/MegaDetector
mamba create -n megadetector-pip python=3.11 pip -y
mamba activate megadetector-pip
pip install -r envs/requirements.txt
export PYTHONPATH="$HOME/git/MegaDetector"
```

### Using the MD Python package

You can also install the MD Python package without cloning the repo at all, like so:

```batch
mamba create -n megadetector-pip python=3.11 pip -y
mamba activate megadetector-pip
pip install megadetector --upgrade
python -m megadetector.detection.run_detector_batch --help
```

This comes with the same caveats as above: this will not produce results that are literally identical to the training environment, so, YMMV.  If you use this route, make sure the MegaDetector and YOLOv5 folders are <i>not</i> on your PYTHONPATH.
