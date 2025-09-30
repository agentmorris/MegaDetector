# Notes about this file

This file documents open work items.  Each level-2 heading is a work item.  Every item should have, in the following order):

* A description (this is all text between the item name and the priority).  This can use arbitrary markdown.
* A priority designated as P[N], on a line by itself.  Priority ranges from 0 to 4, 0 being highest priority.
* An effort level designated as E[N].  Effort ranges from 0 to 4, 4 being the most effort
* At least one tag, indicated as !tag-name.  

Tags can be arbitrary strings, but the most common tags are !feature, !maintenance, !bug, !docs, and !admin.  !admin basically means "this involves a decision by the repo maintainer(s), it's not really a work item".

Special tags include:

!name[issue-name]: assigns a name to this issue so that other issues can refer to it
!also-see[issue-name]: link this issue to another issue

Any amount of whitespace is allowed between each item within an issue.

The section called "title" can contain a title for the page, otherwise it will default to "task viewer"

This file is viewable at:

https://dmorris.net/task-viewer/?file=https://raw.githubusercontent.com/agentmorris/MegaDetector/refs/heads/main/TODO.md


# Title

MegaDetector issue list


# Header

This page tracks work items related to [MegaDetector](https://github.com/agentmorris/MegaDetector).  If you're interested in trying your hand at any of these, create a new [issue](https://github.com/agentmorris/MegaDetector/issues) on the MegaDetector repo, or <a href="mailto:agentmorris+megadetector@gmail.com">email me</a>!

This is just a task list; once a task is in progress, it will be tracked via GitHub Issues.  GitHub Issues is also still the right place for users to raise issues or ask questions.  GitHub Issues is just not, IMHO, a very practical "TODO list".   

Priorities range from 0 (urgent) to 4 (likely will never get done).  Effort ranges from 0 (less than an hour while watching football) to 4 (mega-big).


# Issues

## Remove dependency on ultralytics NMS for ultralytics models

The [Ultralytics NMS function](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py) has some significant limitations, most notably it produces incorrect behavior when a non-configurable timeout is exceeded.  I.e., it just bails after ~2 seconds, which can lead to missed objects if random stuff happens on the machine.  Consequently, there is a custom NMS function in pytorch_detector that replaces this for MDv5, MDv1000-redwood, and MDv1000-cedar.  However, the ultralytics models (sorrel, larch) produce output in a different format that is not supported by the custom NMS function, so currently we fall back to the ultralytics NMS implementation for these models.

I got to a working nms() function that would support both import formats, but it still requires some cleanup.  The working version is in [archive/misc/pytorch_detector_universal_nms.py](https://github.com/agentmorris/MegaDetector/blob/main/archive/misc/pytorch_detector_universal_nms.py).  This needs to be manually merged into pytorch_detector.py, and tested.

Fix this, and remove the ultralytics NMS import.  Before removing this item, consider whether the remaining functions that are still imported from the ultralytics/YOLO libraries are worth it, or whether we can (finally) remove those imports.  This is the only significant utility function that is still imported.

P1

E1

!maintenance

## Handle legacy setup.py issues

Three dependencies - yolov9pip, pyqtree, and clipboard - give this warning during pip installation:<br/><br/>

DEPRECATION: Building 'yolov9pip' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'yolov9pip'. Discussion can be found at https://github.com/pypa/pip/issues/6334

P2

E2

!maintenance


## Support video in postprocessing

postprocess_batch_results currently has no support for video.  When run on a .json file that points to videos, extract frames in a sensible way to generate previews.

P2

E2

!feature


## Allow excluding blanks in postprocessing

Currently when I drive postprocessing via manage_local_batch, I use a somewhat hacky approach to skipping the rendering of blank images, which still leaves empty links to those images.  Instead, allow excluding blanks during process_batch_results.  See the "render_animals_only" variable in manage_local_batch.

P2

E1

!feature


## Add descriptions to all tqdm progress bars

Particularly when running manage_local_batch, there are a lot of progress bars that run without clear indications of what's happening at each step.  Add descriptions to all progress bars to address this.

P2

E0

!maintenance


## Better handling of long filenames in manage_local_batch

manage_local_batch can end up generating previews with very long filenames, which is not important under Linux, but can blow past the archaic filename length restriction that still exists by default in Windows.  Windows actually supports long filenames fine now, but applications - particularly browsers - clip filenames to the historical limit; typically this results in failed images when loading a the HTML output locally.

Address this by:

* Reducing filename length in manage_local_batch wherever possible
* Adding a secondary filename length check in the OSError exception handler in postprocessing, and use something shorter than a GUID in that case
* Catching this case in postprocessing and at least printing a warning

P3

E2

!feature


## Allow a single confidence threshold in compare_batch_results

compare_batch_results takes a dict of class --> threshold mappings for each file being compared.  Allow this to be a float (rather than a dict) that applies to all categories.  The dict already supports a "default" entry, so just take the float value and stick it in a dict as {'default':new_threshold}.

P4

E0

!feature


## Add classification support to compare_batch_results

compare_batch_results supports comparing detections, but not species classifications.  Add support for species classification results.

P3

E2

!feature


## Multi-GPU inference in run_detector_batch

run_detector_batch only supports single-GPU (or single-/multi-CPU) inference.  Add multi-GPU inference.  This is P3 because in practice, when using manage_local_batch to create and run jobs, multi-GPU inference is handled naturally by breaking the task up into multiple lists of images.

P3

E2

!feature


## Add time-based criteria to RDE

The [repeat detection elimination](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination) pipeline currently uses two main parameters to decide which detections are likely repeating false positives: (1) size and (2) number of repeats.  Time is also strongly indicative of false detections: an identical detection 100 times over a month is much more likely to be a false detection than an identical detection 100 times in an hour.

P2

E1

!feature


## Sequence support in postprocessing

Allow postprocess_batch_results to operate on sequences, rather than just images.  Sample based on sequences, do precision/recall analysis based on sequences, and render sequences in a sensible way on the output page.

P2

E2

!feature


## Add postprocessing parameters to output files
 
In [repeat detection elimination](https://github.com/agentmorris/MegaDetector/tree/main/api/megadetector/postprocessing/repeat_detection_elimination) and [sequence-based classification smoothing](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py), write the smoothing parameters into the output file.

P3

E1

!feature


## Postprocessing refactor

postprocess_batch_results.py is an absurd use of Pandas right now, and has an absurd level of duplication between the code paths (with/without ground truth, with/without classification results).  This could use a re-write from scratch.

P3

E2

!maintenance


## RDE refactor

repeat_detections_core is pretty messy, and it has some really bizarre properties right now, like the fact that when you run the main function a second time to apply a set of changes after the manual review step, it repeats all the folder-separation stuff it did the first time, which is brittle and silly.  This requires not quite a total re-write, but a significant cleanup.

P3

E2

!maintenance

 
## Incorrect results in some Mac environments

In certain Apple silicon environments, MD produces incorrect results.  This is not specific to MD, this is a bug in YOLOv5.  See [this issue](https://github.com/ultralytics/yolov5/issues/12654) and [this question](https://github.com/ultralytics/yolov5/issues/12645) on the YOLOv5 repo for details and status.  The work item here is to:

* Assess the scope of this bug (all Apple silicon?  Just M1?  Has it been naturally resolved in M1 environments via recent PyTorch releases?)
* Assess whether this impacts MDv1000 models
* Consider disabling accelerated inference on Apple silicon for impacted models

P0

E2

!bug


## R wrappers

A substantial number (most?) of our users prefer R, and we're forcing them to run a bunch of Python code.  It would be great to either wrap the inference process in R, or port the inference code to R.  IMO it's not urgent to do this for anything in the MD package other than the inference code.  It would likely be acceptable to provide an R wrapper that launches Python at the CLI; this simplifies the implementation quite a bit compared to porting and/or calling Python directly from R.

P2

E2

!integration


## Tiled inference exploration

Though the MD package has support for running inference on small patches and stitching the results together via run_tiled_inference.py, [SAHI](https://github.com/obss/sahi), does the same thing, only with far more thought.  It would be useful to do a thorough evaluation of SAHI applied to MD, to compare the results against run_tiled_inference, and assess the scenarios where tiled inference makes sense, particularly for small objects in high-resolution images (e.g. finding geckos, or faraway ungulates).

P2

E3

!exploratory
!integration


## Client-side RDE tool

The [repeat detection elimination](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination) pipeline currently requires stitching together a bunch of tools: python scripts, a 3P image viewer, the Windows explorer.  It would be nice to integrate this into a proper client-side tool.  This would also be a good opportunity to allow keeping just a couple of images from a repeat detection series; currently if you see one animal and 100 false positives in a detection group, you typically have to just keep the whole detection group and eat 100 false positives.

P2

E3

!frontend
!feature


## Docs page improvements

The [docs page](https://megadetector.readthedocs.io/en/latest/) is complete and up to date, but it could use a design review, updates to a more modern theme, and the addition of some more detailed information that is currently in the MegaDetector User's Guide.  This is vague, I know, but basically "take a close look at the docs page and make it nicer".  For my two cents, I like the styles used by [contextily](https://contextily.readthedocs.io/en/latest) and [pybowler](https://pybowler.io/docs/basics-usage).

Lots of the information from the [MDv1000 release notes](https://github.com/agentmorris/MegaDetector/blob/main/docs/release-notes/mdv1000-release.md) could be re-used for the docs page, especially from the [section about the Python package](https://github.com/agentmorris/MegaDetector/blob/main/docs/release-notes/mdv1000-release.md#formally-introducing-the-md-python-package).

P0

E2

!docs
!design


## Organize COCO and iNat boxes for future training

MDv5a used COCO and iNat boxes; MDv5b and MD1000 do not.  Overall performance on camera trap images is better without COCO and iNat data, but there are some scenarios where the inclusion of this data improves performance, and even more scenarios where it improves detection.  I would like to re-create the equivalent of MDv5b for the MDv1000 family, which requires curating the human/animal/vehicle subset of COCO, and the "animals that might plausibly appear in camera trap images" subset of the [iNat 2017 challenge dataset](https://www.inaturalist.org/projects/inat-2017-challenge-dataset).  The latter is somewhat involved; conceptually, it includes, e.g., mammals, but not whales, and maybe not bats (at least as they might appear in iNat data), and it includes reptiles, but not, e.g., tiny geckos.

P1

E2

!training


## Document the use of the MD Python package to run non-MD detectors

It's often useful to run generic YOLO models on camera trap images, e.g. to complement MD with more fine-grained vehicle or background object classification.  The MD Python package is a useful way to do this, if you want to, e.g., review the results in Timelapse, or combine them with MD/SpeciesNet results.  This does not require any new code, just clear documentation.

P2

E1

!docs


## Document fine-tuning 

I don't generally encourage fine-tuning MD (it's almost always more work than it's worth), but there is a time and place for it, and it would be useful to document (a) how to do this and (b) when it's useful.  This would include some discussion of how bounding boxes come to be, including how to derive them from MD results.  This can point to the [tegu detection](https://github.com/agentmorris/usgs-tegus) and [goanna detection](https://github.com/agentmorris/unsw-goannas) projects as examples (both are fine-tuned MD models).

P3

E2

!training
!docs


## Document labeling notebook/process/repo

I made extensive use of the [custom labelme repo](https://github.com/agentmorris/labelme) when preparing training data for MDv1000, and the process is likely useful for others preparing similar bounding box data for training their own detectors (including fine-tuning MD).  This just needs a bit more documentation of how it all fits together; the repo is a good start, but there is a bit more process in undocumented notebooks around how I break datasets into labeling tasks, etc.  This would be useful for myself as well.

P2

E2

!training
!docs


## Checkpointing for video folder processing

run_detector_batch supports checkpointing, so crashes/reboots/etc. won't cause data loss during long inference jobs. process_video does not yet have this kind of checkpointing functionality.  This is not a huge deal, since in practice you would break large tasks into multiple calls to process_video(), but it would be nice to simplify this, even for large jobs.

P1

E2

!feature


## Performance evaluation for batching, preprocessing

run_detector_batch supports batched inference (for GPUs) and adjustable levels of worker-side preprocessing.  I have not formally evaluated the performance benefit (in terms of time, not accuracy) of using batched inference for various models on various GPUs.  This could use documentation of best practices.

P2

E2

!docs
!testing


## Formal evaluation of image size, augmentation, preprocessing mode

There are several knobs the user can turn when trying to squeeze the most accuracy out of a specific MD model: specifically, the user can tinker with the inference image size, enabling image augmentation, and switching between "classic" and "modern" preprocessing.  I have not yet formally evaluated the performance impacts of these approaches, so this could use formal evaluation and documentation of best practices.

P1

E2

!docs
!testing
!requires-data


## Add quality parameters to visualize_video_output

visualize_video_output renders videos with MD/classifier results.  Currently there is no support for controlling the size or quality of output videos.

P3

E1

!feature


## Reference result updates

* Reduce complexity of reference results: MD's test harness relies on .json files with pre-generated results for MDv5a and MDv5b, for a reference set of images and videos.  Because output varies slightly between PyTorch versions and between hardware environments, I have a number of results files.  This has gotten too complicated; remove most of the results files and increase the allowed tolerance during testing.
* Add test results for MD1000 models: MD's test harness only has results for MDv5, so it tests the not-crashing-ness of the other models, but it does not test correctness.  Add test results for other MD1000 models.
* Vehicle images: none of the test images include vehicles; add vehicle images to testing, including human/vehicle and animal/vehicle images
* Images with lat/lon information in EXIF metadata; make sure EXIF extraction (especially GPS location) is working correctly.

P0

E2

!testing


## Test coverage improvements

This is a placeholder for generally evaluating md_tests and the pytest harness, and deciding which scripts need additional testing.  Effort is highly variable; for example, adding tests for run_speciesnet_and_md is important and very easy.  Adding tests for postprocess_batch_results that actually verify correctness is a pain.  This work item almost certainly starts with asking AI what modules are not covered (or poorly covered) by tests.

P1

E2

!testing


## Add checkpointing to run_speciesnet_and_md

run_speciesnet_and_md does not currently have the same checkpointing support that run_detector_batch has.  The core functionality is there for the detection step, because it's built in to run_detector_batch, but this needs to be exposed to the CLI.  Equivalent functionality needs to be added for the classification step.

P0

E1

!feature


## Add classification smoothing to run_speciesnet_and_md

run_speciesnet_and_md does not currently incorporate sequence-/image-level classification smoothing.  Add this.

P0

E1

!feature
!speciesnet
!name[run_speciesnet_and_md-smoothing]


## Output format refinements

There are a few things I'd like to do to clean up the output format:

* Require that detections get sorted in descending order by confidence
* Make a decision on whether "detections" should be absent or null when "failure" is present
* Make a decision on whether "failure" should be absent or null when detections are present

This requires not just updating the output format and rev'ing the version number, but:

* Actually doing this throughout the repo
* Updating the files in the test dataset
* Updating the format validator
 
P2

E3

!maintenance


## pkg_resources deprecation

YOLO5 depends on the [pkg_resources](https://setuptools.pypa.io/en/latest/pkg_resources.html) module, which is slated for deprecation in November 2025.  This results in the following warning when running MD and/or SpeciesNet:

`/.../site-packages/yolov5/utils/general.py:34: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.`

It's not clear what exactly will happen when this is deprecated.  The action item here is to assess that, and do something about it.

P0

E2

!maintenance


## Improve memory efficiency for multicore inference

Right now multi-CPU inference in run_detector_batch loads separate detector instances on each worker.  It's not clear that it's necessary to load the model weights at all, or at least not to *keep* the model weights after initially loading them.  It's also not clear whether a single instance could be shared across all workers.

P2

E2

!optimization


## Re-evaluate "modern" postprocessing

At the time I added the "modern" postprocessing approach, it super-duper agreed with yolov5’s val.py; add a test back to make sure this is still the case.

P2

E2

!testing


## Support batch inference for video

run_detector_batch supports batch inference (for GPUs); process_video does not.  It's OK if we support batching within a video, but not across videos if that significantly simplifies implementation.

P2

E1

!feature


## Why does "fusing layers" print twice?

YOLOv5 tells me that it's "fusing layers" twice during startup.  A little part of me is concerned that somehow I'm loading the model twice, even when only a single worker is operating.  Assess this.

P0

E1

!bug


## RDE might remove custom fields within a detection object

The RDE process loads detections into a pandas dataframe, then re-generates a new list of detections.  There's no "official" scenario where detections might have custom properties, but I think this will result in the loss of custom properties.  Assess this, then either fix it, remove this item (if this doesn't really happen), or update the effort/priority of this task.

P2

E2

!bug


## More careful stride handling

Currently pytorch_detector uses a stride size of 64 for all 1280px models (which specifically means YOLOv5x6), and a stride size of 32 for all other models.  This is true for all MD models that exist right now, but if we, for example, train YOLOv9 @ 1280px, or train a YOLOv5?6 model (where ? != "x") this heuristic would fail.

P3

E2

!bug


## Remove non-batch inference code

Currently run_detector_batch has a somewhat separate code path for batch/non-batch inference.  This is just me being conservative: we should be able to treat non-batch inference as batch inference with a batch size of 1 to reduce complexity.  I just want to let things percolate for a bit before I do this.

P3

E0

!maintenance
!testing


## Add mdv2/3/4 to an “archive” release

The links for MDv2, MDv3, and MDv4 currently point to LILA.  I would prefer they point to GitHub.  Add an "archive" release that includes those binaries, and redirect URLs in code and in the MD User Guide.

P3

E0

!maintenance


## Benchmark timing results for MDv1000 models other than MDv1000-redwood

The MegaDetector User Guide includes benchmark timing results for MDv5, MDv1000-redwood, and MDv4.  Add benchmark timing results for other MDv1000 models.

P2

E0

!docs

## Graceful handling of augmentation for MDv1000-cedar

MDv1000-cedar does not support image augmentation.  It's not important to add it; the entire point of MDv1000-cedar is for compute-constrained scenarios.  Right now, though, it crashes unhelpfully.  Handle this more gracefully.

P2

E0

!bug


## Document the "modern" preprocessing approach

With the release of MDv1000, I introduced two preprocessing approaches: "classic" (which matches what we always did for MDv5, and roughly matches YOLOv5's detect.py) and "modern" (which roughly matches YOLOv5's val.py).  Neither is obviously better or worse, but they are different.  There is a separate item for evaluating this difference (and others); this item is just about more properly documenting the difference, and documenting how to invoke each approach.  This would include finding a couple of example images where they produce different results.

P1

E1

!docs


## Add category pages to visualize_db

postprocess_batch_results, which we use to visualize detector/classifier output, can break the results down by category in the generated HTML page.  It would be useful to add this to visualize_db, which we use to visualize COCO-formatted databases.

P3

E1

!feature


## Add hash values to .json output

Add md5 values for each image/video processed with run_detector_batch or process_video.  This is useful for the case where a user reorganizes or renames their images after running MD.

P3

E0

!feature


## Revive synchronous (real-time) API

The Flask-based API for serving MD was retired to the archive folder a few months ago for lack of maintenance, but it turns out that some folks were using it.  The short version of this work item is to bring it out of the archive folder and make sure it doesn't mess up linting and testing.  The stretch goal is to take a look at it and update it for how one would do this in 2025, and to make a nice demo out of it.  I'm assigning an effort level just for the short-term goal.

Whenever we tackle this, also bump gunicorn to >= 23 to address dependabot issues.

P3

E0

!maintenance


## Handle warnings about project.license at build time

Building the megadetector and megadetector-utils packages yield the warning described in [this setuptools issue](https://github.com/pypa/setuptools/issues/4903).  Fix that.

P1

E0

!maintenance


## Remove complex MKL requirements

The dependencies currently specify an old version of MKL (2024.0) for all non-Darwin platforms, because of an incompatibility between some versions of MKL and some versions of PyTorch, described in [this PyTorch issue](https://github.com/pytorch/pytorch/issues/123097).  We can remove this quirky dependency if we require PyTorch >= 2.5, which at some point becomes a good idea anyway, supporting ancient versions of PyTorch complicates testing.  The action item here is mostly to think through the implications of requiring PyTorch >= 2.5.

P3

E1

!maintenance
!admin

## Update Colab

Nothing is "wrong" with the [MegaDetector Colab](https://github.com/agentmorris/MegaDetector/blob/main/notebooks/megadetector_colab.ipynb), but it hasn't been updated in a while.  It doesn't mention MDv1000 or SpeciesNet; it would be helpful to just give the Colab a once-over, make sure it's still in good shape, and add optional cells that demonstrate MDv1000 use and SpeciesNet inference (via run_md_and_speciesnet).

P1

E2

!feature


## Explore compiled PyTorch

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) was introduced in 2023, but I haven't evaluated it for MegaDetector (or SpeciesNet).  Evaluate it.

P2

E3

!exploratory


## Test module manipulation in Poetry

run_detector_batch does some complicated manipulation of the global module dictionary to allow switching between YOLO libraries (e.g. YOLOv5 and YOLOv9 use some of the same global symbols to mean different things).  This relies on some elements of of the `site-packages` folder that I've tested in standard Python venv's and in Anaconda/Miniforge environments, but I haven't tested them in Poetry environments.

P1

E2

!testing
!maintenance


## Populate the "summary_report" field

As of version 1.4 of the [MegaDetector output format](lila.science/megadetector-output-format), a "summary_report" field was created to allow presenting a short bit of HTML through Timelapse.  This field is not yet populated by any of the scripts by which I run MD and SpeciesNet.  This is low priority until it also becomes a Timelapse priority.

P3

E1

!feature


## Enhancements to classification smoothing

The classification_postprocessing module provides tools for smoothing classifications within images and sequences, but this hasn't been robustly evaluated, and currently makes no use of confidence values.  Use a more rigorous approach to setting the magic numbers, and explore the use of confidence values, at least to break ties.

P1

E2

!exploratory
!feature


## Standardize removal of temporary files

Replace all deletions of temporary files/folders (every instance of unlink, rmtree, remove, etc.) with a central entry point that can - according to a global setting - recycle rather than deleting (especially on Windows where there is a standard definition of "recycle").

P3

E2

!feature


## Standardize multiprocessing parameters

Many functions in this repo take parameters to indicate (a) a number of parallel workers and (b) whether to use threads or processes.  The number of workers is variously called "n_cores", "n_workers", "n_threads", etc., and the flag to indicate threads vs. processes is variously called "use_threads", "parallelize_with_threads", "pool_type", "worker_type", etc.  Standardize these.  The actual implementations are quite standardized already, it's just the parameter naming that is inconsistent.

P3

E2

!maintenance


## Gracefully handle CPU execution against GPU PyTorch

When the GPU version of PyTorch is installed, but inference is run on the CPU (typically because CUDA_VISIBLE_DEVICES is -1), on Windows, stuff crashes with error 3221225477.  There's no way to fix this and it doesn't really matter, but I'd like to be able to handle this more gracefully.

P3

E1

!bug


## Standardized cast consistency in docs for CLI arguments

There is inconsistent casing in CLI arguments, fix this

P4

E2

!docs


## Validate class parameter documentation

I document class parameters (for publishing with Sphinx) by adding #: tags in the constructor.  It's not entirely straightforward to make sure I've done this for every parameter I want to document, in particular I bet there are some cases where I've forgotten to change this from "#" to "#:".  Find and fix those, and add a custom linting function for this (it's OK if that requires a manually-launched linting script).

P3

E3

!docs


## Avoid extra EXIF step in manage_local_batch

Change manage_local_batch to read datetime during inference, rather than doing a separate EXIF read step.

P2

E0

!feature


## Improvements to md_to_labelme

md_to_labelme is critical to accelerated creation of MD training data, I'd like to add two new features:

1. Support classification data
2. Support variable confidence thresholds across classes

P1

E1

!feature
!training


## Merge get_file_sizes and parallel_get_file_sizes

There is some redundancy between the get_file_sizes and parallel_get_file_sizes functions, clean this redundancy up.

P3

E1

!maintenance


## Merge resize_images and resize_image_folder

There is some redundancy between the resize_images and resize_image_folder functions, clean this redundancy up.

P3

E1

!maintenance


## Argument validation in postprocess_batch_results

postprocess_batch_results should at least notify the user, and probably error, when it’s run without one of (1) absolute paths, (2) ground truth, or (3) an image folder.  I don't think there's a sensible case where you would not want to specify one of those things.

P4

E0

!maintenance


## Add blurring to video rendering

visualize_detector_output allows the caller to blur humans; add this functionality to visualize_video_output.

P2

E0

!feature


## Revisit use_map_location on MPS devices in pytorch_detector

In pytorch_detector, we load MD like this on non-Apple (i.e., non-MPS) devices:

```python
checkpoint = torch.load(model_pt_path, map_location=device, weights_only=False)
```

...but on MPS devices, this leads to errors about 16 vs 32 bit models, so we do this:

```python
checkpoint = torch.load(model_pt_path, weights_only=False)`
...some other stuff...
model.to(device)
```

This task is two-fold:

* Assess whether map_location is supported on Apple silicon in recent versions of PyTorch, so we can eliminate the special case
* Assess whether there is a performance/memory consumption benefit/cost to using map_location. 

I last tried switching to use_map_location on mps devices on 2025.08.18, it did not go well.  Dropping this to P3.

P3

E3

!maintenance


## Remove support for torch 1.x loading

The "weights_only" parameter was added in PyTorch 2.x, and is required for loading MD models.  So as long as we support both PT 1.x and PT 2.x, we have a try/except in pytorch_detector to first try loading with weights_only=False, but then we fall back to omitting this parameter.  If we finally ditch support for PT 1.x, remove this try/except block.

P3

E0

!maintenance


## Remove unnecessary null failures from RDE output

repeat_detections_core adds an unnecessary "failure" field (set to null) for all successful images.  This is not a violation of the format spec, but it's silly.  This happens because this script goes through a pandas dataframe after an intermediate, then converts rows back to dicts before exporting.  Fix this.  The easiest fix is to just remove these prior to export.

P3

E0

!maintenance


## Clean up all long argument lists in run_detector_batch

run_detector_batch (arguably the most important module in the repo) has super-long argument lists for basically every function.  Other modules in the repo handle this by moving the relevant options to a dedicated options class.  Do this for run_detector_batch.  Backwards compatibility is not a huge issue as long as the CLI doesn't break.

P3

E1

!maintenance


## Support running MD in the ultralytics package

MDv5 (and MDv1000-redwood) don't work in the ultralytics package.  They *almost* work, but a few class names have changed.  There's no particular reason this is important, but it would allow new deployment surfaces, so it wouldn't hurt.  I think this will require a one-time step where we move the weights to a slightly different container.

P4

E2

!maintenance


## Consider removing yolov9-pip dependency

The megadetector package takes a dependency on yolov9pip, even though I don't think a lot of people will use MDv1000-cedar.  It would simplify installation if we removed this dependency, and asked users to install yolov9pip when they want to use MDv1000-cedar, like we do for MDv1000-larch.

P1

E1

!admin


## Load class names from detector files if available

Currently we assume MegaDetector classes in run_detector_batch, and we allow custom class mappings via --class_mapping_filename.  Long ago, class names weren't stored in YOLO-style detectors, now they are, so, optionally load class names from detectors.  This doesn't really matter when using MegaDetector, but it removes the hassle of using --class_mapping_filename when using non-MD detectors.

P3

E1

!feature


## Address module-level globals in run_detector_batch and run_detector

The DEFAULT_DETECTOR_LABEL_MAP module-level global variable in run_detector_batch is used to pass custom class mappings around; this is rare and not very important, but sloppy.

Similarly, the USE_MODEL_NATIVE_CLASSES module-level global in run_detector is used to pass a custom option around; this is rare and not very important, but sloppy.

Fix both of these.

P4

E2

!maintenance
