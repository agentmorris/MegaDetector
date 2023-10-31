# MegaDetector challenges

## Table of contents

* [Overview](#overview)
* [Known challenges](#known-challenges)
  * [Reptiles and other under-represented species](#reptiles-and-other-under-represented-species)
  * [Unusual camera angles](#unusual-camera-angles)
  * [Random AI failures that will drive you bonkers](#random-ai-failures)
  * [Small objects that are <i>just</i> at the detection threshold](#very-small-things)
* [What can we do about these cases?](#what-can-we-do-about-these-cases)
* [Ending on a happier note](#ending-on-a-happier-note)


## Overview

We like to think MegaDetector works pretty well most of the time; you can read third-party evaluations like [this one](https://wildeyeconservation.org/megadetector-version-5/) and [this one](https://www.sciencedirect.com/science/article/pii/S2351989422001068?via%3Dihub) if you want to feel good about MegaDetector, or you can look at sample results like [these](https://lila.science/public/snapshot_safari_public/snapshot-safari-kar-2022-00-00-v5a.0.0_0.200/detections_animal.html).  But lest you should think MegaDetector always works, this page is here to harsh your mellow a little.

Of course, any AI model will drive you crazy by missing one image in a dataset where it otherwise works fine, that's not what this page is about.  Also, some things are just outside of MegaDetector's domain: it <i>shouldn't</i> work for fish in underwater video, or for images taken from drones, or for veeeeeeeeeery small things in the distance that aren't discernible (even by humans) as animals in single images.  That's also not what this page is about.

<b>This page is about a few cases we consider within MegaDetector's operating domain, where MegaDetector doesn't work as well as we would like.</b>

The reason we made this page is two-fold:

* We always want to remind users that no matter how many evaluations you read of MegaDetector (or any AI model!) that say it's X% accurate, <i>there's no substitute for trying a model on your own data before you use it to avoid looking at images</i>.  We can help you do this; in the typical case where you don't already know the right answer for lots of images, we typically recommend making a sample page like [this one](https://lila.science/public/snapshot_safari_public/snapshot-safari-kar-2022-00-00-v5a.0.0_0.200/) that helps you quickly grok how well MegaDetector is or isn't working.

* We want to catalyze the community to help us fill these gaps.  There are three ways to fill these gaps: (1) accumulating lots of training data for a hypothetical next version of MegaDetector, (2) fine-tuning MDv5 to work better in specific cases, and (3) adding features to the inference code improve performance on difficult cases, sometimes with a small amount of human intervention.  We're quite bullish on (2) and (3)!

All that said, before we embark on a page full of failure stories, a reminder that for 90% of the camera trap datasets we see, MegaDetector works, where "works" is defined as "can save the user time".

OK, now on to failure stories... these fall into basically four categories:

* [Reptiles and other species](#reptiles-and-other-under-represented-species) that are in MD's domain in principle, but are under-represented in training data.
  
* [Quirky camera angles or environments](#unusual-camera-angles) that are unlike what MD saw in training, e.g. a camera placed upside-down inside a hyena den, or - as per examples below - aquatic mammals in the water, where the same species standing in the middle of a field would work fine.
  
* [Random things where AI should work](#random-ai-failures) but it just doesn't, because... reasons.

* [Things that are very small](#very-small-things), but not so small that we would consider them out of domain.  There is a gray area here by definition, but somewhere at the border between "very very small" and "an elephant right in front of the camera" is an object size that is <i>just barely</i> detectable.

## Known challenges

### Reptiles and other under-represented species

In this series of images, the animal in the burrow is pretty visually obvious.  But for some reason MD completely whiffs on the first, finds the second with 14% confidence, and finds the third (visually very similar to the first) with 75% confidence. These are not "difficult" images per se, rather this is a classic example of "this model hasn't seen enough reptiles like this in training".

<img src="images/failure-examples/sample-cfz-01.jpg" width="600">
<img src="images/failure-examples/sample-cfz-03.jpg" width="600">
<img src="images/failure-examples/sample-cfz-02.jpg" width="600">

<i>Images credit Michelle Hoffman, Central Florida Zoo</i>

In this dataset, MegaDetector worked OK (I'm not showing you all the images where it worked fine on reptiles), but (a) why did it work fine on these turtles and tegus and not the other ones on this page?, and (b) even in this case where it's "working", there are a bunch of images like these where the confidence of the detections is so low that we're basically getting lucky in avoiding false positives.  We typically recommend a confidence threshold of 15% or 20%, and here we're seeing detections at 3% or 4%, a range in which you will usually get lots of false positives.

<img src="images/failure-examples/sample-crocdocs-01.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-02.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-03.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-04.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-05.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-06.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-07.jpg" width="600">
<img src="images/failure-examples/sample-crocdocs-08.jpg" width="600">

<i>Images credit Croc Docs, University of Florida</i>

And here are some big, obvious reptiles on which MegaDetector succeeds/fails to varying degrees (1%, 5%, and 19% confidence, respectively).  In practice 1% would almost certainly be a miss (though here we happen to get lucky and have no other false positives on this image), 5% would be a maybe, and 19% would be a success.

<img src="images/failure-examples/sample-usgsfcsc-03_detections.jpg" width="600">
<img src="images/failure-examples/sample-usgsfcsc-04_detections.jpg" width="600">
<img src="images/failure-examples/sample-usgsfcsc-05_detections.jpg" width="600">

<i>Images credit USGS Fort Collins Science Center</i>

This series is really a two-for-one: reptiles <i>and</i> a close-up, top-down camera angle we rarely see in training.  The first one is a hit at 16% confidence, the others are more or less total misses.

<img src="images/failure-examples/sample-arc-02.jpg" width="600">
<img src="images/failure-examples/sample-arc-01.jpg" width="600">
<img src="images/failure-examples/sample-arc-04.jpg" width="600">

<i>Images credit Michael Knoerr, [ARC](https://arcprotects.org/)</i>

And rounding out our tour de reptiles, another combination of reptiles and a slightly unusual camera angle gives us success, possibly-success, and probably-failure, respectively (74%, 9%, and 3% confidence, respectively).

<img src="images/failure-examples/sample-ufldwec-02.jpg" width="600">
<img src="images/failure-examples/sample-ufldwec-01.jpg" width="600">
<img src="images/failure-examples/sample-ufldwec-03.jpg" width="600">

<i>Images credit University of Florida, Department of Wildlife Ecology and Conservation</i>

#### Reptile retrospective

One thing all of those examples have in common: there's clearly signal there, so fine-tuning is likely to work well.  We just need lots of bounding boxes on reptiles (probably beyond the boxes on iNaturalist data that are included in MDv5a's training data).

### Unusual camera angles

This is a bit of a catch-all category, including some things we tried to fix between MDv4 and MDv5 (e.g. cameras in trees looking straight down), but a lot of things that are rare and difficult to fix just by accumulating data: cameras inside dens or nests, cameras looking straight down a metal pole, etc.

#### Aquatic mammals with their adorable heads just peeking out of the water

Top of mind in this category right now is cameras looking out over water, especially with aquatic mammals swimming in said water.  Sometimes MD does fine, sometimes not so much, sometimes resizing the images matters, basically all the things that suggest "right at the edge of the training domain".

In the first image here, it looks like we get the swimming otter at 95%... so, all good right?  Not quite: that's a rare case where MDv4 finds something that MDv5 misses, the second image is MDv5's total whiff on this image.  And that MDv4 hit is cherry-picked; all versions of MD are unpredictable at best on most of these images.  And then you get the third example, which is not only a hit, it even manages to separate the two happy beavers.  This unpredictability again screams "edge of the training domain".

<img src="images/failure-examples/sample-sdsucheeseman-01.jpg" width="600">
<img src="images/failure-examples/sample-sdsucheeseman-02.jpg" width="600">
<img src="images/failure-examples/sample-sdsucheeseman-03.jpg" width="600">

<i>Images credit Cheeseman Lab, South Dakota State University</i>

#### Mega-occlusion that humans are smart enough to parse, but AI is not

Fences/cages/etc. that partially occlude things are often fine, but having animals interrupted by vertical bars can definitely degrade performance.  Here's an extreme case, where you and I can see an animal behind the fence, but it's more than 50% obscured, and MegaDetector misses it completely:

<img src="images/failure-examples/sample-skyisland-behindfence.png" width="600">

MegaDetector does fine when the animals are in front of the fence:

<img src="images/failure-examples/sample-skyisland-frontfence-0.png" width="600">

...even when they're a bit difficult:

<img src="images/failure-examples/sample-skyisland-frontfence-1.png" width="600"><br/>
<i>Images credit Sky Island Alliance</i>

...but animals behind that fence are a little too much to ask of MD right now.


### Random AI failures

And last but not least, sometimes we see a set of images that we consider to be squarely in the training domain - and in one of these cases, <i>literally in the training data</i> - but for some reason that our eyes can't see, MD gets confused.  I wouldn't think of these as "something about the animal is difficult", it's likely more like "something about the camera or jpeg encoding is difficult".

The first series of images comes from [this issue](https://github.com/ultralytics/yolov5/issues/9294), where MegaDetector doesn't exactly "miss" a series of obvious animals, but it puts the boxes in inexplicable locations.  And, bizarrely, when we process the image at a quarter of MD's standard resolution, everything works fine.  High-resolution == bad, low-resolution == better?  AI is maddening!

<i>Processed at the standard input size of 1280 pixels on the long side</i>

<img src="images/failure-examples/sample-snapshotsafari-01.jpg" width="600">

<i>Processed at 640 pixels on the long size</i>

<img src="images/failure-examples/sample-snapshotsafari-02.jpg" width="600">

<i>Processed at the standard input size of 1280 pixels on the long side</i>

<img src="images/failure-examples/sample-snapshotsafari-03.jpg" width="600">

<i>Processed at 640 pixels on the long size</i>

<img src="images/failure-examples/sample-snapshotsafari-04.jpg" width="600">

Maddening!!!  But also a reminder that in the <i>vast</i> majority of savanna data like this, MD does fine.  This appears to be something about a particular subset of older cameras from the Snapshot Serengeti program that confuse MD.

Last but not least, a dataset full of mountain goats where MD had no trouble with probably 90% of the images with goats, but in the other 10% we can just stare straight at a goat that MD doesn't see:

<img src="images/failure-examples/sample-dzf-01.jpg" width="600">
<img src="images/failure-examples/sample-dzf-02.jpg" width="600">
<img src="images/failure-examples/sample-dzf-07.jpg" width="600">
<img src="images/failure-examples/sample-dzf-11.jpg" width="600">

<i>Images credit Denver Zoo</i>

Are they camouflage?  Do they look like snow?  Maybe, but those are anthropomorphic explanations.  These are obvious animals in good light, and we have lots of goats and sheep in our training data, so I file these under "AI mysteries".

Sigh, just to make myself feel better, here are some from the 90% where it worked, complete with an adorable baby goat:

<img src="images/failure-examples/sample-dzf-101.jpg" width="600">
<img src="images/failure-examples/sample-dzf-102.jpg" width="600">
<img src="images/failure-examples/sample-dzf-103.jpg" width="600">
<img src="images/failure-examples/sample-dzf-104.jpg" width="600">
<img src="images/failure-examples/sample-dzf-105.jpg" width="600">

<i>Images credit Denver Zoo</i>

Another user reported that MegaDetector was having some difficulty finding sage-grouse, like the one in this image:

<img src="images/failure-examples/sage-grouse-raw.jpg" width="600">

<i>Image credit University of Idaho</i>

This is not the easiest image ever, but the grouse is fairly obvious to a human, and birds are adequately represented in training.  So we'd like MD to find this, and it does, but only at 8% confidence, which is below typical confidence thresholds, and some other, slightly more challenging grouse were missed entirely or found at an even lower confidence.  In this case, there is some good news; using YOLOv5's [test-time augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) (TTA) via [run_inference_with_yolov5_val.py](https://github.com/agentmorris/MegaDetector/blob/main/detection/run_inference_with_yolov5_val.py) made a substantial difference, boosting the confidence of this grouse, for example, to 53% (well above any reasonable threshold):

<img src="images/failure-examples/sage-grouse-tta.jpg" width="600">

<i>Image credit University of Idaho</i>

Even after TTA, recall was not 100% on this dataset, but TTA substantially improved performance on grouse (boosting recall into the high 90s) and made a qualitative difference in the utility of MegaDetector for this user.


## Very small things

Any object detector will have a lower limit on the size of objects it can detect.  Furthermore, most computer vision tools reduce the size of images to a standard size as part of the inference process: in MegaDetector's case, the default inference size is 1280 pixels on the long side.  That's pretty large, but at some point an object on the horizon becomes too small to meaningfully classify as an animal (both both humans and AI); those objects would be "out of domain", and thus outside the scope of this page.

Rather than come up with a new example, I'm going to include a figure from an excellent paper about this very issue:

> Leorna S, Brinkman T. Human vs. machine: [Detecting wildlife in camera trap images](https://www.sciencedirect.com/science/article/pii/S1574954122003260). Ecological Informatics. 2022 Oct 27:101876.

They compare the minimum size of an animal required for detection by (a) the camera's PIR sensor, (b) MDv4, and (c) a human looking at a <i>sequence</i> of images:

<img src="images/failure-examples/leorna-2022-figure.jpg" width="500">

<i>Image credit [Leorna et al](https://www.sciencedirect.com/science/article/pii/S1574954122003260).</i>

The numbers in this figure are based on MDv4, which had a smaller input size than MDv5 and was less accurate overall, so the numbers would change a little if the same experiments were run with MDv5.  But the qualitative conclusions would be the same for MDv5: MD can detect things much smaller than a PIR sensor, but humans looking at sequences will always be able to see things that are much smaller than an object detector looking at a single image.

[Klemens at al](https://esa2023.eventscribe.net/fsPopup.asp?Mode=presInfo&PresentationID=1275706) came to a similar conclusion; they placed cameras on long, straight stretches of highway to monitor animal crossings, and found that MDv5 did well for animals within some distance of the camera, but at some point far away, performance started to fall off.

But what about objects that are <i>right</i> on that line?  Can we do something to find them?  As discussed above, [test-time augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) is likely to help in some of these cases.  Also, if your original image is larger than 1280 pixels on the long side, you are sort of throwing away some pixels when you run your image through MegaDetector.  We have a [preliminary script](https://github.com/agentmorris/MegaDetector/blob/main/detection/run_tiled_inference.py) that chops images into smaller images (typically 1280px), runs MD on the smaller images, and stitches the results together.  This creates some new false positives, but definitely improves recall for far-away objects.  If you're interested in trying this out, <a href="mailto:cameratraps@lila.science">email us</a>!


## What can we do about these cases?

<a href="mailto:cameratraps@lila.science">Email us!</a>  We want to hear about the cases where MD doesn't work; this helps us prioritize training data.  This is how we made some improvements between MDv4 and MDv5, for example: we heard from users that bait stations in particular where causing MDv4 to misbehave, so we went wild with bait station training data for MDv5, and we think the situation is lots better now.

As with the grouse example above, there is often something we can do at inference time - without training any new models - to improve results.  This typically includes [test-time augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/), [repeat detection elimination](https://github.com/agentmorris/MegaDetector/tree/main/api/batch_processing/postprocessing/repeat_detection_elimination) (which can allow a much lower confidence threshold), and merging of MDv5a/MDv5b results.

Sometimes a difficult case is beyond the reach of inference-time enhancements, but there is real signal in *all* of the above cases, which suggests that a modest amount of fine-tuning would work really well.  We are excited to work with the community to make the annotation and fine-tuning processes easier, which is 99% about ergonomics and workflow, 1% about AI.


## Ending on a happier note

With all that negativity, we sure hope this page isn't your first introduction to MegaDetector. :)  

This page is really hitting on the long tail: at least 90% - probably more like 95% - of the use cases we see don't look like the above examples at all.  Typically we see very high recall, with varying degrees of precision, but almost always high enough precision to save users time.  So overall, we're really happy with the performance of MDv5, and the third-party evaluations ([1](https://wildeyeconservation.org/megadetector-version-5/), [2](https://www.sciencedirect.com/science/article/pii/S2351989422001068?via%3Dihub)) and [sample results](https://lila.science/public/snapshot_safari_public/snapshot-safari-kar-2022-00-00-v5a.0.0_0.200/detections_animal.html) that we mentioned above should drive that point home.

The takeaway from this page isn't that these cases are common (they're not!), rather that with any AI model, always test with a critical eye on your own data!
