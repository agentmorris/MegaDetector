---
name: run-megadetector
description: >-
  Run MegaDetector on a folder of camera trap images and/or videos to find animals, people, and
  vehicles (and thereby identify blank images). Use this whenever the user asks to run MegaDetector
  on their images, or asks for help separating animal images from blank/empty/misfire/false-trigger
  images, or asks to use AI to analyze, sort, or triage camera trap images. Handles Python setup,
  running the model, and turning the results into whatever the user actually needs (a results file,
  a spreadsheet, or images sorted into folders).
---

# Running MegaDetector

## What MegaDetector is

MegaDetector is an object detector for motion-triggered wildlife camera ("camera trap") images and videos. It has three categories: "animal", "person", and "vehicle". It does not identify species; it just finds animals, people, and vehicles, which makes it especially useful for filtering out the blank images that dominate most camera trap datasets. Most people run it to spend their time looking at animals instead of blanks and humans.

This skill runs MegaDetector recursively on a folder of images (and/or videos) and produces whatever output the user actually needs. Your job is to (a) figure out what form of output will be useful to this particular user, (b) get a suitable Python environment running with as little fuss as possible, (c) run the model, and (d) postprocess the results into the form they want.

You are assumed to be a capable command-line agent. The commands below are the canonical way to do each step, but you can adapt when a specific machine calls for it.

## The overall flow

The steps, roughly in order:

- Clarify what output the user wants (this is usually the most important step).
- Look at the folder: count images, count videos, note if both are present.
- Make sure a Python environment with the `megadetector` package is available.
- Give the user a rough time estimate so they know what kind of job this is.
- Run MegaDetector (images with `run_detector_batch`, videos with `process_video`).
- Postprocess into the requested form.
- Optionally generate a results-preview page.
- Offer to clean up anything you installed.

Environment setup and counting the folder can happen early and in any order; the time estimate and the run come after both are done.

## Step 1: Clarify what the user wants

In a narrow set of cases the request needs no clarification. If the user says something like "run MegaDetector on the images in c:\my_images and save the results to c:\my_images\md_results.json", just do it. But that's the exception. Almost always you should figure out how the user plans to use the results before you run anything.

If the user just says "run MegaDetector" (or "help me find the animals", "help me get rid of the blanks", etc.), ask how they're planning to use the results. Their answer will usually map onto one of three outputs:

- **A `.json` results file.** This is the native [MegaDetector output format](http://lila.science/megadetector-output-format), which is recognized by image-review tools, most notably [Timelapse](https://timelapse.ucalgary.ca/). If the user mentions Timelapse, or says they've heard there's a file format other tools can read, the default `.json` output is what they want. Most users never open this file themselves.
- **A `.csv` spreadsheet.** Some users want a spreadsheet, often to read with their own R code or load into another tool.
- **Images sorted into folders** like "animal", "empty", "person", "vehicle". Common when the user wants to physically separate the interesting images from the blanks.

You don't need to force a decision if they already told you. But if it's ambiguous, ask. A user who is new to this often hasn't thought past "I want to skip the blank images", so a short, plain-language exchange here saves a lot of wasted work.

### If the user wants images sorted into folders

Folder separation happens after MegaDetector runs, but you need three pieces of information up front, before running the model:

1. **A destination folder.** Confirm there's enough free space on the destination drive by checking it yourself (see below) — don't ask the user to check.
2. **Copy or move.** Strongly recommend copying rather than moving, as long as there's disk space. Take no potentially destructive action unless there is genuinely no other option. If space is tight, say something like "you might want to consider using an external hard drive so you can copy images rather than moving them" before you ever consider moving.
3. **Bounding boxes.** Ask whether they want boxes drawn around the detected objects on the copied images.

You also need to explain confidence thresholds, because sorting into folders forces a concrete yes/no decision that a `.json`/`.csv` file does not. Here is a good way to start that explanation (adapt as needed):

> AI models don't tell you what's in an image, they just assign confidence values. For each thing MegaDetector finds, it reports how confident it is (from 0 to 1) that the object is an animal, a person, or a vehicle. To actually sort images into folders, we have to pick a cutoff: anything at or above the cutoff counts as a detection, anything below it gets treated as blank. A lower cutoff catches more animals but also lets more false positives (waving grass, shadows, etc.) through; a higher cutoff is cleaner but risks missing some real animals. I'll use MegaDetector's default cutoff, which works well across a wide range of datasets. And you don't have to get this perfect up front — re-sorting at a different cutoff is quick, because it reuses the results and doesn't re-run the slow AI step.

Unless the user asks to change it, leave the confidence threshold at its default (the separation tool automatically uses a sensible per-model default). Reassure them that re-sorting later at a different cutoff is cheap.

### A note on confidence thresholds and `.json`/`.csv` output

Confidence thresholds are only relevant for folder separation. When you produce a `.json` or `.csv` file, the only threshold applied is an internal one very close to 0, so essentially every detection is retained; all real thresholding happens later in whatever tool the user loads the file into. Don't raise thresholds with the user in the `.json`/`.csv` cases.

## Step 2: Look at the folder

Count the images and videos in the target folder (recursively). The package has helpers you can reuse rather than reinventing extension lists:

- `megadetector.utils.path_utils.find_images(folder, recursive=True)`
- `megadetector.detection.video_utils.find_videos(folder, recursive=True)`

You need the image count for the time estimate. You also need to know whether videos are present.

Still images are far more common than video, but video comes up. If you find **both** images and videos in the folder, confirm with the user that they want both processed. They are handled by two different tools and produce two separate results files (see below). You don't need any special flag to process only the images: `run_detector_batch` reads only image files and simply ignores any videos in the folder (and `process_video` likewise reads only videos), so in a mixed folder you can point both tools at the same root and each picks up only its own media type — processing photos only just means not running `process_video` at all.

## Step 3: Estimate how long this will take

Do this from heuristics. Do not run a timing test, and don't prompt the user for one — just give a rough estimate based on the hardware and the image count, and make clear it's *very* approximate. The goal is to help the user decide what kind of job this is: a "sit and watch the progress bar" job, a "coffee break" job, an overnight job, a long-weekend job, or completely impractical on this machine.

First figure out what accelerator the machine has:

- **Nvidia GPU:** `nvidia-smi` succeeds. Get the GPU name from it.
- **Apple Silicon (MPS):** on a Mac, `python -c "import torch; print(torch.backends.mps.is_available())"` reports True.
- **Neither:** CPU only.

Then estimate throughput. The `known_models` / `device_token_to_mdv5_inference_speed` tables in `megadetector/detection/run_detector.py` are your reference; these are per-second rates for MDv5 / MDv1000-redwood (the models this skill runs are all the same speed class):

| Hardware | ~images/sec | ~images/day |
|---|---|---|
| RTX 4090 | 17.6 | 1,500,000 |
| RTX 3090 | 11.4 | 985,000 |
| RTX 3080 | 9.5 | 820,000 |
| RTX 3050 (desktop) | 4.2 | 363,000 |
| RTX 3050 (laptop) | 3.0 | 250,000 |
| Quadro P2000 | 2.1 | 180,000 |
| Apple M3 Pro (18 GPU cores) | 4.6 | 398,000 |
| Apple M1 (8 GPU cores) | 1.85 | 160,000 |
| Intel i7-13700K (CPU, 1 core) | 0.8 | 69,000 |
| Intel i7-12700 (CPU, 1 core) | 0.5 | 43,000 |

If the user has a GPU or an MPS accelerator whose exact model isn't in the table, do a quick web search to see how their chip compares to the nearest entries (roughly, by relative compute), and scale from there. This will be approximate, and that's fine — say so.

If the user has no GPU, don't overthink it: assume **30,000–50,000 images per day** and give a very rough estimate, stressing that it's very approximate and can vary a lot with image size and disk speed.

Translate the image count into wall-clock time and frame it for the user, e.g.:

- Up to ~15 minutes: a "sit and watch the progress bar" job.
- ~15 minutes to a couple of hours: a "coffee break" (or lunch) job.
- A few hours to overnight: start it near the end of the day and it'll be ready in the morning.
- ~1–3 days: a "long weekend" job — start it Friday, check back Monday.
- More than that: probably impractical on this machine. Suggest a machine with an Nvidia GPU if they have access to one, and mention that the MegaDetector developers will process large batches for free (see Contact, below) — this is genuinely offered and is often the best answer for someone with a big backlog and a slow computer.

Communicate expectations rather than treating speed as a dealbreaker. A big job on a modest machine is usually fine if the user knows to start it at the end of the day (or on a Friday) and let it run.

## Step 4: Get a Python environment

The user probably has limited Python familiarity and may not have deliberately installed any Python tools. When you have to talk to them about this, avoid jargon like "venv", "conda environment", or "pip" — say things like "the tools needed to run MegaDetector". You can get more technical only if the user shows Python familiarity. Prompt the user as little as possible; rely on the agent harness to handle any permission prompts. Ideally the whole setup is just a brief notification like "I'm installing the tools needed to run MegaDetector now — this is a one-time setup that takes a few minutes and downloads a couple of GB."

Work down this ladder and stop at the first rung that applies:

**Rung 0 — is it already runnable?** Check first whether the `megadetector` package already imports in the Python you'd naturally use (try `python`, `python3`, `py -3`): `python -c "import megadetector"`. If it imports, install nothing further — note that interpreter's absolute path and use it for everything below. (Still do the GPU-wheel check below; it applies here too.)

**Rung 1 — reuse an existing Python (>= 3.10) to make an isolated environment.** If there's a suitable Python on the system, create a dedicated environment for MegaDetector with it (`python -m venv <env-dir>`), then install into that. If the only thing available is an existing conda/mamba/Miniforge install, a dedicated conda environment is equally fine.

**Rung 2 — nothing suitable is installed, so install the smallest thing that works.** The recommended tool here is [`uv`](https://docs.astral.sh/uv/): it's a single small self-contained binary, needs no admin rights, can install a private copy of Python for you (`uv python install 3.12`), can create the environment (`uv venv <env-dir>`), and installs packages quickly (`uv pip install ...`). It's the least invasive way to get a clean Python without bothering the user, and cleanup is just deleting one folder. If for some reason you can't obtain `uv`, fall back to installing Miniforge.

Put the environment in a **stable per-user location** so it survives reboots and can be reused next time — for example `%LOCALAPPDATA%\megadetector` on Windows or `~/.megadetector` on macOS. Do **not** put the environment in system temp: it's several GB (mostly PyTorch), and the OS can reclaim temp space, which would force a multi-GB re-download and would defeat the "leave it installed for next time" option in cleanup. If the per-user environment already exists from a previous run, reuse it. (System temp is fine for the transient preview page in Step 7 — just not for the environment.)

Install the package with `<env-python> -m pip install --upgrade megadetector` (or `uv pip install --upgrade megadetector`). On Linux and macOS the `megadetector` package is the only dependency you need.

**GPU wheel (Windows + Nvidia) — applies whether you just installed or found an existing environment.** When installing PyTorch with pip on Windows, the CPU-only wheel is usually what gets installed, even when the machine has an Nvidia GPU — and users often don't know they need to fix this. So on Windows: if `nvidia-smi` succeeds (an Nvidia GPU is present) but `python -c "import torch; print(torch.cuda.is_available())"` prints False, install the GPU build of PyTorch:

```
pip install torch torchvision --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

(Use the corresponding `uv pip install ...` form if you're using `uv`.) This is not needed on macOS (Apple Silicon acceleration works with the default wheel) and is out of scope for this skill on Linux. Gate it on the real signal — a present Nvidia GPU that PyTorch isn't using — rather than reinstalling unconditionally.

**Always invoke the environment's Python by its absolute path** (`<env-dir>\Scripts\python.exe` on Windows, `<env-dir>/bin/python` on macOS) for every subsequent command. Do not rely on "activating" the environment — activation doesn't reliably persist across separate commands in an agent harness, and calling the interpreter directly avoids a whole class of "it used the wrong Python" bugs.

**Confirm the environment works.** A quick check with `python -m megadetector.utils.gpu_test` confirms the install and reports whether the accelerator is visible. The MegaDetector model file (~280 MB for MDv5a) downloads automatically the first time you run the model; if the machine is offline and the download fails, say so clearly rather than letting it look like a slow run. Camera trap folders often live on paths with spaces — always quote paths.

## Step 5: Run MegaDetector on images

Use `run_detector_batch`. Unless the user says otherwise, run **MDv5a**. Tell the user that's what you're using, but don't ask them to confirm the model choice. If the user explicitly asks for MDv1000-redwood or MDv5b instead, just swap the model name — nothing else changes. If the user asks for any *other* MegaDetector model (including other models in the MDv1000 family), tell them that's outside the scope of this skill: the other models trade accuracy for compute speed, and that tradeoff usually costs more human time than it saves in compute time, which is a nuanced discussion this skill doesn't cover.

The command:

```
<env-python> -m megadetector.detection.run_detector_batch MDV5A "<image_folder>" "<output.json>" \
  --recursive \
  --output_relative_filenames \
  --use_image_queue \
  --preprocess_on_image_queue \
  --use_threads_for_queue
```

- **detector_file** is the model name, `MDV5A` (not a filename). The model downloads automatically on first use.
- **image_file** is the folder to process.
- **output_file** is the `.json` results file:
  - If the user asked for a `.json` file, use the filename they gave you; if they gave only a folder, put it there under a sensible default name like `md_results.json` (and confirm the name/location).
  - If the user asked for a `.csv` file, put the `.json` in the same folder as the `.csv`, and keep it (don't delete it).
  - If the user asked for folder separation, put the `.json` in the destination folder, and keep it there after separating (don't delete it). A reasonable default name is `md_results.json`.
- Always pass `--recursive`, `--output_relative_filenames`, `--use_image_queue`, `--preprocess_on_image_queue`, and `--use_threads_for_queue` unless the user specifically indicates otherwise. Relative filenames are almost always correct, and folder separation *requires* them.

**Checkpointing.** For anything that will run more than a few minutes, add `--checkpoint_frequency 10000` (writes progress every 10,000 images). There's essentially no downside — writing checkpoints is fast. If the job is interrupted (crash, reboot, closed window), resume it by re-running the *exact same command* with `--resume_from_checkpoint auto` added; you'll see something like "Restored N entries from the checkpoint".

**Running a long job without freezing.** These jobs can run for hours. Start the run as a background process that is still tied to your session (for example, Claude Code's background command execution) so you stay responsive to the user — they may have a side question or want a progress update — but do **not** fully detach it into an independent system service. It's fine (and expected) for the job to stop if the user closes the agent session. Tell the user plainly: this will take about `<your estimate>`; please leave this session open while it runs; closing it will stop the job; you can ask me for progress at any time; and if it does stop, I can resume it from where it left off.

Decide that the run is finished by watching for the `run_detector_batch` (or, for video, `process_video`) process itself to exit — the process completing is the signal that it's done, and a fully-written output `.json` is then a good sign that it succeeded. Do **not** judge completion from the progress bar or from the mere appearance of the `.json` file: `tqdm` progress is quirky (even "100%" means *approximately* done, and the process may still be finalizing the `.json`), and the file can exist before the run is truly finished. Use the progress bar only when the user asks for a progress update or a rough ETA — not to determine whether the job is done.

Backgrounding the run keeps you responsive, but launching it is **not** the finish line — you are responsible for seeing the run through to completion and then carrying out the remaining steps (postprocessing, and any preview or cleanup). Do not treat the task as done when you kick off the run. When it finishes, continue the workflow, and then tell the user plainly that the job is complete, exactly where the results are, and a short summary of what was produced (for example, how many images were processed and how many had detections).

## Step 5b: Process videos (only if videos are present)

If the folder has videos (and the user confirmed they want them processed), run them separately with `process_video`, producing a **second** results file. Explain to the user that MegaDetector only "sees" one image at a time, so to process video we extract frames and run those through MegaDetector. By default we run three frames per second; confirm that's OK with them.

```
<env-python> -m megadetector.detection.process_video MDV5A "<video_folder>" \
  --recursive \
  --output_json_file "<videos.json>" \
  --time_sample 0.333333
```

- `MDV5A` and the video folder are positional (model first, then folder).
- Always pass `--recursive` and `--output_json_file`.
- Pass `--time_sample 0.333333` (≈ 3 frames/second) unless the user specifically wants every frame processed, in which case omit it.
- Video frame extraction uses OpenCV, which comes with the `megadetector` package — no separate video software (e.g. ffmpeg) is needed.

If a video job will run for a while, background it and monitor it exactly as in Step 5: treat it as finished when the `process_video` process exits, not when the `.json` appears.

Skip the results-preview step (Step 7) for videos.

## Step 6: Postprocess into the requested form

**If the user wanted a `.json` file:** nothing to do. You're done after Step 5.

**If the user wanted a `.csv` file (and didn't specify a particular format):** leave the `.json` in place, and also convert it to MegaDetector's `.csv` format:

```
<env-python> -m megadetector.postprocessing.convert_output_format "<results.json>" --output_path "<results.csv>"
```

This is a MegaDetector-specific (non-standard) `.csv`. If the user needs a *specific* `.csv` format (e.g. for a particular tool or a piece of R code), there's no universal answer — you'll likely need to write a small, usually simple, bespoke converter that reads the `.json` and writes exactly the columns they need. Ask what columns/format the downstream tool expects.

**If the user wanted images sorted into folders:** use `separate_detections_into_folders`. Before running, confirm there's enough free space on the destination drive by summing the source image sizes and comparing against free space (with some headroom); do this yourself rather than asking. Copy by default.

```
<env-python> -m megadetector.postprocessing.separate_detections_into_folders \
  "<results.json>" "<image_folder>" "<destination_folder>" \
  --n_threads 4 \
  --allow_existing_directory
```

- The three positionals are the results file, the input image folder, and the destination folder.
- Pass `--allow_existing_directory`. Because the results `.json` lives in the destination folder (per Step 5), the destination is already non-empty, and without this flag the tool aborts with "Target folder exists and is not empty".
- Add `--render_boxes` if the user wanted bounding boxes drawn on the output images.
- Leave confidence thresholds at their defaults unless the user asked to change them (the tool picks a sensible per-model default automatically).
- Use `--n_threads 4`.
- Do **not** pass `--move_images` unless copying is genuinely impossible for lack of disk space and the user has explicitly accepted moving after you've recommended an external drive. Moving is potentially destructive; copying is the strong default.
- This creates folders like `animals`, `empty`, `people`, `vehicles`, and combinations (e.g. `animal_person`) under the destination. Images MegaDetector couldn't read (corrupt files, etc.) go into a `processing_failure` folder — glance at it if it's non-empty. Keep the `.json` file (in the destination folder); don't delete it.

Because re-sorting reads the existing `.json`, you can re-run this step at a different threshold quickly, without re-running MegaDetector. Mention this if the user is unsure about the cutoff.

## Step 7: Results-preview page (images only)

If the user processed **more than 5,000 images**, ask whether they'd like a preview page to help them quickly assess the results. Tell them it needs no intervention from them and takes about 1–5 minutes to generate. This page randomly samples a few thousand images, splits them into "detections" and "non-detections" at a reasonable confidence threshold, and renders them — it's extremely helpful for a new user who needs to confirm at a glance that MegaDetector did something reasonable on their images.

If they say yes:

```
<env-python> -m megadetector.postprocessing.postprocess_batch_results \
  "<results.json>" "<temp_output_dir>" \
  --image_base_dir "<image_folder>" \
  --num_images_to_sample 7500 \
  --viz_target_width 1000 \
  --max_figures_per_html_file 1000
```

- The two positionals are the results file and the output directory. Put the output directory in **system temp** — this is a transient artifact.
- `--image_base_dir` points at the image folder.
- Everything else is left at its default.

When it finishes, give the user the path to the generated `index.html` (or open it for them if you can). Skip this step for video results.

## Step 8: Cleanup

If you had to install anything — Python itself, and/or the `megadetector` package and its dependencies — offer to clean it up at the end. Say something like:

> It took a few minutes to install the tools needed to run MegaDetector, so if you're likely to do this often, it might be best to leave them installed. But they take up a few GB, so if you're unlikely to do this again, I can clean it up for you. Would you like me to clean up the installed components?

If you install into the stable per-user location described in Step 4, cleanup is just deleting that one folder. If you installed nothing (Rung 0 — the environment was already there), there's nothing to clean up.

## If the user asks about species classification

MegaDetector only does blank/animal/person/vehicle detection — it does not identify species. Tell them that, and that there are a variety of other models available; for a full list they can look at the [camera-trap ML model list](https://agentmorris.github.io/camera-trap-ml-survey/#publicly-available-ml-models-for-camera-traps). If they aren't sure which to run, a good starting point is [SpeciesNet](https://github.com/google/cameratrapai/), which has wide geographic and taxonomic coverage; its repo README explains how to run it (and will likely soon have a skill like this one).

## If the user asks about AI tools for camera traps in general

Point them at [Everything I know about machine learning and camera traps](https://agentmorris.github.io/camera-trap-ml-survey/).

## Contact

If the user has a question about MegaDetector (or using AI for camera trap data) that you can't answer, or reports a bug that seems real and that you can't solve, it's appropriate to share the developer's contact information: Dan Morris <cameratraps@lila.science>. This is also the address to use for the "we'll process your big backlog for free" offer mentioned in Step 3.
