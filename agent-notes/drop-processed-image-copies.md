# Not queueing original images after preprocessing

## Problem

When `preprocess_on_image_queue` is enabled, loader workers run the detector's `preprocess_image()` and put the whole result dict on the queue. For `PTDetector`, that dict contains the letterboxed image (`img_processed`) plus two full-resolution copies of the original image: `img_original` (a numpy array) and `img_original_pil` (a PIL image). The originals are most of the bytes pickled for each image: about 2/3, or roughly 21 MB for a 5 MP frame. The consumer has to unpickle all of it. `multiprocessing.JoinableQueue` pickles items even when the loaders are threads, so `use_threads_for_queue` pays the same cost.

The originals are there for historical reasons. The queue originally only loaded images, and preprocessing happened in the consumer. When preprocessing moved to the loaders, passing the original along was the smallest change that kept the downstream code working, since that code still needed the image's shape and metadata.

Nothing downstream needs the pixels:

- PyTorch inference only needs the original's shape. `PTDetector._process_batch_group()` (`pytorch_detector.py`) copies `image_info['img_original']` into the per-image metadata, and later uses only `img_original.shape`, both for the `ratio` computation and for the `scale_coords()` call in classic compatibility mode.
- The only other consumer-side reads are the metadata blocks in `run_detector_batch._process_batch()` and `run_detector_batch._process_image()`. These pull `img_original_pil` out of the dict to compute width/height (`include_image_size`), the timestamp (`include_image_timestamp`), and EXIF tags (`include_exif_tags`).

Benchmarks for [PR 256](https://github.com/agentmorris/MegaDetector/pull/256) used an RTX 3050, 20 logical CPUs, 2-3 MP images, and batch size 8. With MDv1000-spruce, dropping the originals raised queue throughput from about 14.5 to about 20.6 img/s, with consistent repeats. With MDv1000-redwood, it went from about 4.9 to about 5.35 img/s. On that machine, preprocessing on the queue without the fix was *slower* than not using the queue at all; with the fix, it matched the no-queue path. The PR author's machine had 5 MP images, 8 vCPUs, and a faster GPU, and saw 2-3x gains, with the queue path ending up well ahead of the no-queue path. MDV5A on the RTX 3050 was GPU-bound (100% utilization), so nothing on the CPU side mattered there.

PR 256 fixed this by having loaders set both original-image fields to `None` before queueing, unless a metadata flag was set. The fix below is more complete: it removes the flag and also covers the metadata path and video.

## Recommendation

Loaders should queue only what the consumer needs: the preprocessed image, the original's shape, and any requested metadata, computed in the loader. The original image should not cross the queue at all for detectors that don't run inference on it.

### Record the original shape in `preprocess_image()`

Add `img_original_shape` to the dict returned by `PTDetector.preprocess_image()`, set *after* the optional resize inside that function so that it matches the `img_original.shape` the current code reads. Change `_process_batch_group()` to carry and use `img_original_shape` instead of `img_original`. PR 256 has a working version of exactly this change.

Once nothing reads them, `preprocess_image()` can stop returning `img_original` and `img_original_pil` entirely. That is simpler than stripping them before queueing, and there are no external users of this API to worry about.

### Compute metadata in the loader

In `run_detector_batch._producer_func()`, after loading the image and before (or alongside) preprocessing, compute whichever of width/height, timestamp, and EXIF tags were requested. Use the same code the consumer uses now: `image.width`/`image.height`, `get_image_datetime(image)`, and `read_exif.read_pil_exif(image, exif_options)`. Attach the results to the queued item, e.g. as a small `metadata` dict alongside the preprocessed dict. Then remove the `img_original_pil` lookups from `_process_batch()` and `_process_image()`, and have them copy the precomputed fields into the result instead.

To do this, `_run_detector_with_image_queue()` has to pass `include_image_size`, `include_image_timestamp`, `include_exif_tags`, and the EXIF options to the producers. Today those only go to the consumer.

Preserve the current error behavior. A failure while reading EXIF or the timestamp should not turn into an image load failure: it should produce the same per-image result it does today, i.e. the same fields, missing or populated the same way. Check how the consumer currently handles exceptions from `get_image_datetime()` and `read_pil_exif()`, and mirror that in the loader.

Width/height/timestamp/EXIF must come from the same image object the consumer uses now: the result of `vis_utils.load_image(im_file)` in the producer. That's the same call, so this should fall out naturally, but check it explicitly, e.g. whether width/height are taken before or after any EXIF-orientation handling in `load_image()`.

When the queue is enabled but `preprocess_on_image_queue` is not, the loader queues the loaded image itself, and the consumer preprocesses it. There, the consumer needs the image anyway, so the metadata can stay where it is. Computing it in the loader in that case too would give one code path, though. Either is fine, but one path is probably easier to maintain.

### RF-DETR

`RFDETRDetector` runs inference on the original image, so its `preprocess_image()` dict legitimately carries the pixels (`rfdetr_detector.py` reads `img['img_original']` during batch inference). Don't drop `img_original` for it. However, its dict also carries `img_original_pil` when the input was a PIL image, and that copy is redundant. It only exists for the consumer-side metadata reads, which go away under this plan. So RF-DETR should queue one copy (the numpy array) instead of two.

The cleanest way to handle the difference is to let each detector's `preprocess_image()` decide what it returns: `PTDetector` returns the processed image and the shape, and `RFDETRDetector` returns the numpy original. The loader then never needs to know which detector it's feeding. PR 256's approach, checking for `'img_original_shape' in image` in the producer, works but is more fragile.

### Video

Video has its own loader, and it has the same problem. `process_video._video_producer_func()` defines a `frame_callback()` that calls `run_detector_batch._preprocess_image()` on each decoded frame and queues the full result dict, original frame included. Frames arrive as numpy arrays, so `img_original_pil` is `None` there, but `img_original` is still a full-resolution copy of every frame. Video produces many more items than a typical image folder, so the cost adds up.

`_run_detector_on_videos_with_queue()` calls `run_detector_batch._consumer_func()` without any of the metadata flags, so the video path never needs image metadata. Once `preprocess_image()` stops returning the original (for `PTDetector`), video gets the fix for free, because it uses the same `_preprocess_image()` wrapper. If the original is instead stripped in the loader, as PR 256 does, `frame_callback()` needs the same change, and PR 256 didn't make it.

Without the queue (`use_image_queue` False), video frames are passed to the detector as raw numpy arrays. `PTDetector.generate_detections_one_batch()` then calls `preprocess_image()` internally and feeds the resulting dicts to `_process_batch_group()`. So the in-process path (images and video) also depends only on `img_original_shape` once `_process_batch_group()` is changed, and it works unchanged if `preprocess_image()` stops returning the originals.

### Other callers

`run_md_and_speciesnet.py` calls `load_and_run_detector_batch()` with `use_image_queue=True`, `preprocess_on_image_queue=True`, and all metadata flags off. `run_tiled_inference.py` also enables `preprocess_on_image_queue`. Both benefit automatically. The MD+SpeciesNet path is probably the most common real-world user of the preprocessing queue.

## Testing

- Output equivalence: compare the `images` list against `main` for images (queue on/off, `preprocess_on_image_queue` on/off, threads and processes, batch size 1 and 8) and for videos. At batch size 8 on the queue, `main` itself is not deterministic run to run, because batches are formed in the order images arrive. Compare against two runs of `main` rather than expecting identical output.
- Metadata: run with `--include_image_size --include_image_timestamp --include_exif_tags all`, and confirm that width, height, datetime, and EXIF fields match `main` exactly. Include images with no EXIF, corrupt EXIF, and EXIF orientation tags. The `gps-tests` folder in the MD test data has useful edge cases.
- RF-DETR: run an RF-DETR model with `preprocess_on_image_queue` to confirm it still gets its original image.
- Performance: a small model (MDv1000-spruce) with `--use_image_queue --preprocess_on_image_queue --batch_size 8` shows the effect most clearly. Large models can be GPU-bound, which hides it. Compare with the no-queue path too: the goal is for the queue to be at least as fast as no queue.
