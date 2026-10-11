# Reporting loader worker failures to the consumer

## Problem

When the image queue or frame queue is enabled, each loader worker (`run_detector_batch._producer_func()` or `process_video._video_producer_func()`) sends `None` to the queue when it finishes, and the consumer (`run_detector_batch._consumer_func()`) returns when it has received one `None` per worker. If a worker raises an uncaught exception, it never sends `None`, so the consumer waits forever in `q.get()`. The worker's traceback is printed, but the job hangs.

Per-image and per-video errors are already caught inside the workers, so in practice the only exceptions that escape are failures during worker setup (`_load_preprocessor()`, which only runs when `preprocess_on_image_queue` is enabled) and bugs.

## Why "send None in a finally block" isn't enough

`None` means "this worker finished all of its items", and the consumer has no way to tell a normal finish from a failure.

- Images: the job would "succeed", and the output file would be missing every image the failed worker hadn't loaded yet. Nothing checks the number of results against the number of images, so this would be silent data loss, which is worse than a hang.
- Videos: the hang just moves. After all the `None`s arrive, `_run_detector_on_videos_with_queue()` waits (`_receive_video_info(wait=True)`) for a metadata record for every video, and the failed worker never sends records for its remaining videos.

## Recommendation

Workers send an explicit failure marker, and the consumer raises when it receives one.

- Worker side: wrap the body of each worker function in `try`/`except Exception`. On failure, print the traceback and put a failure marker on the frame/image queue (not the metadata queue) instead of `None`, then return. The marker should be a dict, e.g. `{'worker_failure': traceback.format_exc(), 'producer_id': producer_id}`. Normal items are lists and `None` means "finished", so a dict is unambiguous. Send the traceback as a string, since exception objects aren't always picklable.
- Consumer side: in `_consumer_func()`, right after `q.get()`, if the item is a dict, raise an exception that includes the worker's ID and traceback.

The existing `except BaseException` handlers in `_run_detector_with_image_queue()` and `_run_detector_on_videos_with_queue()` already stop the remaining workers (`_stop_producers()`) and re-raise, so nothing else is needed. Because `_consumer_func()` is shared, the video path gets the consumer-side change for free, and the consumer raises before the video metadata drain.

## Relationship to the liveness check

A liveness check (a timeout on the consumer's `q.get()`, plus a check for workers that are no longer alive but haven't sent `None`, and a bounded wait in the video metadata drain) would also catch this case, as well as workers that die without running any Python cleanup (e.g. killed by the OOM killer, or a crash in a native decoder). However, it would only catch it after the timeout, and with a generic "worker exited unexpectedly" message. The failure marker reports the actual error immediately. The liveness check is the more general fix; if only one of the two gets done, it should be that one.

## Testing

The hang probe scripts can exercise this by replacing `run_detector_batch._load_preprocessor` with a function that raises, at module level in the probe script (module-level code also runs in spawned worker processes), and running with `preprocess_on_image_queue` enabled. This only fails in the workers, since the consumer loads the model with `load_detector()`. Check images and videos, with threads and processes, on Windows and Linux: the job should exit promptly with the worker's error, and leave no worker processes behind.
