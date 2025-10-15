# MegaDetector Bug Review Progress

## Instructions for Agent Sessions

**Objective**: Review all .py files in the megadetector folder for obvious bugs that can be assessed from within a module and maybe one level deep in the functions it calls.

**Review Approach**:
- Focus on **local bugs** within each file
- Check signatures of functions called from other modules (one level deep)
- Look for things like:
  - Missing null/None checks before dereferencing
  - Off-by-one errors in loops/array indexing
  - Type errors or incorrect type assumptions
  - Resource leaks (unclosed files, etc.)
  - Logic errors in conditionals
  - Incorrect exception handling
  - Division by zero possibilities
  - Other obvious issues that might have escaped testing
- **Important**: Do NOT flag assertions as bugs. Assertions represent intentional assumptions and constraints. If the code asserts something (e.g., `assert len(fields) == 7`), the author is deliberately imposing that limitation and is aware of the consequences.
- **Important**: Do NOT flag issues in code within `if False:` blocks. These are interactive driver sections that are intentionally disabled and not meant to execute.

**Process**:
1. Ask user if they want to review the next unchecked file
2. If yes, review the file and report findings to the user
3. Wait for user decision (fix or skip)
4. Check off the file in this list
5. Continue to next file

**Checklist notation**:
- `[ ]` = Not yet reviewed
- `[x]` = Reviewed (completed)
- `[SKIPPED]` = Skipped (not reviewed)

**Exclusions**:
- `__init__.py` files are excluded
- `tests/` folder is excluded

---

## Review Checklist

### api/batch_processing/integration/digiKam
- [SKIPPED] megadetector/api/batch_processing/integration/digiKam/setup.py
- [SKIPPED] megadetector/api/batch_processing/integration/digiKam/xmp_integration.py

### api/batch_processing/integration/eMammal/test_scripts
- [SKIPPED] megadetector/api/batch_processing/integration/eMammal/test_scripts/config_template.py
- [SKIPPED] megadetector/api/batch_processing/integration/eMammal/test_scripts/push_annotations_to_emammal.py
- [SKIPPED] megadetector/api/batch_processing/integration/eMammal/test_scripts/select_images_for_testing.py

### classification
- [SKIPPED] megadetector/classification/aggregate_classifier_probs.py
- [SKIPPED] megadetector/classification/analyze_failed_images.py
- [SKIPPED] megadetector/classification/cache_batchapi_outputs.py
- [SKIPPED] megadetector/classification/create_classification_dataset.py
- [SKIPPED] megadetector/classification/crop_detections.py
- [SKIPPED] megadetector/classification/csv_to_json.py
- [SKIPPED] megadetector/classification/detect_and_crop.py
- [SKIPPED] megadetector/classification/evaluate_model.py
- [SKIPPED] megadetector/classification/identify_mislabeled_candidates.py
- [SKIPPED] megadetector/classification/json_to_azcopy_list.py
- [SKIPPED] megadetector/classification/json_validator.py
- [SKIPPED] megadetector/classification/map_classification_categories.py
- [SKIPPED] megadetector/classification/merge_classification_detection_output.py
- [SKIPPED] megadetector/classification/prepare_classification_script.py
- [SKIPPED] megadetector/classification/prepare_classification_script_mc.py
- [SKIPPED] megadetector/classification/run_classifier.py
- [SKIPPED] megadetector/classification/save_mislabeled.py
- [SKIPPED] megadetector/classification/train_classifier.py
- [SKIPPED] megadetector/classification/train_classifier_tf.py
- [SKIPPED] megadetector/classification/train_utils.py

### classification/efficientnet
- [SKIPPED] megadetector/classification/efficientnet/model.py
- [SKIPPED] megadetector/classification/efficientnet/utils.py

### data_management
- [x] megadetector/data_management/animl_to_md.py
- [x] megadetector/data_management/camtrap_dp_to_coco.py
- [x] megadetector/data_management/cct_json_utils.py
- [x] megadetector/data_management/cct_to_md.py
- [x] megadetector/data_management/cct_to_wi.py
- [x] megadetector/data_management/coco_to_labelme.py
- [x] megadetector/data_management/coco_to_yolo.py
- [x] megadetector/data_management/generate_crops_from_cct.py
- [x] megadetector/data_management/get_image_sizes.py
- [x] megadetector/data_management/labelme_to_coco.py
- [x] megadetector/data_management/labelme_to_yolo.py
- [x] megadetector/data_management/mewc_to_md.py
- [x] megadetector/data_management/ocr_tools.py
- [x] megadetector/data_management/read_exif.py
- [x] megadetector/data_management/remap_coco_categories.py
- [x] megadetector/data_management/remove_exif.py
- [x] megadetector/data_management/rename_images.py
- [x] megadetector/data_management/resize_coco_dataset.py
- [x] megadetector/data_management/speciesnet_to_md.py
- [x] megadetector/data_management/wi_download_csv_to_coco.py
- [x] megadetector/data_management/yolo_output_to_md_output.py
- [x] megadetector/data_management/yolo_to_coco.py
- [x] megadetector/data_management/zamba_to_md.py

### data_management/annotations
- [SKIPPED] megadetector/data_management/annotations/annotation_constants.py

### data_management/databases
- [x] megadetector/data_management/databases/add_width_and_height_to_db.py
- [x] megadetector/data_management/databases/combine_coco_camera_traps_files.py
- [x] megadetector/data_management/databases/integrity_check_json_db.py
- [x] megadetector/data_management/databases/subset_json_db.py

### data_management/lila
- [SKIPPED] megadetector/data_management/lila/create_lila_blank_set.py
- [SKIPPED] megadetector/data_management/lila/create_lila_test_set.py
- [SKIPPED] megadetector/data_management/lila/create_links_to_md_results_files.py
- [x] megadetector/data_management/lila/download_lila_subset.py
- [x] megadetector/data_management/lila/generate_lila_per_image_labels.py
- [SKIPPED] megadetector/data_management/lila/get_lila_annotation_counts.py
- [SKIPPED] megadetector/data_management/lila/get_lila_image_counts.py
- [x] megadetector/data_management/lila/lila_common.py
- [SKIPPED] megadetector/data_management/lila/test_lila_metadata_urls.py

### detection
- [x] megadetector/detection/change_detection.py
- [x] megadetector/detection/process_video.py
- [x] megadetector/detection/pytorch_detector.py
- [x] megadetector/detection/run_detector.py
- [x] megadetector/detection/run_detector_batch.py
- [x] megadetector/detection/run_inference_with_yolov5_val.py
- [x] megadetector/detection/run_md_and_speciesnet.py
- [x] megadetector/detection/run_tiled_inference.py
- [x] megadetector/detection/tf_detector.py
- [x] megadetector/detection/video_utils.py

### postprocessing
- [x] megadetector/postprocessing/add_max_conf.py
- [x] megadetector/postprocessing/categorize_detections_by_size.py
- [x] megadetector/postprocessing/classification_postprocessing.py
- [x] megadetector/postprocessing/combine_batch_outputs.py
- [x] megadetector/postprocessing/compare_batch_results.py
- [x] megadetector/postprocessing/convert_output_format.py
- [x] megadetector/postprocessing/create_crop_folder.py
- [x] megadetector/postprocessing/detector_calibration.py
- [x] megadetector/postprocessing/generate_csv_report.py
- [x] megadetector/postprocessing/load_api_results.py
- [x] megadetector/postprocessing/md_to_coco.py
- [x] megadetector/postprocessing/md_to_labelme.py
- [x] megadetector/postprocessing/md_to_wi.py
- [x] megadetector/postprocessing/merge_detections.py
- [x] megadetector/postprocessing/postprocess_batch_results.py
- [x] megadetector/postprocessing/remap_detection_categories.py
- [x] megadetector/postprocessing/render_detection_confusion_matrix.py
- [x] megadetector/postprocessing/separate_detections_into_folders.py
- [x] megadetector/postprocessing/subset_json_detector_output.py
- [x] megadetector/postprocessing/top_folders_to_bottom.py
- [x] megadetector/postprocessing/validate_batch_results.py

### postprocessing/repeat_detection_elimination
- [x] megadetector/postprocessing/repeat_detection_elimination/find_repeat_detections.py
- [x] megadetector/postprocessing/repeat_detection_elimination/remove_repeat_detections.py
- [x] megadetector/postprocessing/repeat_detection_elimination/repeat_detections_core.py

### taxonomy_mapping
- [SKIPPED] megadetector/taxonomy_mapping/map_lila_taxonomy_to_wi_taxonomy.py
- [SKIPPED] megadetector/taxonomy_mapping/map_new_lila_datasets.py
- [SKIPPED] megadetector/taxonomy_mapping/prepare_lila_taxonomy_release.py
- [SKIPPED] megadetector/taxonomy_mapping/preview_lila_taxonomy.py
- [SKIPPED] megadetector/taxonomy_mapping/retrieve_sample_image.py
- [SKIPPED] megadetector/taxonomy_mapping/simple_image_download.py
- [SKIPPED] megadetector/taxonomy_mapping/species_lookup.py
- [SKIPPED] megadetector/taxonomy_mapping/taxonomy_csv_checker.py
- [SKIPPED] megadetector/taxonomy_mapping/taxonomy_graph.py
- [SKIPPED] megadetector/taxonomy_mapping/validate_lila_category_mappings.py

### utils
- [x] megadetector/utils/ct_utils.py
- [x] megadetector/utils/directory_listing.py
- [x] megadetector/utils/extract_frames_from_video.py
- [x] megadetector/utils/gpu_test.py
- [x] megadetector/utils/md_tests.py
- [x] megadetector/utils/path_utils.py
- [x] megadetector/utils/process_utils.py
- [x] megadetector/utils/split_locations_into_train_val.py
- [x] megadetector/utils/string_utils.py
- [x] megadetector/utils/url_utils.py
- [SKIPPED] megadetector/utils/wi_platform_utils.py
- [x] megadetector/utils/wi_taxonomy_utils.py
- [x] megadetector/utils/write_html_image_list.py

### visualization
- [x] megadetector/visualization/plot_utils.py
- [x] megadetector/visualization/render_images_with_thumbnails.py
- [x] megadetector/visualization/visualization_utils.py
- [x] megadetector/visualization/visualize_db.py
- [x] megadetector/visualization/visualize_detector_output.py
- [x] megadetector/visualization/visualize_video_output.py

---

## Review Notes

### megadetector/data_management/animl_to_md.py
- **FIXED**: Line 70 - Type checking bug where `isinstance(row['category'], int)` would fail when pandas reads the column as float64 (e.g., 1.0, 2.0, 3.0). Changed to check if value is numeric and represents an integer using `pd.notna()` and `float().is_integer()`. Also improved error message to include row number.
- Noted but not fixed: Missing NaN checks for detection_conf, bbox values, classification_conf, and class name could allow invalid data in output JSON

### megadetector/data_management/camtrap_dp_to_coco.py
- Line 129: `isinstance(row['scientificName'], str)` would fail if scientificName column has NaN values (pandas treats as float)
- Line 109: Missing NaN check before dictionary lookup - if mediaID is NaN, assertion will fail confusingly
- Line 99: Date parsing without validation - if timestamp is NaN or invalid, could fail or return None
- Line 125: Human observation validation doesn't handle NaN scientificName
- User chose not to fix these issues

### megadetector/data_management/cct_json_utils.py
- **FIXED**: Lines 234-235 - Redundant assignment where `self.image_id_to_annotations = {}` was immediately overwritten by `self.image_id_to_annotations = defaultdict(list)`. Removed the first line.
- Line 62: Missing KeyError handling when category_id not in cat_id_to_name
- Line 289: Same KeyError issue with cat_id_to_name lookup
- Line 111: Will raise KeyError if requested field doesn't exist in image entry
- Line 116: Will raise KeyError if annotation references non-existent image_id
- Line 543: Redundant assertion (always True) in sequence processing
- User chose not to fix issues 2-6

### megadetector/data_management/cct_to_md.py
- **FIXED** (by user): Line 155 - Return value mismatch where function returned `output_filename` (string) instead of `results` (dict) as documented. Now returns `results`.
- **FIXED** (by user): Line 67 - Misleading error message said "Cannot find category" but now says "Cannot find key"
- Lines 129-130: Division by zero risk if image_w or image_h is 0
- Lines 100, 103-104: Missing field checks for 'location', 'height', 'width' - will raise KeyError if missing
- User chose not to fix remaining issues

### megadetector/data_management/cct_to_wi.py
- **FIXED** (by user): Line 267 - Moved assertion to separate line for better style (was inline with assignment)
- Line 253: Missing KeyError handling if category_name not in taxonomy_mapping
- Line 233: Missing KeyError handling if category_id not in category_id_to_name
- User chose not to fix remaining issues
- Note: Several assertions were noted during review but are intentional constraints (e.g., single annotation per image, 7 taxonomy fields)

### megadetector/data_management/coco_to_labelme.py
- Line 76: Missing KeyError handling if category_id not in category_id_to_name
- User chose not to fix

### megadetector/data_management/coco_to_yolo.py
- **FIXED** (by user): Line 84 - Copy-paste error where `f.write('val: ...')` was used instead of `f.write('test: ...')` when writing test folder path
- **FIXED** (by user): Line 538 - NameError when class_file_name is None; now sets `class_list_filename = None` explicitly
- Line 356: Missing KeyError handling if category_id not in coco_id_to_yolo_id
- Lines 375-379: Division by zero risk if img_w or img_h is 0
- User chose not to fix remaining issues

### megadetector/data_management/generate_crops_from_cct.py
- **FIXED** (by user): Line 75 - Redundant nested os.path.join() call removed
- **FIXED** (by user): Lines 107-108 - Off-by-one error in box clipping; changed from `img.width-1` to `img.width` because PIL's crop() uses exclusive upper bounds. Added explanatory comment.
- **FIXED** (by user): Line 115 - Type error fixed by adding str() around ann['id'] for string concatenation

### megadetector/data_management/get_image_sizes.py
- **FIXED**: Lines 91-95 - Directory check failed for output files in current directory because os.path.dirname() returns empty string. Now checks if dirname is non-empty before validating.
- **FIXED** (by user): Line 78 - Typo "imgae" corrected to "image" in docstring
- **FIXED** (by user): Removed interactive driver section that had incorrect function call (wrong parameter name n_threads instead of n_workers)

### megadetector/data_management/labelme_to_coco.py
- **FIXED** (by user): Line 426 - Function parameter `recursive` was being ignored; always passed `recursive=True` to find_images(). Now properly uses the parameter value.
- **FIXED** (by user): Lines 355-358 - Added None check before calling pool.close() and pool.join() in finally block to prevent AttributeError if pool creation fails
- **FIXED** (by user): Line 503 - Typo "sortec_categories" corrected to "sorted_categories" in interactive driver

### megadetector/data_management/labelme_to_yolo.py
- **FIXED** (by user): Lines 255-258 - Added None check before calling pool.close() and pool.join() in finally block to prevent AttributeError if pool creation fails
- **FIXED**: Lines 110-124 - Division by zero when im_width or im_height is 1. Added separate checks for 1-pixel-wide and 1-pixel-tall images; sets corresponding relative coordinates to 0.0 in each case.
- **FIXED** (by user): Removed interactive driver section (if False block) that had incorrect variable assignment

### megadetector/data_management/mewc_to_md.py
- Line 156: Unhandled ValueError if snip_id token cannot be converted to int
- Line 143: Unused variable (ext) with # noqa comment
- Lines 80, 274: Unnecessary del statements for variables going out of scope
- User chose not to fix these issues

### megadetector/data_management/ocr_tools.py
- **FIXED** (by user): Lines 269-277 - Duplicate code block removed (same x/y/w/h assignments repeated twice)
- **FIXED** (by user): Lines 652-656 - Added None check before calling pool.close() and pool.join() in finally block to prevent AttributeError if pool creation fails
- Lines 730-732: Unreachable code after raise Exception in if False block (ignored per review guidelines)

### megadetector/data_management/read_exif.py
- **FIXED** (by user): Line 397 - Wrong variable used in filter; was checking against `options.tags_to_exclude` instead of the lowercase-normalized `tags_to_exclude` list, breaking case-insensitive matching
- **FIXED** (by user): Lines 637-640 - Added None check before calling pool.close() and pool.join() in finally block to prevent AttributeError if pool creation fails
- **FIXED** (by user): Line 780 - Wrong library name check; was checking for 'exif' instead of 'exiftool', so exiftool availability was never validated

### megadetector/data_management/remap_coco_categories.py
- **FIXED** (by user): Line 129 - Wrong return value; was returning `input_data` instead of `output_data` (though they pointed to the same object, this was confusing and likely a copy-paste error)
- Line 78: Missing KeyError handling if input_category_name_to_output_category_name contains a category name not in the input COCO data
- User chose not to fix remaining issue

### megadetector/data_management/remove_exif.py
- **FIXED** (by user): Lines 80-84 - Missing slash in glob pattern; user switched from problematic `glob.glob(image_base_folder + "*/**")` to `recursive_file_list()` for proper file enumeration
- **FIXED** (by user): Lines 101-108 - Added None check before calling pool.close() and pool.join() in finally block to prevent AttributeError if pool creation fails
- Line 71: Bare except clause catches all exceptions including system exits (ignored per user request)

### megadetector/data_management/rename_images.py
- **FIXED** (by user): Lines 31-32 - Incorrect docstring copy-pasted from another function; described COCO/labelme format conversion instead of image renaming. Now correctly describes the image copying/renaming functionality.
- **FIXED** (by user): Line 59 - Duplicate 'DateTime' tag in tags_to_include list (appeared twice); removed duplicate
- **FIXED** (by user): Lines 113-117 - Critical bug where parallel_copy_files() was called inside the for loop, running once per image with a single-file dict each time, completely defeating parallelization. Dedented to run after loop builds complete dictionary.

### megadetector/data_management/resize_coco_dataset.py
- **FIXED** (by user): Lines 239, 260-263 - Added None check and initialization before calling pool.close() and pool.join() in finally block to prevent AttributeError if pool creation fails
- **FIXED**: Lines 68-71 - Directory creation could fail for flat datasets where output_fn_abs has no directory component (os.path.dirname returns empty string). Added length check before makedirs call.
- Lines 127-128: Division by zero if input image has zero width or height (ignored per user request)

### megadetector/data_management/speciesnet_to_md.py
- No bugs found. Simple command-line wrapper with correct function call.

### megadetector/data_management/wi_download_csv_to_coco.py
- Line 126: Missing NaN check for common_name before string operations (ignored per user request)

### megadetector/data_management/yolo_output_to_md_output.py
- **FIXED** (by user): Line 190 - Typo "Duplication image IDs" corrected to "Duplicate image IDs"
- Line 405: IndexError if user forgot to use --save-conf with YOLO (row only has 5 elements, not 6) (ignored per user request)
- Line 401: KeyError if YOLO output contains category IDs outside hardcoded map {0:1, 1:2, 2:3} (ignored per user request)
- Lines 297-300: Division by zero if image has zero width or height (ignored per user request)

### megadetector/data_management/yolo_to_coco.py
- **FIXED** (by user): Lines 202, 217-220 - Added None check and initialization before calling pool.close() and pool.join() in validate_yolo_dataset's finally block
- **FIXED**: Lines 597, 612-616 - Added try/finally block with None check for pool cleanup in yolo_to_coco function's parallel processing section
- **FIXED** (by user): Line 301 - Added isinstance check to avoid TypeError when class_name_file is a list
- **FIXED** (by user): Lines 179-195 - Fixed blank line validation by stripping lines immediately after reading; eliminated redundant class_names variable

### megadetector/data_management/zamba_to_md.py
- Line 126: KeyError if top_k_label doesn't match extracted category names (ignored per user request)
- After line 93: No validation that category_names is non-empty (ignored per user request)
- Lines 123-124: No NaN handling for labels/probabilities from CSV (ignored per user request)

### megadetector/data_management/databases/add_width_and_height_to_db.py
- Line 60: PIL Image not explicitly closed, but becomes eligible for garbage collection immediately (ignored per user request)

### megadetector/data_management/databases/combine_coco_camera_traps_files.py
- **FIXED** (by user): Line 133 - TypeError when seq_id is integer; added str() cast
- **FIXED** (by user): Line 146 - TypeError when im['id'] is integer; added str() cast
- **FIXED** (by user): Line 155 - TypeError when ann['image_id'] is integer; added str() cast
- **FIXED** (by user): Line 156 - TypeError when ann['id'] is integer; added str() cast

### megadetector/data_management/databases/integrity_check_json_db.py
- **FIXED**: Lines 113-115 - PIL Image not closed after getting size; added pil_im.close() call

### megadetector/data_management/databases/subset_json_db.py
- **FIXED**: Lines 154-157 - Directory creation could fail for output files in current directory (os.path.dirname returns empty string). Added length check before makedirs call.

### megadetector/data_management/lila/download_lila_subset.py
- No bugs found. Simple example script with straightforward logic.

### megadetector/data_management/lila/generate_lila_per_image_labels.py
- **FIXED** (by user): Lines 151-153 - Wasteful loop where category_id_to_name dict was recreated identically for each category (outer loop variable shadowed by comprehension). Removed pointless loop.
- **FIXED** (by user): Line 738 - Type conversion error where int() would fail on string "3.0" from CSV. Added float() conversion first: int(float(row['frame_num'])).

### megadetector/data_management/lila/lila_common.py
- **FIXED** (by user): Line 117 - Wrong file being read; was reading from URL instead of local taxonomy_filename downloaded on line 114. Changed to read from local file.
- **FIXED** (by user): Line 68 - force_download parameter ignored; if CSV existed, function returned without checking force_download. Added check: `if os.path.exists(wi_taxonomy_csv_path) and (not force_download)`.

### megadetector/detection/change_detection.py
- **FIXED** (by user): Line 473 - Extra quotes in print statement would print literal quote marks around error message. Removed extra quotes.
- **FIXED**: Lines 831-832 - Division by zero if no images were processed (total_images == 0). Added check: `if total_images > 0:` before percentage calculation.

### megadetector/detection/process_video.py
- **FIXED** (by user): Line 306 - Missing space in command line construction; would concatenate with previous argument. Added leading space: `cmd += ' --detector_options {}'`.

### megadetector/detection/pytorch_detector.py
- **FIXED** (by user): Lines 114-117 - Logic error in model type preference; when `prefer_model_type_source == 'table'`, was using file metadata instead of table. Swapped: table → model_type_from_model_version, file → model_type_from_model_file_metadata.
- **FIXED** (by user): Line 376 - Typo in error message: "is not installed, but . " → "is not installed. "
- **FIXED** (by user): Lines 1316-1318 - Invalid use of print() in assert statement; print() returns None which would cause assert to always fail. Changed to use string message in assert.

### megadetector/detection/run_detector.py
- **FIXED**: Lines 869-873 - Incomplete zip validation; testzip() return value was not checked. Now checks if corrupt_file is not None and prints error with corrupt file name before returning False.
- **FIXED**: Lines 761-765 - KeyError in collision handling; after modifying fn with prefix on line 764, line 765 tried to increment that prefixed key which doesn't exist in dict. Now saves original fn before modifying it.

### megadetector/detection/run_detector_batch.py
- **FIXED** (by user): Lines 890-902 - Missing return statement in _process_images when use_image_queue is True; was calling _run_detector_with_image_queue but not returning its results. Now assigns to results and returns it.
- Line 2125: Division by zero if elapsed is 0 (theoretically possible if inference is instantaneous) (ignored per user request).

### megadetector/detection/run_inference_with_yolov5_val.py
- **FIXED**: Line 354 - Directory creation could fail for output files in current directory (os.path.dirname returns empty string). Added length check before makedirs call.
- Lines 711-713: IndexError if category list is empty; accessing category_ids[0] and category_ids[-1] without checking if list is non-empty (ignored per user request).

### megadetector/detection/run_md_and_speciesnet.py
- **FIXED** (by user): Line 79 - Typo in constant name: DEAFULT_SECONDS_PER_VIDEO_FRAME → DEFAULT_SECONDS_PER_VIDEO_FRAME. Propagated to lines 1268, 1274, 1311, 1470.
- **FIXED** (by user): Line 579 - Incorrect logic for enable_rollup check; was checking `enable_rollup is not None` for a boolean parameter, causing ensemble to load even when rollup disabled. Changed to `if enable_rollup or (country is not None)`.

### megadetector/detection/run_tiled_inference.py
- **FIXED** (by user): Line 513 - Orphaned string that should have been part of a ValueError; changed to proper `raise ValueError(...)`.
- **FIXED** (by user): Line 624/626 - Wrong variable used in nested loop; was using undefined `im` instead of loop variable `patch_info`. Changed line 624 to use `patch_info['error']` and line 626 to use `patch_info['patches']`.
- **FIXED** (by user): Line 955 - Wrong default value for tile_size_y; was using `default_tile_size[0]` (x dimension) instead of `default_tile_size[1]` (y dimension).

### megadetector/detection/tf_detector.py
- **FIXED** (by user): Line 45 - Typo in docstring: "path to .pdb file" corrected to "path to .pb file" (protobuf model file, not Python debugger file).
- Line 51: TF session never closed; no cleanup method or destructor provided (ignored per user request).
- Lines 52-55: No error handling for missing tensors when loading graph (ignored per user request).

### megadetector/detection/video_utils.py
- **FIXED**: Line 224 - Directory creation could fail for output files in current directory (os.path.dirname returns empty string). Added length check before makedirs call.
- **FIXED** (by user): Line 228 - Debug code `cv2.imshow('video',frame)` was left in; would cause problems in headless environments. Commented out.
- **FIXED** (by user): Line 300 - Typo in error message: "Filename {} does contain a valid frame number" corrected to "does not contain".
- **FIXED** (by user): Lines 1062-1064 - Pool cleanup without None check in finally block. Added `if pool is not None:` check before calling pool.close() and pool.join().

### megadetector/postprocessing/add_max_conf.py
- **FIXED**: Lines 49-51 - Directory creation could fail if output directory doesn't exist. Added check for output directory and creation with makedirs before writing output file.

### megadetector/postprocessing/categorize_detections_by_size.py
- **FIXED**: Lines 160-163 - Directory creation could fail if output directory doesn't exist. Added check for output directory and creation with makedirs before writing output file.
- Line 80: max() of empty sequence if category_keys is empty (ignored per user request).
- Line 106: Potential KeyError if malformed data has None detections without a failure key (ignored per user request).

### megadetector/postprocessing/classification_postprocessing.py
- **FIXED** (by user): Line 451 - Wrong variable used in debug print; was using `classification_descriptions[c[1]]` instead of `most_common_category` variable that was already defined.
- **FIXED** (by user): Line 423 - Debug code `from IPython import embed; embed()` left in production code; user commented it out.
- **FIXED**: Lines 922-924, 1099-1101, 1691-1693 - Missing directory creation for three output files. Added `import os` at line 16, then added checks for output directory existence before writing files in smooth_classification_results_image_level(), smooth_classification_results_sequence_level(), and restrict_to_taxa_list().

### megadetector/postprocessing/combine_batch_outputs.py
- **FIXED** (by user): Line 206-207 - Resource leak where file opened with `json.load(open(fn))` was never closed. Changed to use context manager `with open(fn,'r') as f:`.
- **FIXED** (by user): Line 218 - Inefficient code using `detections.extend([d])` to add single element. Changed to `detections.append(d)`.

### megadetector/postprocessing/compare_batch_results.py
- **FIXED**: Lines 355-360 - Length mismatch between gt_boxes and gt_categories; gt_boxes only included annotations with bboxes, but gt_categories included ALL annotations, causing assertion failure on line 367. Changed to initialize both as empty lists and append to both only when 'bbox' is present in annotation.
- **FIXED** (by user): Lines 477, 489 - Typos in comments: "mistakse" corrected to "mistakes".
- **FIXED** (by user): Line 817 - Wrong variable used; was using gt_filenames (list) instead of gt_filenames_set (set) for membership check.
- **FIXED** (by user): Lines 1680-1681 - Orphaned string literal not part of assertion; fixed with backslash continuation to make string part of assert message.
- **FIXED** (by user): Lines 1183-1193 - Duplicate code block removed.
- **FIXED** (by user): Lines 678-687 - Missing None check before iterating options.category_names_to_include; added check to populate all category IDs when None, otherwise filter by specified categories.

### megadetector/postprocessing/convert_output_format.py
- **FIXED** (by user): Line 180-181 - Wrong variable used for classification confidence; was storing detection confidence (d['conf']) instead of classification_conf when updating max.
- **FIXED** (by user): Line 213 - Unused output_encoding parameter; now properly passed to df.to_csv().
- **FIXED** (by user): Line 298 - Resource leak where json.dump opened file without context manager; changed to use write_json() utility function.
- **FIXED** (by user): Line 375 - Typo in help text; changed "Output" to "Omit" to match parameter name omit_bounding_boxes.

### megadetector/postprocessing/create_crop_folder.py
- **FIXED**: Lines 172-174 - Directory creation could fail for output files in current directory (os.path.dirname returns empty string). Added length check before makedirs call.
- **FIXED**: Lines 350-352 - Same directory creation issue in create_crop_folder function. Added length check before makedirs call.
- **FIXED** (by user): Lines 262-264 - n_skipped_detections variable was initialized but never incremented. Added increment when skip_detection is True.
- **FIXED** (by user): Line 606 - Missing colon in print statement for consistency with surrounding lines. Changed "Input image folder {}" to "Input image folder: {}".

### megadetector/postprocessing/detector_calibration.py
- Lines 530-531: Division by zero when plotting categories with no samples (samples_this_category empty). Would create weights arrays with len(x)=0 causing division by zero. User chose to skip.

### megadetector/postprocessing/generate_csv_report.py
- **FIXED** (by user): Lines 253-257 - NameError when datetime_source not provided; datetime_string was undefined if filename_to_datetime_string was None. Added initialization: `datetime_string = ''` before the conditional block.
- **FIXED** (by user): Lines 387-392 - IndexError if no output records generated; accessing output_records[0] would fail on empty list. Added check for `len(output_records) == 0` with warning message.
- **FIXED** (by user): Lines 394-397 - Missing output directory creation; added check for output directory existence with length check before makedirs call.

### megadetector/postprocessing/load_api_results.py
- **FIXED** (by user): Lines 142-146 - Missing output directory creation in write_api_results(); added directory creation check before writing. Also replaced json.dump with write_json() utility function.
- Line 181: No error handling for json.loads() in load_api_results_csv() when deserializing detections; would fail on NaN or malformed JSON. User chose not to fix (deprecated function).
- **FIXED** (by user): Lines 217-221 - Missing output directory creation in write_api_results_csv(); added directory creation check before writing CSV.

### megadetector/postprocessing/md_to_coco.py
- Lines 184-186: PIL image not explicitly closed after reading dimensions; resource leak if many images accumulate before garbage collection. User chose to skip (images are eligible for GC immediately, only issue if processing very large batches before GC runs).
- **FIXED** (by user): Line 299 - Missing output directory creation; changed to use write_json() utility function which handles directory creation. Also updated write_json() in ct_utils.py (lines 244-247) to include directory creation for all future callers.

### megadetector/postprocessing/md_to_labelme.py
- **FIXED** (by user): Line 63 - Logic error with inverted condition; was checking `if category_id_to_name:` which set default ONLY when parameter was provided. Changed to `if category_id_to_name is None:` to properly set default when not provided.
- **FIXED** (by user): Line 141 - Missing output directory creation; changed to use write_json() utility function which handles directory creation.
- Line 219: PIL image not explicitly closed after reading dimensions; same resource leak issue as md_to_coco.py. User chose not to fix.
- **FIXED** (by user): Lines 259-262 - Pool cleanup without None check; added `if pool is not None:` before calling pool.close() and pool.join().

### megadetector/postprocessing/md_to_wi.py
- **FIXED** (by user): Line 37 - Command-line argument ignored; was passing hardcoded `base_folder=None` instead of `base_folder=args.base_folder`, causing user's --base_folder argument to be ignored.

### megadetector/postprocessing/merge_detections.py
- **FIXED** (by user): Line 124/293 - Directory creation issue; removed explicit os.makedirs() call that would fail for files without directory component. Changed line 293 to use write_json() which handles directory creation internally.
- **FIXED** (by user): Line 310 - Typo in help text; "reuslts" corrected to "results".
- **FIXED** (by user): Line 362 - Wrong help text; --categories_to_exclude parameter had help text saying "to include" instead of "to exclude".

### megadetector/postprocessing/postprocess_batch_results.py
- **FIXED** (by user): Lines 1907-1911 - Logic error in classification category counting; was initializing new categories to 0 instead of 1, causing all counts to be off by one (too low). Changed line 1908 from `= 0` to `= 1`.
- Lines 1252-1255: Division by zero in precision/recall/F1 calculation if no predictions at threshold, no ground truth positives, or both precision and recall are 0. User chose to skip.
- Line 470: PIL image not explicitly closed after reading; resource leak if many images accumulate before garbage collection. User chose to skip (same reasoning as previous PIL image issues).

### megadetector/postprocessing/remap_detection_categories.py
- **FIXED** (by user): Line 137 - Missing output directory creation; changed to use write_json() utility function which handles directory creation.

### megadetector/postprocessing/render_detection_confusion_matrix.py
- **FIXED** (by user): Lines 255-258 - Pool cleanup without None check in finally block; added `if pool is not None:` check before calling pool.close() and pool.join().
- **FIXED** (by user): Lines 384-396 - Critical indentation bug where lines 384-389 were inside the detection loop instead of after it. This caused predicted_category_name to be set repeatedly during loop instead of once after processing all detections. Dedented lines 391-396 to run after the loop completes.

### megadetector/postprocessing/separate_detections_into_folders.py
- **FIXED** (by user): Line 497 - Resource leak where file opened with `json.load(open())` was never closed. Changed to use context manager.
- **FIXED** (by user): Lines 620-627 - Pool cleanup without finally block; wrapped pool operations in try/finally to ensure pool.close() and pool.join() are called even if exception occurs.
- **FIXED** (by user): Line 739 - Wrong default value for box_expansion argument; was using `default_line_thickness` (8) instead of `default_box_expansion` (3). Changed to correct default.

### megadetector/postprocessing/subset_json_detector_output.py
- **FIXED** (by user): Line 436 - Missing return value in subset_json_detector_output_by_list(); was returning None instead of data when options.keep_files_in_list is None, breaking the chain of subsetting operations.

### megadetector/postprocessing/top_folders_to_bottom.py
- **FIXED** (by user): Lines 169-176 - Pool cleanup without try/finally block; wrapped pool operations in try/finally to ensure pool.close() and pool.join() are called even if exception occurs.
- **FIXED** (by user): Lines 48-53, 68, 217-222 - Command-line argument --overwrite ignored; added overwrite parameter to __init__ method and passed args.overwrite when creating options object.

### megadetector/postprocessing/validate_batch_results.py
- **FIXED** (by user): Line 42 - Typo in docstring; "validate_bach_results" corrected to "validate_batch_results".
- Lines 178-186: Inconsistent error handling using assert statements instead of raise ValueError() like the rest of the function; assertions can be disabled with python -O. User chose to skip.
- Line 328: No output from command-line driver; validation results are never printed or reported to user. User chose to skip.

### megadetector/postprocessing/repeat_detection_elimination/find_repeat_detections.py
- No bugs found. Simple command-line driver with straightforward argparse setup.

### megadetector/postprocessing/repeat_detection_elimination/remove_repeat_detections.py
- **FIXED** (by user): Line 40 - Logic error where filtering_dir was asserted to be a directory on line 40, but then checked if it was a file on line 42, making that code path unreachable. Changed assertion to check os.path.exists() instead of os.path.isdir(), allowing filtering_dir to be either a file or directory as intended.

### megadetector/postprocessing/repeat_detection_elimination/repeat_detections_core.py
- **FIXED** (by user): Line 872 - Incomplete bounding box comparison; was comparing only first 3 elements `[0:3]` (x, y, width) instead of all 4 elements (x, y, width, height). Changed to `[0:4]` to compare complete bounding box.
- **FIXED** (by user): Line 1152 - Resource leak where `open(detection_index_file_name, 'r').read()` opened file without context manager. Changed to use `with open(...) as f:` to ensure file is closed.
- **FIXED** (by user): Line 1385 - Resource leak where `open(candidate_detection_file, 'r').read()` opened file without context manager. Changed to use `with open(...) as f:` to ensure file is closed.

### megadetector/utils/ct_utils.py
- **FIXED** (by user): Line 568 - Incomplete docstring; missing closing bracket in "sorted copy of [d". Added closing bracket to make it "sorted copy of [d]".
- **FIXED** (by user): Lines 1028-1029 - Incorrect tuple definition in parse_bool_string(); `false_strings = ('false')` and `true_strings = ('true')` were strings not tuples (missing trailing commas), causing substring matching instead of exact matching. Changed to `('false',)` and `('true',)` with trailing commas to create proper single-element tuples.

### megadetector/utils/directory_listing.py
- No bugs found. Clean HTML directory listing generator with proper context manager usage.

### megadetector/utils/extract_frames_from_video.py
- No bugs found. Clean frame extraction utility with proper error checking and resource management.

### megadetector/utils/gpu_test.py
- **FIXED** (by user): Line 37 - Missing return value in torch_test(); was returning None when PyTorch import failed, but docstring specified return type int. Changed to return 0.
- **FIXED** (by user): Line 74 - Incorrect docstring in tf_test(); said "The number of CUDA devices reported by PyTorch" instead of "TensorFlow" (copy-paste error). Corrected to say "TensorFlow".
- **FIXED** (by user): Line 82 - Missing return value in tf_test(); was returning None when TensorFlow import failed, but docstring specified return type int. Changed to return 0.

### megadetector/utils/md_tests.py
- **FIXED** (by user): Line 389 - Copy-paste error in output_files_are_identical(); was comparing `len(fn1_results['images'])` to itself instead of to `len(fn2_results['images'])`, causing comparison to always succeed. Changed second reference to fn2_results to properly detect file count mismatches.
- **FIXED** (by user): Lines 1272-1285 - Duplicate test block removed and replaced with different test variation. Was exact duplicate of lines 1256-1269 (running with image queue and worker-side preprocessing), changed to test image queue WITHOUT worker-side preprocessing, writing to inference_output_file_no_preprocess_queue.

### megadetector/utils/path_utils.py (lines 1-1634)
- **FIXED** (by user): Line 154 - Redundant assignment; `folders = []` was initialized on line 152, then immediately reassigned to `[]` again at the start of the recursive block. Removed redundant assignment.
- **FIXED**: Lines 372-374 - Directory creation issue in safe_create_link(); `os.path.dirname(link_new)` returns empty string for files in current directory, causing makedirs to fail or create incorrect structure. Added check `if len(link_new_dir) > 0:` before makedirs.
- **FIXED**: Lines 992-994 - Directory creation issue in _copy_file(); `os.path.dirname(target_fn)` returns empty string for files in current directory. Added check `if len(target_dir) > 0:` before makedirs.
- **FIXED** (by user): Lines 1040-1043 - Pool cleanup without None check in parallel_copy_files(); if pool creation failed, finally block would crash trying to call close()/join() on None. Added `if pool is not None:` check.
- **FIXED** (by user): Lines 1103-1118 - Missing pool cleanup in parallel_delete_files(); created pool but never called close() or join(), leaking resources. Wrapped in try/finally with None check.
- **FIXED** (by user): Line 1214 - Type error in parallel_get_file_sizes(); `len(filenames)` was called at line 1188 before checking if filenames was a string. If filenames was a folder path string, would calculate wrong n_workers. Moved n_workers calculation to after filenames type check and conversion.
- **FIXED** (by user): Lines 1219-1244 - Missing pool cleanup in parallel_get_file_sizes(); created pool but never closed or joined it. Wrapped in try/finally with None check.
- **FIXED** (by user): Line 1412 - Inconsistent return value in zip_folder(); returned `None` instead of `output_fn` when file exists and not overwriting. All other branches return `output_fn`, breaking docstring contract. Changed to `return output_fn`.
- **FIXED** (by user): Lines 1455-1473 - Missing pool cleanup in parallel_zip_files(); created pool but never closed or joined it. Wrapped in try/finally.
- **FIXED** (by user): Lines 1500-1519 - Missing pool cleanup in parallel_zip_folders(); created pool but never closed or joined it. Wrapped in try/finally.
- **FIXED** (by user): Lines 1669-1684 - Missing pool cleanup in parallel_compute_file_hashes(); created pool but never closed or joined it. Wrapped in try/finally.

### megadetector/utils/process_utils.py
- No bugs found. Clean process execution utility with proper resource management.

### megadetector/utils/split_locations_into_train_val.py
- **FIXED** (by user): Line 224 - Dead code; `random_seed = min_error_seed` was assigned but never used afterwards. Removed unused assignment.
- **FIXED** (by user): Lines 226-229 - Redundant sort operation; `category_to_val_fraction` was sorted by its own values (val fractions), then immediately re-sorted by `category_id_to_count`, completely replacing the first sort's ordering. Removed the first redundant sort.

### megadetector/utils/string_utils.py
- No bugs found. Clean string utilities with proper error handling and comprehensive test coverage.

### megadetector/utils/url_utils.py
- **FIXED** (by user): Lines 134-136 - Directory creation issue in download_url(); `os.path.dirname(destination_filename)` returns empty string for files in current directory. Added check `if len(target_dir) > 0:` before makedirs.

### megadetector/utils/wi_taxonomy_utils.py
- **FIXED** (by user): Lines 346-364 - IndexError risk in generate_whole_image_detections_for_classifications(); while loop incremented i_score without bounds checking when ignoring blank classifications. If all classifications were blank/no_cv_result, i_score would exceed list length causing IndexError. Added check inside loop: if i_score >= len(classes), print warning, reset i_score to 0, and break.
- **FIXED** (by user): Line 477 - Missing parameter in round_floats_in_nested_dict() call; was passing `max_decimals` as positional argument but function requires named parameter `decimal_places=max_decimals`.
- **FIXED**: Line 740-742 - Directory creation issue in generate_predictions_json_from_md_results(); `os.path.dirname(predictions_json_file)` returns empty string for files in current directory. Added check `if len(output_dir) > 0:` before makedirs.
- **FIXED**: Lines 816-818 - Directory creation issue in generate_instances_json_from_folder(); `os.path.dirname(output_file)` returns empty string for files in current directory. Added check `if len(output_dir) > 0:` before makedirs.
- **FIXED**: Lines 900-902 - Directory creation issue in merge_prediction_json_files(); `os.path.dirname(output_prediction_file)` returns empty string for files in current directory. Added check `if len(output_dir) > 0:` before makedirs.

### megadetector/utils/write_html_image_list.py
- **FIXED** (by user): Line 188 - Malformed HTML structure; was missing `<head>` and `</head>` tags around title element. Changed from `'<html>{}<body>\n'` to `'<html><head>{}</head><body>\n'` to match structure at line 138.
- **FIXED** (by user): Line 113 - Dead code; variable `text_style` was assigned but never used (line 114 directly used `options['defaultTextStyle']`). Removed unused assignment.

### megadetector/visualization/plot_utils.py
- **FIXED** (by user): Lines 129-133 - Wrong parameter names in matplotlib set() call; was using `x_label`, `y_label`, `x_lim`, `y_lim` with underscores, but matplotlib expects `xlabel`, `ylabel`, `xlim`, `ylim` without underscores. Removed broken try/except block and kept correct method calls.
- **FIXED** (by user): Lines 173-176 - Missing None check for series_labels parameter; function signature allows `series_labels=None` but line 178 accessed `series_labels[i]` without checking. Added check: `if series_labels is None:` to generate default label.
- **FIXED**: Line 186 - Wrong tick labels for sparse ticks; when setting ticks at positions [0, 20, 40, ...], was passing all col_labels instead of every 20th element. Matplotlib only uses first N labels where N = number of ticks, causing tick at position 20 to get col_labels[1] instead of col_labels[20]. Changed to `col_labels[::20]` to properly slice.

### megadetector/visualization/render_images_with_thumbnails.py
- Lines 105-106: Missing empty list check; accesses `secondary_image_bounding_box_list[0]` without checking if list is empty, would cause IndexError. User chose to skip (function implicitly requires at least one secondary image).
- **FIXED** (by user): Lines 188-190 - Missing output directory creation; `output_image.save(output_image_filename)` would fail with FileNotFoundError if parent directory doesn't exist. Added check for directory length before makedirs.

### megadetector/visualization/visualization_utils.py
- **FIXED** (by user): Line 1275 - Invalid use of print() in assert statement; `assert ..., print('Illegal crop size...')` would always fail with "AssertionError: None" because print() returns None. Removed print() call and used string directly.
- **FIXED** (by user): Lines 1394-1397 - Syntax error in _resize_absolute_image(); had `status:'status'` (swapped key and value) instead of `'status':status` in return dict. Fixed to proper dict syntax.
- **FIXED** (by user): Lines 1486-1489 - Missing None check before pool cleanup in resize_images(); added `if pool is not None:` check in finally block before calling pool.close() and pool.join().
- **FIXED** (by user): Lines 1689-1695 - Missing pool cleanup in parallel_get_image_sizes(); wrapped pool.imap() in try/finally block with pool.close() and pool.join() to prevent resource leak.

### megadetector/visualization/visualize_db.py
- **FIXED** (by user): Line 153 - Wrong path separator constant; was using `os.pathsep` (path entry separator like ':' or ';') instead of `os.sep` (directory separator like '/' or '\\'). The pathsep_replacement feature now works correctly.
- **FIXED** (by user): Lines 315-317 - Double base directory join; was calling `_image_filename_to_path()` which already does `os.path.join(image_base_dir, filename)`, then joining with image_base_dir again, creating malformed paths like `/data/images/data/images/file.jpg`. Also wasn't passing `options.pathsep_replacement` parameter. Fixed to single call with all three parameters.
- **FIXED** (by user): Lines 182-183 - Missing type conversion for `extra_annotation_fields_to_print` parameter; function handled string-to-list conversion for `extra_image_fields_to_print` but not for `extra_annotation_fields_to_print`, causing substring matching instead of list membership when user passed a string. Added matching conversion check.

### megadetector/visualization/visualize_detector_output.py
- No bugs found. Clean implementation with proper resource management, pool cleanup, and error handling.

### megadetector/visualization/visualize_video_output.py
- **FIXED** (by user): Lines 303-305 - Directory creation issue; `os.path.dirname(output_fn_abs)` returns empty string for files in current directory. Added check `if len(parent_dir) > 0:` before makedirs.
- Lines 195-197: Missing KeyError check in trim_to_detections; accesses `detection['frame_number']` without checking if key exists, while lines 175-184 do check. User chose to skip (malformed data scenario).
- Lines 240-242: Missing KeyError check in _get_detections_for_frame(); accesses `detection['frame_number']` without checking if key exists. User chose to skip (malformed data scenario).
- **FIXED** (by user): Lines 211-246 - Missing video writer cleanup; if exception occurred during frame writing, `video_writer.release()` was never called. Added try/finally block with None check and nested try/except around release() for extra safety.
