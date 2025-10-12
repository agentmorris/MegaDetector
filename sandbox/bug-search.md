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
- [ ] megadetector/data_management/labelme_to_yolo.py
- [ ] megadetector/data_management/mewc_to_md.py
- [ ] megadetector/data_management/ocr_tools.py
- [ ] megadetector/data_management/read_exif.py
- [ ] megadetector/data_management/remap_coco_categories.py
- [ ] megadetector/data_management/remove_exif.py
- [ ] megadetector/data_management/rename_images.py
- [ ] megadetector/data_management/resize_coco_dataset.py
- [ ] megadetector/data_management/speciesnet_to_md.py
- [ ] megadetector/data_management/wi_download_csv_to_coco.py
- [ ] megadetector/data_management/yolo_output_to_md_output.py
- [ ] megadetector/data_management/yolo_to_coco.py
- [ ] megadetector/data_management/zamba_to_md.py

### data_management/annotations
- [ ] megadetector/data_management/annotations/annotation_constants.py

### data_management/databases
- [ ] megadetector/data_management/databases/add_width_and_height_to_db.py
- [ ] megadetector/data_management/databases/combine_coco_camera_traps_files.py
- [ ] megadetector/data_management/databases/integrity_check_json_db.py
- [ ] megadetector/data_management/databases/subset_json_db.py

### data_management/lila
- [ ] megadetector/data_management/lila/create_lila_blank_set.py
- [ ] megadetector/data_management/lila/create_lila_test_set.py
- [ ] megadetector/data_management/lila/create_links_to_md_results_files.py
- [ ] megadetector/data_management/lila/download_lila_subset.py
- [ ] megadetector/data_management/lila/generate_lila_per_image_labels.py
- [ ] megadetector/data_management/lila/get_lila_annotation_counts.py
- [ ] megadetector/data_management/lila/get_lila_image_counts.py
- [ ] megadetector/data_management/lila/lila_common.py
- [ ] megadetector/data_management/lila/test_lila_metadata_urls.py

### detection
- [ ] megadetector/detection/change_detection.py
- [ ] megadetector/detection/process_video.py
- [ ] megadetector/detection/pytorch_detector.py
- [ ] megadetector/detection/run_detector.py
- [ ] megadetector/detection/run_detector_batch.py
- [ ] megadetector/detection/run_inference_with_yolov5_val.py
- [ ] megadetector/detection/run_md_and_speciesnet.py
- [ ] megadetector/detection/run_tiled_inference.py
- [ ] megadetector/detection/tf_detector.py
- [ ] megadetector/detection/video_utils.py

### postprocessing
- [ ] megadetector/postprocessing/add_max_conf.py
- [ ] megadetector/postprocessing/categorize_detections_by_size.py
- [ ] megadetector/postprocessing/classification_postprocessing.py
- [ ] megadetector/postprocessing/combine_batch_outputs.py
- [ ] megadetector/postprocessing/compare_batch_results.py
- [ ] megadetector/postprocessing/convert_output_format.py
- [ ] megadetector/postprocessing/create_crop_folder.py
- [ ] megadetector/postprocessing/detector_calibration.py
- [ ] megadetector/postprocessing/generate_csv_report.py
- [ ] megadetector/postprocessing/load_api_results.py
- [ ] megadetector/postprocessing/md_to_coco.py
- [ ] megadetector/postprocessing/md_to_labelme.py
- [ ] megadetector/postprocessing/md_to_wi.py
- [ ] megadetector/postprocessing/merge_detections.py
- [ ] megadetector/postprocessing/postprocess_batch_results.py
- [ ] megadetector/postprocessing/remap_detection_categories.py
- [ ] megadetector/postprocessing/render_detection_confusion_matrix.py
- [ ] megadetector/postprocessing/separate_detections_into_folders.py
- [ ] megadetector/postprocessing/subset_json_detector_output.py
- [ ] megadetector/postprocessing/top_folders_to_bottom.py
- [ ] megadetector/postprocessing/validate_batch_results.py

### postprocessing/repeat_detection_elimination
- [ ] megadetector/postprocessing/repeat_detection_elimination/find_repeat_detections.py
- [ ] megadetector/postprocessing/repeat_detection_elimination/remove_repeat_detections.py
- [ ] megadetector/postprocessing/repeat_detection_elimination/repeat_detections_core.py

### taxonomy_mapping
- [ ] megadetector/taxonomy_mapping/map_lila_taxonomy_to_wi_taxonomy.py
- [ ] megadetector/taxonomy_mapping/map_new_lila_datasets.py
- [ ] megadetector/taxonomy_mapping/prepare_lila_taxonomy_release.py
- [ ] megadetector/taxonomy_mapping/preview_lila_taxonomy.py
- [ ] megadetector/taxonomy_mapping/retrieve_sample_image.py
- [ ] megadetector/taxonomy_mapping/simple_image_download.py
- [ ] megadetector/taxonomy_mapping/species_lookup.py
- [ ] megadetector/taxonomy_mapping/taxonomy_csv_checker.py
- [ ] megadetector/taxonomy_mapping/taxonomy_graph.py
- [ ] megadetector/taxonomy_mapping/validate_lila_category_mappings.py

### utils
- [ ] megadetector/utils/ct_utils.py
- [ ] megadetector/utils/directory_listing.py
- [ ] megadetector/utils/extract_frames_from_video.py
- [ ] megadetector/utils/gpu_test.py
- [ ] megadetector/utils/md_tests.py
- [ ] megadetector/utils/path_utils.py
- [ ] megadetector/utils/process_utils.py
- [ ] megadetector/utils/split_locations_into_train_val.py
- [ ] megadetector/utils/string_utils.py
- [ ] megadetector/utils/url_utils.py
- [ ] megadetector/utils/wi_platform_utils.py
- [ ] megadetector/utils/wi_taxonomy_utils.py
- [ ] megadetector/utils/write_html_image_list.py

### visualization
- [ ] megadetector/visualization/plot_utils.py
- [ ] megadetector/visualization/render_images_with_thumbnails.py
- [ ] megadetector/visualization/visualization_utils.py
- [ ] megadetector/visualization/visualize_db.py
- [ ] megadetector/visualization/visualize_detector_output.py
- [ ] megadetector/visualization/visualize_video_output.py

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
