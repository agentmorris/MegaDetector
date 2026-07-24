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
