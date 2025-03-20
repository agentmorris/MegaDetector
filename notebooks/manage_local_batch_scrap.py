"""

manage_local_batch_scrap.py

This file contains optional cells that I ocassionally run at the end of manage_local_batch.py

"""

#%% Avoid triggering execution on import

if False:
    
    pass

    #%% Compare results files for different model versions (or before/after RDE)
    
    import itertools
    
    from megadetector.postprocessing.compare_batch_results import \
        BatchComparisonOptions, PairwiseBatchComparisonOptions, compare_batch_results
    
    options = BatchComparisonOptions()
    
    options.job_name = organization_name_short
    options.output_folder = os.path.join(postprocessing_output_folder,'model_comparison')
    options.image_folder = input_path
    
    options.pairwise_options = []
    
    filenames = [
        '/postprocessing/organization/mdv4_results.json',
        '/postprocessing/organization/mdv5a_results.json',
        '/postprocessing/organization/mdv5b_results.json'    
        ]
    
    detection_thresholds = [0.7,0.15,0.15]
    
    assert len(detection_thresholds) == len(filenames)
    
    rendering_thresholds = [(x*0.6666) for x in detection_thresholds]
    
    # Choose all pairwise combinations of the files in [filenames]
    for i, j in itertools.combinations(list(range(0,len(filenames))),2):
            
        pairwise_options = PairwiseBatchComparisonOptions()
        
        pairwise_options.results_filename_a = filenames[i]
        pairwise_options.results_filename_b = filenames[j]
        
        pairwise_options.rendering_confidence_threshold_a = rendering_thresholds[i]
        pairwise_options.rendering_confidence_threshold_b = rendering_thresholds[j]
        
        pairwise_options.detection_thresholds_a = {'animal':detection_thresholds[i],
                                                   'person':detection_thresholds[i],
                                                   'vehicle':detection_thresholds[i]}
        pairwise_options.detection_thresholds_b = {'animal':detection_thresholds[j],
                                                   'person':detection_thresholds[j],
                                                   'vehicle':detection_thresholds[j]}
        options.pairwise_options.append(pairwise_options)
    
    results = compare_batch_results(options)
    
    from megadetector.utils.path_utils import open_file
    open_file(results.html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
    
    
    #%% Merge in high-confidence detections from another results file
    
    from megadetector.postprocessing.merge_detections import \
        MergeDetectionsOptions,merge_detections
    
    source_files = ['']
    target_file = ''
    output_file = target_file.replace('.json','_merged.json')
    
    options = MergeDetectionsOptions()
    options.max_detection_size = 1.0
    options.target_confidence_threshold = 0.25
    options.categories_to_include = [1]
    options.source_confidence_thresholds = [0.2]
    merge_detections(source_files, target_file, output_file, options)
    
    merged_detections_file = output_file
    
    
    #%% Create a new category for large boxes
    
    from megadetector.postprocessing import categorize_detections_by_size
    
    size_options = categorize_detections_by_size.SizeCategorizationOptions()
    
    size_options.size_thresholds = [0.9]
    size_options.size_category_names = ['large_detections']
    
    size_options.categories_to_separate = [1]
    size_options.measurement = 'size' # 'width'
    
    threshold_string = '-'.join([str(x) for x in size_options.size_thresholds])
    
    input_file = filtered_output_filename
    size_separated_file = input_file.replace('.json','-size-separated-{}.json'.format(
        threshold_string))
    d = categorize_detections_by_size.categorize_detections_by_size(input_file,size_separated_file,
                                                                    size_options)
    
    
    #%% Preview large boxes
    
    output_base_large_boxes = os.path.join(postprocessing_output_folder, 
        base_task_name + '_{}_{:.3f}_size_separated_boxes'.format(rde_string, options.confidence_threshold))    
    os.makedirs(output_base_large_boxes, exist_ok=True)
    print('Processing post-RDE, post-size-separation to {}'.format(output_base_large_boxes))
    
    options.md_results_file = size_separated_file
    options.output_dir = output_base_large_boxes
    
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
    
    
    #%% String replacement
        
    data = None
    
    from megadetector.postprocessing.subset_json_detector_output import \
        subset_json_detector_output, SubsetJsonDetectorOutputOptions
    
    input_filename = filtered_output_filename
    output_filename = input_filename.replace('.json','_replaced.json')
    
    options = SubsetJsonDetectorOutputOptions()
    options.query = folder_name + '/'
    options.replacement = ''
    subset_json_detector_output(input_filename,output_filename,options)
    
    
    #%% Splitting images into folders
    
    from megadetector.postprocessing.separate_detections_into_folders import \
        separate_detections_into_folders, SeparateDetectionsIntoFoldersOptions
    
    default_threshold = 0.2
    base_output_folder = os.path.expanduser('~/data/{}-{}-separated'.format(base_task_name,default_threshold))
    
    options = SeparateDetectionsIntoFoldersOptions(default_threshold)
    
    options.results_file = filtered_output_filename
    options.base_input_folder = input_path
    options.base_output_folder = os.path.join(base_output_folder,folder_name)
    options.n_threads = default_workers_for_parallel_tasks
    options.allow_existing_directory = False
    
    separate_detections_into_folders(options)
    
    
    #%% Convert frame-level results to video-level results
    
    # This cell is only useful if the files submitted to this job were generated via
    # video_folder_to_frames().
    
    from megadetector.detection.video_utils import frame_results_to_video_results
    
    video_output_filename = filtered_output_filename.replace('.json','_aggregated.json')
    frame_results_to_video_results(filtered_output_filename,video_output_filename)
