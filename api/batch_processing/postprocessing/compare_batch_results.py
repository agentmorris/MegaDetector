########
#
# compare_batch_results.py
# 
# Compare sets of batch results; typically used to compare:
#
# * MegaDetector versions
# * Results before/after RDE
# * Results with/without augmentation
#
# Makes pairwise comparisons, but can take lists of results files (will perform 
# all pairwise comparisons).  Results are written to an HTML page that shows the number
# and nature of disagreements (in the sense of each image being a detection or non-detection), 
# with sample images for each category.
# 
########

#%% Imports

import json
import os
import random
import copy
import urllib
import itertools

from tqdm import tqdm
from functools import partial

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

from md_visualization import visualization_utils
from md_utils.write_html_image_list import write_html_image_list
from md_utils import path_utils


#%% Constants and support classes

# We will confirm that this matches what we load from each file
default_detection_categories = {'1': 'animal', '2': 'person', '3': 'vehicle'}
    
class PairwiseBatchComparisonOptions:
    """
    Defines the options used for a single pairwise comparison; a list of these
    pairwise options sets is stored in the BatchComparisonsOptions class.
    """
    
    results_filename_a = None
    results_filename_b = None
    
    results_description_a = None
    results_description_b = None
    
    detection_thresholds_a = {'animal':0.15,'person':0.15,'vehicle':0.15}
    detection_thresholds_b = {'animal':0.15,'person':0.15,'vehicle':0.15}

    rendering_confidence_threshold_a = 0.1
    rendering_confidence_threshold_b = 0.1

# ...class PairwiseBatchComparisonOptions


class BatchComparisonOptions:
    """
    Defines the options for a set of (possibly many) pairwise comparisons.
    """
    
    output_folder = None
    image_folder = None
    job_name = ''
    
    max_images_per_category = 1000
    colormap_a = ['Red']
    colormap_b = ['RoyalBlue']

    # Process-based parallelization isn't supported yet; this must be "True"
    parallelize_rendering_with_threads = True
    
    target_width = 800
    n_rendering_workers = 20
    random_seed = 0
    
    error_on_non_matching_lists = True
    
    pairwise_options = []
    
# ...class BatchComparisonOptions
    

class PairwiseBatchComparisonResults:
    """
    The results from a single pairwise comparison
    """
    
    html_content = None
    pairwise_options = None
    
    # A dictionary with keys including:
    #
    # "common_detections"
    # common_non_detections
    # detections_a_only
    # detections_b_only
    # class_transitions
    #
    # Each of these maps a filename to a two-element list (the image in set A, the image in set B).
    categories_to_image_pairs = None

# ...class PairwiseBatchComparisonResults
    
    
class BatchComparisonResults:
    """
    The results from a set of pairwise comparisons
    """
    
    html_output_file = None
    
    # An list of PairwiseBatchComparisonResults
    pairwise_results = None
    
# ...class BatchComparisonResults    


main_page_style_header = """<head>
    <style type="text/css">
    a { text-decoration: none; }
    body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
    div.contentdiv { margin-left: 20px; }
    </style>
    </head>"""

main_page_header = '<html>\n{}\n<body>\n'.format(main_page_style_header)
main_page_footer = '<br/><br/><br/></body></html>\n'


#%% Comparison functions

def render_image_pair(fn,image_pairs,category_folder,options,pairwise_options):
    """
    Render two sets of results (i.e., a comparison) for a single image.
    """
    
    input_image_path = os.path.join(options.image_folder,fn)
    assert os.path.isfile(input_image_path), 'Image {} does not exist'.format(input_image_path)
    
    im = visualization_utils.open_image(input_image_path)
    image_pair = image_pairs[fn]
    detections_a = image_pair[0]['detections']
    detections_b = image_pair[1]['detections']
    
    custom_strings_a = [''] * len(detections_a)
    custom_strings_b = [''] * len(detections_b)    
    
    # This function is often used to compare results before/after various merging
    # steps, so we have some special-case formatting based on the "transferred_from"
    # field generated in merge_detections.py.
    for i_det,det in enumerate(detections_a):
        if 'transferred_from' in det:
            custom_strings_a[i_det] = '({})'.format(
                det['transferred_from'].split('.')[0])
    
    for i_det,det in enumerate(detections_b):
        if 'transferred_from' in det:
            custom_strings_b[i_det] = '({})'.format(
                det['transferred_from'].split('.')[0])
    
    if options.target_width is not None:
        im = visualization_utils.resize_image(im, options.target_width)
        
    visualization_utils.render_detection_bounding_boxes(detections_a,im,
        confidence_threshold=pairwise_options.rendering_confidence_threshold_a,
        thickness=4,expansion=0,
        colormap=options.colormap_a,
        textalign=visualization_utils.TEXTALIGN_LEFT,
        custom_strings=custom_strings_a)
    visualization_utils.render_detection_bounding_boxes(detections_b,im,
        confidence_threshold=pairwise_options.rendering_confidence_threshold_b,
        thickness=2,expansion=0,
        colormap=options.colormap_b,
        textalign=visualization_utils.TEXTALIGN_RIGHT,
        custom_strings=custom_strings_b)

    output_image_fn = path_utils.flatten_path(fn)
    output_image_path = os.path.join(category_folder,output_image_fn)
    im.save(output_image_path)
    return output_image_path

# ...def render_image_pair()


def pairwise_compare_batch_results(options,output_index,pairwise_options):
    """
    The main entry point for this module is compare_batch_results(), which calls 
    this function for each pair of comparisons the caller has requested.  Generates an
    HTML page for this comparison.  Returns a BatchComparisonResults object.
    
    options: an instance of BatchComparisonOptions
    
    output_index: a numeric index used for generating HTML titles
    
    pairwise_options: an instance of PairwiseBatchComparisonOptions    
    """
    
    # pairwise_options is passed as a parameter here, and should not be specified
    # in the options object.
    assert options.pairwise_options is None
    
    random.seed(options.random_seed)

    # Warn the user if some "detections" might not get rendered
    max_classification_threshold_a = max(list(pairwise_options.detection_thresholds_a.values()))
    max_classification_threshold_b = max(list(pairwise_options.detection_thresholds_b.values()))
    
    if pairwise_options.rendering_confidence_threshold_a > max_classification_threshold_a:
        print('*** Warning: rendering threshold A ({}) is higher than max confidence threshold A ({}) ***'.format(
            pairwise_options.rendering_confidence_threshold_a,max_classification_threshold_a))
    
    if pairwise_options.rendering_confidence_threshold_b > max_classification_threshold_b:
        print('*** Warning: rendering threshold B ({}) is higher than max confidence threshold B ({}) ***'.format(
            pairwise_options.rendering_confidence_threshold_b,max_classification_threshold_b))
    

    ##%% Validate inputs
    
    assert os.path.isfile(pairwise_options.results_filename_a), \
        "Can't find results file {}".format(pairwise_options.results_filename_a)
    assert os.path.isfile(pairwise_options.results_filename_b), \
        "Can't find results file {}".format(pairwise_options.results_filename_b)
    assert os.path.isdir(options.image_folder), \
        "Can't find image folder {}".format(pairwise_options.image_folder)
    os.makedirs(options.output_folder,exist_ok=True)
    
    
    ##%% Load both result sets
    
    with open(pairwise_options.results_filename_a,'r') as f:
        results_a = json.load(f)
    
    with open(pairwise_options.results_filename_b,'r') as f:
        results_b = json.load(f)
        
    # assert results_a['detection_categories'] == default_detection_categories
    # assert results_b['detection_categories'] == default_detection_categories
    assert results_a['detection_categories'] == results_b['detection_categories']
    detection_categories = results_a['detection_categories']
    
    if pairwise_options.results_description_a is None:
        if 'detector' not in results_a['info']:
            print('No model metadata supplied for results-A, assuming MDv4')
            pairwise_options.results_description_a = 'MDv4 (assumed)'
        else:            
            pairwise_options.results_description_a = results_a['info']['detector']
    
    if pairwise_options.results_description_b is None:
        if 'detector' not in results_b['info']:
            print('No model metadata supplied for results-B, assuming MDv4')
            pairwise_options.results_description_b = 'MDv4 (assumed)'
        else:            
            pairwise_options.results_description_b = results_b['info']['detector']
    
    images_a = results_a['images']
    images_b = results_b['images']
    
    filename_to_image_a = {im['file']:im for im in images_a}
    filename_to_image_b = {im['file']:im for im in images_b}
    
    
    ##%% Make sure they represent the same set of images
    
    filenames_a = [im['file'] for im in images_a]
    filenames_b_set = set([im['file'] for im in images_b])
    
    if len(images_a) != len(images_b):
        s = 'set A has {} iamges, set B has {}'.format(len(images_a),len(images_b))
        if options.error_on_non_matching_lists:
            raise ValueError(s)
        else:
            print('Warning: ' + s)
    else:
        if options.error_on_non_matching_lists:
            for fn in filenames_a:
                assert fn in filenames_b_set

    assert len(filenames_a) == len(images_a)
    assert len(filenames_b_set) == len(images_b)    
    
    
    ##%% Find differences
    
    # Each of these maps a filename to a two-element list (the image in set A, the image in set B)
    #
    # Right now, we only handle a very simple notion of class transition, where the detection
    # of maximum confidence changes class *and* both images have an above-threshold detection.
    common_detections = {}
    common_non_detections = {}
    detections_a_only = {}
    detections_b_only = {}
    class_transitions = {}
    
    # fn = filenames_a[0]
    for fn in tqdm(filenames_a):
    
        if fn not in filename_to_image_b:
            
            # We shouldn't have gotten this far if error_on_non_matching_lists is set
            assert not options.error_on_non_matching_lists
            
            print('Skipping filename {}, not in image set B'.format(fn))
            continue
        
        im_a = filename_to_image_a[fn]
        im_b = filename_to_image_b[fn]
        
        categories_above_threshold_a = set()

        if not 'detections' in im_a or im_a['detections'] is None:
            assert 'failure' in im_a and im_a['failure'] is not None
            continue
        
        if not 'detections' in im_b or im_b['detections'] is None:
            assert 'failure' in im_b and im_b['failure'] is not None
            continue
                
        invalid_category_error = False

        # det = im_a['detections'][0]
        for det in im_a['detections']:
            
            category_id = det['category']
            
            if category_id not in detection_categories:
                print('Warning: unexpected category {} for model A on file {}'.format(category_id,fn))
                invalid_category_error = True
                break
                
            conf = det['conf']
            
            if conf >= pairwise_options.detection_thresholds_a[detection_categories[category_id]]:
                categories_above_threshold_a.add(category_id)
                            
        if invalid_category_error:
            continue
        
        categories_above_threshold_b = set()
        
        for det in im_b['detections']:
            
            category_id = det['category']
            
            if category_id not in detection_categories:
                print('Warning: unexpected category {} for model B on file {}'.format(category_id,fn))
                invalid_category_error = True
                break
            
            conf = det['conf']
            
            if conf >= pairwise_options.detection_thresholds_b[detection_categories[category_id]]:
                categories_above_threshold_b.add(category_id)
                            
        if invalid_category_error:
            continue
        
        im_pair = (im_a,im_b)
        
        detection_a = (len(categories_above_threshold_a) > 0)
        detection_b = (len(categories_above_threshold_b) > 0)
                
        if detection_a and detection_b:            
            if categories_above_threshold_a == categories_above_threshold_b:
                common_detections[fn] = im_pair
            else:
                class_transitions[fn] = im_pair
        elif (not detection_a) and (not detection_b):
            common_non_detections[fn] = im_pair
        elif detection_a and (not detection_b):
            detections_a_only[fn] = im_pair
        else:
            assert detection_b and (not detection_a)
            detections_b_only[fn] = im_pair
            
    # ...for each filename
    
    print('Of {} files:\n{} common detections\n{} common non-detections\n{} A only\n{} B only\n{} class transitions'.format(
        len(filenames_a),len(common_detections),
        len(common_non_detections),len(detections_a_only),
        len(detections_b_only),len(class_transitions)))
        
    
    ##%% Sample and plot differences
    
    if options.n_rendering_workers > 1:
       worker_type = 'processes'
       if options.parallelize_rendering_with_threads:
           worker_type = 'threads'
       print('Rendering images with {} {}'.format(options.n_rendering_workers,worker_type))
       if options.parallelize_rendering_with_threads:
           pool = ThreadPool(options.n_rendering_workers)    
       else:
           pool = Pool(options.n_rendering_workers)    
        
    categories_to_image_pairs = {
        'common_detections':common_detections,
        'common_non_detections':common_non_detections,
        'detections_a_only':detections_a_only,
        'detections_b_only':detections_b_only,
        'class_transitions':class_transitions
    }
    
    categories_to_page_titles = {
        'common_detections':'Detections common to both models',
        'common_non_detections':'Non-detections common to both models',
        'detections_a_only':'Detections reported by model A only',
        'detections_b_only':'Detections reported by model B only',
        'class_transitions':'Detections reported as different classes by models A and B'
    }

    local_output_folder = os.path.join(options.output_folder,'cmp_' + \
                                       str(output_index).zfill(3))
    
    def render_detection_comparisons(category,image_pairs,image_filenames):
        
        print('Rendering detections for category {}'.format(category))
                
        category_folder = os.path.join(local_output_folder,category)
        os.makedirs(category_folder,exist_ok=True)
        
        # fn = image_filenames[0]
        if options.n_rendering_workers <= 1:
            output_image_paths = []
            for fn in tqdm(image_filenames):        
                output_image_paths.append(render_image_pair(fn,image_pairs,category_folder,
                                                            options,pairwise_options))
        else:            
            output_image_paths = list(tqdm(pool.imap(
                partial(render_image_pair, image_pairs=image_pairs, 
                        category_folder=category_folder,options=options,
                        pairwise_options=pairwise_options),
                image_filenames), 
                total=len(image_filenames)))
        
        return output_image_paths
    
    # ...def render_detection_comparisons()
    
    # For each category, generate comparison images and the 
    # comparison HTML page.
    #
    # category = 'common_detections'
    for category in categories_to_image_pairs.keys():
        
        # Choose detection pairs we're going to render for this category
        image_pairs = categories_to_image_pairs[category]
        image_filenames = list(image_pairs.keys())
        if len(image_filenames) > options.max_images_per_category:
            print('Sampling {} of {} image pairs for category {}'.format(
                options.max_images_per_category,
                len(image_filenames),
                category))
            image_filenames = random.sample(image_filenames,
                                            options.max_images_per_category)
        assert len(image_filenames) <= options.max_images_per_category

        input_image_absolute_paths = [os.path.join(options.image_folder,fn) for fn in image_filenames]
        
        category_image_output_paths = render_detection_comparisons(category,
                                                            image_pairs,image_filenames)
        
        category_html_filename = os.path.join(local_output_folder,
                                              category + '.html')
        category_image_output_paths_relative = [os.path.relpath(s,local_output_folder) \
                                         for s in category_image_output_paths]
        
        image_info = []
        
        assert len(category_image_output_paths_relative) == len(input_image_absolute_paths)
        
        for i_fn,fn in enumerate(category_image_output_paths_relative): 
            
            input_path_relative = image_filenames[i_fn]
            image_pair = image_pairs[input_path_relative]
            assert len(image_pair) == 2; image_a = image_pair[0]; image_b = image_pair[1]
            
            def maxempty(L):
                if len(L) == 0:
                    return 0
                else:
                    return max(L)
                
            max_conf_a = maxempty([det['conf'] for det in image_a['detections']])
            max_conf_b = maxempty([det['conf'] for det in image_b['detections']])
            
            title = input_path_relative + ' (max conf {:.2f},{:.2f})'.format(max_conf_a,max_conf_b)
            info = {
                'filename': fn,
                'title': title,
                'textStyle': 'font-family:verdana,arial,calibri;font-size:' + \
                    '80%;text-align:left;margin-top:20;margin-bottom:5',
                'linkTarget': urllib.parse.quote(input_image_absolute_paths[i_fn])
            }
            image_info.append(info)
    
        category_page_header_string = '<h1>{}</h1>'.format(categories_to_page_titles[category])
        category_page_header_string += '<p style="font-weight:bold;">\n'
        category_page_header_string += 'Model A: {}<br/>\n'.format(
            pairwise_options.results_description_a)
        category_page_header_string += 'Model B: {}'.format(pairwise_options.results_description_b)
        category_page_header_string += '</p>\n'
        
        category_page_header_string += '<p>\n'
        category_page_header_string += 'Detection thresholds for A ({}):\n{}<br/>'.format(
            pairwise_options.results_description_a,str(pairwise_options.detection_thresholds_a))
        category_page_header_string += 'Detection thresholds for B ({}):\n{}<br/>'.format(
            pairwise_options.results_description_b,str(pairwise_options.detection_thresholds_b))
        category_page_header_string += 'Rendering threshold for A ({}):\n{}<br/>'.format(
            pairwise_options.results_description_a,
            str(pairwise_options.rendering_confidence_threshold_a))
        category_page_header_string += 'Rendering threshold for B ({}):\n{}<br/>'.format(
            pairwise_options.results_description_b,
            str(pairwise_options.rendering_confidence_threshold_b))
        category_page_header_string += '</p>\n'        
        
        write_html_image_list(
            category_html_filename,
            images=image_info,
            options={
                'headerHtml': category_page_header_string
            })
        
    # ...for each category
    
    
    ##%% Write the top-level HTML file content

    html_output_string  = ''
    
    html_output_string += '<p>Comparing <b>{}</b> (A, red) to <b>{}</b> (B, blue)</p>'.format(
        pairwise_options.results_description_a,pairwise_options.results_description_b)
    html_output_string += '<div class="contentdiv">\n'
    html_output_string += 'Detection thresholds for {}:\n{}<br/>'.format(
        pairwise_options.results_description_a,
        str(pairwise_options.detection_thresholds_a))
    html_output_string += 'Detection thresholds for {}:\n{}<br/>'.format(
        pairwise_options.results_description_b,
        str(pairwise_options.detection_thresholds_b))
    html_output_string += 'Rendering threshold for {}:\n{}<br/>'.format(
        pairwise_options.results_description_a,
        str(pairwise_options.rendering_confidence_threshold_a))
    html_output_string += 'Rendering threshold for {}:\n{}<br/>'.format(
        pairwise_options.results_description_b,
        str(pairwise_options.rendering_confidence_threshold_b))
    
    html_output_string += '<br/>'
    
    html_output_string += 'Rendering a maximum of {} images per category<br/>'.format(
        options.max_images_per_category)
    
    html_output_string += '<br/>'
    
    html_output_string += ('Of {} total files:<br/><br/><div style="margin-left:15px;">{} common detections<br/>{} common non-detections<br/>{} A only<br/>{} B only<br/>{} class transitions</div><br/>'.format(
        len(filenames_a),len(common_detections),
        len(common_non_detections),len(detections_a_only),
        len(detections_b_only),len(class_transitions)))
    
    html_output_string += 'Comparison pages:<br/><br/>\n'
    html_output_string += '<div style="margin-left:15px;">\n'
        
    comparison_path_relative = os.path.relpath(local_output_folder,options.output_folder)    
    for category in categories_to_image_pairs.keys():
        category_html_filename = os.path.join(comparison_path_relative,category + '.html')
        html_output_string += '<a href="{}">{}</a><br/>\n'.format(
            category_html_filename,category)
    
    html_output_string += '</div>\n'
    html_output_string += '</div>\n'
    
    pairwise_results = PairwiseBatchComparisonResults()
    
    pairwise_results.html_content = html_output_string
    pairwise_results.pairwise_options = pairwise_options
    pairwise_results.categories_to_image_pairs = categories_to_image_pairs
            
    return pairwise_results
        
# ...def compare_batch_results()


def compare_batch_results(options):
    """
    The main entry point for this module.  Runs one or more batch results comparisons, 
    writing results to an html page.  Most of the work is deferred to
    pairwise_compare_batch_results().
    """
    
    assert options.output_folder is not None
    assert options.image_folder is not None
    assert options.pairwise_options is not None

    options = copy.deepcopy(options)
 
    if not isinstance(options.pairwise_options,list):
        options.pairwise_options = [options.pairwise_options]
    
    pairwise_options_list = options.pairwise_options
    n_comparisons = len(pairwise_options_list)
    
    options.pairwise_options = None
    
    html_content = ''
    all_pairwise_results = []
    
    # i_comparison = 0; pairwise_options = pairwise_options_list[i_comparison]
    for i_comparison,pairwise_options in enumerate(pairwise_options_list):
        print('Running comparison {} of {}'.format(i_comparison,n_comparisons))
        pairwise_results = \
            pairwise_compare_batch_results(options,i_comparison,pairwise_options)
        html_content += pairwise_results.html_content
        all_pairwise_results.append(pairwise_results)

    html_output_string = main_page_header
    job_name_string = ''
    if len(options.job_name) > 0:
        job_name_string = ' for {}'.format(options.job_name)
    html_output_string += '<h2>Comparison of results{}</h2>\n'.format(
        job_name_string)
    html_output_string += html_content
    html_output_string += main_page_footer
    
    html_output_file = os.path.join(options.output_folder,'index.html')    
    with open(html_output_file,'w') as f:
        f.write(html_output_string) 
    
    results = BatchComparisonResults()
    results.html_output_file = html_output_file
    results.pairwise_results = all_pairwise_results
    return results


def n_way_comparison(filenames,options,detection_thresholds=None,rendering_thresholds=None):
    """
    Performs N pairwise comparisons for the list of results files in [filenames], by generating
    sets of pairwise options and calling compare_batch_results.
    """
    
    if detection_thresholds is None:
        detection_thresholds = [0.15] * len(filenames)
    assert len(detection_thresholds) == len(filenames)

    if rendering_thresholds is not None:
        assert len(rendering_thresholds) == len(detection_thresholds)
    else:
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

    return compare_batch_results(options)

# ...n_way_comparison()


#%% Interactive driver

if False:
    
    #%% Running KGA test
    
    # CUDA_VISIBLE_DEVICES=0 python run_detector_batch.py  ~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt ~/data/KGA/ ~/data/KGA-5a.json --recursive --output_relative_filenames --quiet
    # CUDA_VISIBLE_DEVICES=1 python run_detector_batch.py  ~/models/camera_traps/megadetector/md_v5.0.0/md_v5b.0.0.pt ~/data/KGA/ ~/data/KGA-5b.json --recursive --output_relative_filenames --quiet
    
    # python run_detector_batch.py  ~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb ~/data/KGA ~/data/KGA-4.json --recursive --output_relative_filenames --quiet
    
    # CUDA_VISIBLE_DEVICES=0 python run_detector_batch.py  ~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt ~/data/KGA/ ~/data/KGA-5a-pillow-9.2.0.json --recursive --output_relative_filenames --quiet


    #%% Test two-way comparison
    
    options = BatchComparisonOptions()

    options.parallelize_rendering_with_threads = False
    
    options.job_name = 'KGA-test'
    options.output_folder = os.path.expanduser('~/tmp/md-comparison-test')
    options.image_folder = os.path.expanduser('~/data/KGA')

    options.pairwise_options = []

    filenames = [
        os.path.expanduser('~/data/KGA-5a.json'),
        os.path.expanduser('~/data/KGA-5b.json')
        ]

    detection_thresholds = [0.15,0.15]
    rendering_thresholds = None
    
    results = n_way_comparison(filenames,options,detection_thresholds,rendering_thresholds=rendering_thresholds)
    
    from md_utils.path_utils import open_file
    open_file(results.html_output_file)

    
    #%% Test three-way comparison
    
    options = BatchComparisonOptions()

    options.parallelize_rendering_with_threads = False

    options.job_name = 'KGA-test'
    options.output_folder = os.path.expanduser('~/tmp/md-comparison-test')
    options.image_folder = os.path.expanduser('~/data/KGA')

    options.pairwise_options = []

    filenames = [
        os.path.expanduser('~/data/KGA-4.json'),
        os.path.expanduser('~/data/KGA-5a.json'),
        os.path.expanduser('~/data/KGA-5b.json')
        ]

    detection_thresholds = [0.7,0.15,0.15]

    results = n_way_comparison(filenames,options,detection_thresholds,rendering_thresholds=None)
    
    from md_utils.path_utils import open_file
    open_file(results.html_output_file)


#%% Command-line driver

"""
python compare_batch_results.py ~/tmp/comparison-test ~/data/KGA ~/data/KGA-5a.json ~/data/KGA-5b.json ~/data/KGA-4.json --detection_thresholds 0.15 0.15 0.7 --rendering_thresholds 0.1 0.1 0.6 --use_processes
"""

import sys,argparse,textwrap

def main():
    
    options = BatchComparisonOptions()
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
           Example:
        
           python compare_batch_results.py output_folder image_folder mdv5a.json mdv5b.json mdv4.json --detection_thresholds 0.15 0.15 0.7
           '''))
          
    parser.add_argument('output_folder', type=str, help='folder to which to write html results')

    parser.add_argument('image_folder', type=str, help='image source folder')

    parser.add_argument('results_files', nargs='*', type=str, help='list of .json files to be compared')

    parser.add_argument('--detection_thresholds', nargs='*', type=float,
                        help='list of detection thresholds, same length as the number of .json files, ' + \
                             'defaults to 0.15 for all files')
    
    parser.add_argument('--rendering_thresholds', nargs='*', type=float,
                        help='list of rendering thresholds, same length as the number of .json files, ' + \
                             'defaults to 0.10 for all files')
    
    parser.add_argument('--max_images_per_category', type=int, default=options.max_images_per_category,
                       help='number of images to sample for each agreement category (common detections, etc.)')
    
    parser.add_argument('--target_width', type=int, default=options.target_width, 
                        help='output image width, defaults to {}'.format(options.target_width))
    
    parser.add_argument('--use_processes', action='store_true', 
                        help='use processes rather than threads for parallelization')
    
    parser.add_argument('--open_results', action='store_true', 
                        help='open the output html file when done')
    
    parser.add_argument('--n_rendering_workers', type=int, default=options.n_rendering_workers,
                        help='number of workers for parallel rendering, defaults to {}'.format(
                            options.n_rendering_workers))
        
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()
    
    print('Output folder:')
    print(args.output_folder)
    
    print('\nResults files:')
    print(args.results_files)
    
    print('\nDetection thresholds:')
    print(args.detection_thresholds)
    
    print('\nRendering thresholds:')
    print(args.rendering_thresholds)    
    
    # Convert to options objects
    options = BatchComparisonOptions()
    
    options.output_folder = args.output_folder
    options.image_folder = args.image_folder
    options.target_width = args.target_width
    options.n_rendering_workers = args.n_rendering_workers    
    options.max_images_per_category = args.max_images_per_category
    
    if args.use_processes:
        options.parallelize_rendering_with_threads = False
        
    results = n_way_comparison(args.results_files,options,args.detection_thresholds,args.rendering_thresholds)
    
    if args.open_results:
        path_utils.open_file(results.html_output_file)
        
    print('Wrote results to {}'.format(results.html_output_file))
    
# ...main()

    
if __name__ == '__main__':
    
    main()
    