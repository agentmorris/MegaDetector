"""

render_detection_confusion_matrix.py

Given a CCT-formatted ground truth file and a MegaDetector-formatted results file,
render an HTML confusion matrix.  Typically used for multi-class detectors.  Currently
assumes a single class per image.

"""

#%% Imports and constants

import os
import json

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from functools import partial

from megadetector.utils.path_utils import find_images
from megadetector.utils.path_utils import flatten_path
from megadetector.utils.write_html_image_list import write_html_image_list
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.visualization import plot_utils

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool


#%% Support functions

def _image_to_output_file(im,preview_images_folder):
    """
    Produces a clean filename from im (if [im] is a str) or im['file'] (if [im] is a dict).
    """
    
    if isinstance(im,str):
        filename_relative = im
    else:
        filename_relative = im['file']
        
    fn_clean = flatten_path(filename_relative).replace(' ','_')
    return os.path.join(preview_images_folder,fn_clean)


def _render_image(im,render_image_constants):
    """
    Internal function for rendering a single image to the confusion matrix preview folder.
    """
    
    filename_to_ground_truth_im = render_image_constants['filename_to_ground_truth_im']
    image_folder = render_image_constants['image_folder']
    preview_images_folder = render_image_constants['preview_images_folder']
    force_render_images = render_image_constants['force_render_images']
    results_category_id_to_name = render_image_constants['results_category_id_to_name']
    rendering_confidence_thresholds = render_image_constants['rendering_confidence_thresholds']
    target_image_size = render_image_constants['target_image_size']
    
    assert im['file'] in filename_to_ground_truth_im
    
    output_file = _image_to_output_file(im,preview_images_folder)
    if os.path.isfile(output_file) and not force_render_images:
        return output_file
    
    input_file = os.path.join(image_folder,im['file'])
    assert os.path.isfile(input_file)
                          
    detections_to_render = []
    
    for det in im['detections']:
        category_name = results_category_id_to_name[det['category']]
        detection_threshold = rendering_confidence_thresholds['default']
        if category_name in rendering_confidence_thresholds:
            detection_threshold = rendering_confidence_thresholds[category_name]
        if det['conf'] > detection_threshold:
            detections_to_render.append(det)
        
    vis_utils.draw_bounding_boxes_on_file(input_file, output_file, detections_to_render,
                                          detector_label_map=results_category_id_to_name,
                                          label_font_size=20,target_size=target_image_size)
    
    return output_file


#%% Main function

def render_detection_confusion_matrix(ground_truth_file,
                                      results_file,
                                      image_folder,
                                      preview_folder,
                                      force_render_images=False, 
                                      confidence_thresholds=None,
                                      rendering_confidence_thresholds=None,
                                      target_image_size=(1280,-1),
                                      parallelize_rendering=True,
                                      parallelize_rendering_n_cores=None,
                                      parallelize_rendering_with_threads=False,
                                      job_name='unknown',
                                      model_file=None,
                                      empty_category_name='empty',
                                      html_image_list_options=None):
    """    
    Given a CCT-formatted ground truth file and a MegaDetector-formatted results file,
    render an HTML confusion matrix in [preview_folder.  Typically used for multi-class detectors. 
    Currently assumes a single class per image.
    
    confidence_thresholds and rendering_confidence_thresholds are dictionaries mapping
    class names to thresholds.  "default" is a special token that will be used for all
    classes not otherwise assigned thresholds.
    
    Args:
        ground_truth_file (str): the CCT-formatted .json file with ground truth information
        results_file (str): the MegaDetector results .json file
        image_folder (str): the folder where images live; filenames in [ground_truth_file] and
            [results_file] should be relative to this folder.
        preview_folder (str): the output folder, i.e. the folder in which we'll create our nifty
            HTML stuff.
        force_rendering_images (bool, optional): if False, skips images that already exist
        confidence_thresholds (dict, optional): a dictionary mapping class names to thresholds;
            all classes not explicitly named here will use the threshold for the "default" category.
        rendering_thresholds (dict, optional): a dictionary mapping class names to thresholds;
            all classes not explicitly named here will use the threshold for the "default" category.
        target_image_size (tuple, optional): output image size, as a pair of ints (width,height).  If one 
            value is -1 and the other is not, aspect ratio is preserved.  If both are -1, the original image
            sizes are preserved.
        parallelize_rendering (bool, optional): enable (default) or disable parallelization when rendering
        parallelize_rendering_n_core (int, optional): number of threads or processes to use for rendering, only
            used if parallelize_rendering is True
        parallelize_rendering_with_threads: whether to use threads (True) or processes (False) when rendering,
            only used if parallelize_rendering is True
        job_name (str, optional): job name to include in big letters in the output file
        model_file (str, optional) model filename to include in HTML output
        empty_category_name (str, optional): special category name that we should treat as empty, typically
            "empty"
        html_image_list_options (dict, optional): options listed passed along to write_html_image_list; 
            see write_html_image_list for documentation.            
    """
    
    ##%% Argument and path handling
    
    preview_images_folder = os.path.join(preview_folder,'images')
    os.makedirs(preview_images_folder,exist_ok=True)

    if confidence_thresholds is None:
        confidence_thresholds = {'default':0.5}
    if rendering_confidence_thresholds is None:
        rendering_confidence_thresholds = {'default':0.4}
    

    ##%% Load ground truth 
        
    with open(ground_truth_file,'r') as f:
        ground_truth_data_cct = json.load(f)
    
    filename_to_ground_truth_im = {}
    for im in ground_truth_data_cct['images']:
        assert im['file_name'] not in filename_to_ground_truth_im
        filename_to_ground_truth_im[im['file_name']] = im
    
    
    ##%% Confirm that the ground truth images are present in the image folder
    
    ground_truth_images = find_images(image_folder,return_relative_paths=True,recursive=True)
    assert len(ground_truth_images) == len(ground_truth_data_cct['images'])
    del ground_truth_images
    
    
    ##%% Map images to categories
    
    # gt_image_id_to_image = {im['id']:im for im in ground_truth_data_cct['images']}
    gt_image_id_to_annotations = defaultdict(list)
    
    ground_truth_category_id_to_name = {}
    for c in ground_truth_data_cct['categories']:
        ground_truth_category_id_to_name[c['id']] = c['name']
    
    # Add the empty category if necessary
    if empty_category_name not in ground_truth_category_id_to_name.values():
        empty_category_id = max(ground_truth_category_id_to_name.keys())+1
        ground_truth_category_id_to_name[empty_category_id] = empty_category_name
        
    ground_truth_category_names = sorted(list(ground_truth_category_id_to_name.values()))
        
    for ann in ground_truth_data_cct['annotations']:
        gt_image_id_to_annotations[ann['image_id']].append(ann)
        
    gt_filename_to_category_names = defaultdict(set)
    
    for im in ground_truth_data_cct['images']:
        annotations_this_image = gt_image_id_to_annotations[im['id']]
        for ann in annotations_this_image:
            category_name = ground_truth_category_id_to_name[ann['category_id']]
            gt_filename_to_category_names[im['file_name']].add(category_name)
            
    for filename in gt_filename_to_category_names:
        
        category_names_this_file = gt_filename_to_category_names[filename]
        
        # The empty category should be exclusive
        if empty_category_name in category_names_this_file:
            assert len(category_names_this_file) == 1, \
              'Empty category assigned along with another category for {}'.format(filename)
        assert len(category_names_this_file) > 0, \
            'No ground truth category assigned to {}'.format(filename)
    
        
    ##%% Load results
    
    with open(results_file,'r') as f:
        md_formatted_results = json.load(f)
    
    results_category_id_to_name = md_formatted_results['detection_categories']
    
    
    ##%% Render images with detections    
    
    render_image_constants = {}
    render_image_constants['filename_to_ground_truth_im'] = filename_to_ground_truth_im
    render_image_constants['image_folder'] = image_folder
    render_image_constants['preview_images_folder'] = preview_images_folder
    render_image_constants['force_render_images'] = force_render_images
    render_image_constants['results_category_id_to_name'] = results_category_id_to_name
    render_image_constants['rendering_confidence_thresholds'] = rendering_confidence_thresholds
    render_image_constants['target_image_size'] = target_image_size    
    
    if parallelize_rendering:

        if parallelize_rendering_n_cores is None:                
            if parallelize_rendering_with_threads:
                pool = ThreadPool()
            else:
                pool = Pool()
        else:
            if parallelize_rendering_with_threads:
                pool = ThreadPool(parallelize_rendering_n_cores)
                worker_string = 'threads'
            else:
                pool = Pool(parallelize_rendering_n_cores)
                worker_string = 'processes'
            print('Rendering images with {} {}'.format(parallelize_rendering_n_cores,
                                                       worker_string))
            
        _ = list(tqdm(pool.imap(partial(_render_image,render_image_constants=render_image_constants),
                                md_formatted_results['images']),
                                total=len(md_formatted_results['images'])))        
    
    else:
        
        # im = md_formatted_results['images'][0]
        for im in tqdm(md_formatted_results['images']):    
            _render_image(im,render_image_constants)
    
    
    ##%% Map images to predicted categories, and vice-versa
    
    filename_to_predicted_categories = defaultdict(set)
    predicted_category_name_to_filenames = defaultdict(set)
    
    # im = md_formatted_results['images'][0]
    for im in tqdm(md_formatted_results['images']):
        
        assert im['file'] in filename_to_ground_truth_im
        
        # det = im['detections'][0]
        for det in im['detections']:
            category_name = results_category_id_to_name[det['category']]
            detection_threshold = confidence_thresholds['default']
            if category_name in confidence_thresholds:
                detection_threshold = confidence_thresholds[category_name]
            if det['conf'] > detection_threshold:
                filename_to_predicted_categories[im['file']].add(category_name)
                predicted_category_name_to_filenames[category_name].add(im['file'])
                
        # ...for each detection
    
    # ...for each image
    
    
    ##%% Create TP/TN/FP/FN lists
    
    category_name_to_image_lists = {}
    
    sub_page_tokens = ['fn','tn','fp','tp']
    
    for category_name in ground_truth_category_names:
        
        category_name_to_image_lists[category_name] = {}
        for sub_page_token in sub_page_tokens:
            category_name_to_image_lists[category_name][sub_page_token] = []
        
    # filename = next(iter(gt_filename_to_category_names))
    for filename in gt_filename_to_category_names.keys():
        
        ground_truth_categories_this_image = gt_filename_to_category_names[filename]
        predicted_categories_this_image = filename_to_predicted_categories[filename]
        
        for category_name in ground_truth_category_names:
            
            assignment = None
            
            if category_name == empty_category_name:
                # If this is an empty image
                if category_name in ground_truth_categories_this_image:
                    assert len(ground_truth_categories_this_image) == 1
                    if len(predicted_categories_this_image) == 0:
                        assignment = 'tp'
                    else:
                        assignment = 'fn'
                # If this not an empty image
                else:
                    if len(predicted_categories_this_image) == 0:
                        assignment = 'fp'
                    else:
                        assignment = 'tn'
                    
            else:
                if category_name in ground_truth_categories_this_image:
                    if category_name in predicted_categories_this_image:
                        assignment = 'tp'
                    else:
                        assignment = 'fn'
                else:
                    if category_name in predicted_categories_this_image:
                        assignment = 'fp'
                    else:
                        assignment = 'tn'        
                            
            category_name_to_image_lists[category_name][assignment].append(filename)
                        
    # ...for each filename
    
    
    ##%% Create confusion matrix
    
    gt_category_name_to_category_index = {}
    
    for i_category,category_name in enumerate(ground_truth_category_names):
        gt_category_name_to_category_index[category_name] = i_category
    
    n_categories = len(gt_category_name_to_category_index)    
    
    # indexed as [true,predicted]
    confusion_matrix = np.zeros(shape=(n_categories,n_categories),dtype=int)
    
    filename_to_results_im = {im['file']:im for im in md_formatted_results['images']}
    
    true_predicted_to_file_list = defaultdict(list)
    
    # filename = next(iter(gt_filename_to_category_names.keys()))
    for filename in gt_filename_to_category_names.keys():
        
        ground_truth_categories_this_image = gt_filename_to_category_names[filename]
        assert len(ground_truth_categories_this_image) == 1
        ground_truth_category_name = next(iter(ground_truth_categories_this_image))
        
        results_im = filename_to_results_im[filename]
        
        # If there were no detections at all, call this image empty
        if len(results_im['detections']) == 0:
            predicted_category_name = empty_category_name
        # Otherwise look for above-threshold detections
        else:            
            results_category_name_to_confidence = defaultdict(int)
            for det in results_im['detections']:
                category_name = results_category_id_to_name[det['category']]
                detection_threshold = confidence_thresholds['default']
                if category_name in confidence_thresholds:
                    detection_threshold = confidence_thresholds[category_name]
                if det['conf'] > detection_threshold:
                    results_category_name_to_confidence[category_name] = max(
                        results_category_name_to_confidence[category_name],det['conf'])
                # If there were no detections above threshold
                if len(results_category_name_to_confidence) == 0:
                    predicted_category_name = empty_category_name
                else:
                    predicted_category_name = max(results_category_name_to_confidence,
                        key=results_category_name_to_confidence.get)
        
        ground_truth_category_index = gt_category_name_to_category_index[ground_truth_category_name]
        predicted_category_index = gt_category_name_to_category_index[predicted_category_name]
        
        true_predicted_token = ground_truth_category_name + '_' + predicted_category_name
        true_predicted_to_file_list[true_predicted_token].append(filename)
        
        confusion_matrix[ground_truth_category_index,predicted_category_index] += 1
    
    # ...for each file
    
    plt.ioff()    
    
    fig_h = 3 + 0.3 * n_categories
    fig_w = fig_h
    fig = plt.figure(figsize=(fig_w, fig_h),tight_layout=True)
        
    plot_utils.plot_confusion_matrix(
        matrix=confusion_matrix,
        classes=ground_truth_category_names,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues,
        vmax=1.0,
        use_colorbar=False,
        y_label=True,
        fig=fig)
    
    cm_figure_fn_relative = 'confusion_matrix.png'
    cm_figure_fn_abs = os.path.join(preview_folder, cm_figure_fn_relative)
    # fig.show()
    fig.savefig(cm_figure_fn_abs,dpi=100)
    plt.close(fig)
    
    # open_file(cm_figure_fn_abs)
    
    
    ##%% Create HTML confusion matrix
    
    html_confusion_matrix = '<table class="result-table">\n'
    html_confusion_matrix += '<tr>\n'
    html_confusion_matrix += '<td>{}</td>\n'.format('True category')
    for category_name in ground_truth_category_names:
        html_confusion_matrix += '<td>{}</td>\n'.format('&nbsp;')
    html_confusion_matrix += '</tr>\n'
    
    for true_category in ground_truth_category_names:
        
        html_confusion_matrix += '<tr>\n'
        html_confusion_matrix += '<td>{}</td>\n'.format(true_category)
        
        for predicted_category in ground_truth_category_names:
            
            true_predicted_token = true_category + '_' + predicted_category
            image_list = true_predicted_to_file_list[true_predicted_token]
            if len(image_list) == 0:
                td_content = '0'
            else:
                if html_image_list_options is None:
                    html_image_list_options = {}
                title_string = 'true: {}, predicted {}'.format(
                    true_category,predicted_category)
                html_image_list_options['headerHtml'] = '<h1>{}</h1>'.format(title_string)
                
                html_image_info_list = []
                
                for image_filename_relative in image_list:
                    html_image_info = {}
                    detections = filename_to_results_im[image_filename_relative]['detections']
                    if len(detections) == 0:
                        max_conf = 0
                    else:
                        max_conf = max([d['conf'] for d in detections])
                    
                    title = '<b>Image</b>: {}, <b>Max conf</b>: {:0.3f}'.format(
                        image_filename_relative, max_conf)
                    image_link = 'images/' + os.path.basename(
                        _image_to_output_file(image_filename_relative,preview_images_folder))
                    html_image_info = {
                        'filename': image_link,
                        'title': title,
                        'textStyle':\
                         'font-family:verdana,arial,calibri;font-size:80%;' + \
                             'text-align:left;margin-top:20;margin-bottom:5'
                    }                
                    
                    html_image_info_list.append(html_image_info)
                
                target_html_file_relative = true_predicted_token + '.html'
                target_html_file_abs = os.path.join(preview_folder,target_html_file_relative)
                write_html_image_list(
                    filename=target_html_file_abs,
                    images=html_image_info_list,
                    options=html_image_list_options)
                
                td_content = '<a href="{}">{}</a>'.format(target_html_file_relative,
                                                          len(image_list))
            
            html_confusion_matrix += '<td>{}</td>\n'.format(td_content)
        
        # ...for each predicted category
        
        html_confusion_matrix += '</tr>\n'
        
    # ...for each true category    
    
    html_confusion_matrix += '<tr>\n'
    html_confusion_matrix += '<td>&nbsp;</td>\n'
    
    for category_name in ground_truth_category_names:
        html_confusion_matrix += '<td class="rotate"><p style="margin-left:20px;">{}</p></td>\n'.format(
            category_name)
    html_confusion_matrix += '</tr>\n'
    
    html_confusion_matrix += '</table>'
    
        
    ##%% Create HTML sub-pages and HTML table
    
    html_table = '<table class="result-table">\n'
    
    html_table += '<tr>\n'
    html_table += '<td>{}</td>\n'.format('True category')
    for sub_page_token in sub_page_tokens:
        html_table += '<td>{}</td>'.format(sub_page_token)
    html_table += '</tr>\n'
        
    filename_to_results_im = {im['file']:im for im in md_formatted_results['images']}
    
    sub_page_token_to_page_name = {
        'fp':'false positives',
        'tp':'true positives',
        'fn':'false negatives',
        'tn':'true negatives'
    }
    
    # category_name = ground_truth_category_names[0]
    for category_name in ground_truth_category_names:
        
        html_table += '<tr>\n'
        
        html_table += '<td>{}</td>\n'.format(category_name)
        
        # sub_page_token = sub_page_tokens[0]
        for sub_page_token in sub_page_tokens:
            
            html_table += '<td>\n'
            
            image_list = category_name_to_image_lists[category_name][sub_page_token]
            
            if len(image_list) == 0:
                
                html_table += '0\n'
                
            else:
                
                html_image_list_options = {}
                title_string = '{}: {}'.format(category_name,sub_page_token_to_page_name[sub_page_token])
                html_image_list_options['headerHtml'] = '<h1>{}</h1>'.format(title_string)
                
                target_html_file_relative = '{}_{}.html'.format(category_name,sub_page_token)
                target_html_file_abs = os.path.join(preview_folder,target_html_file_relative)
                
                html_image_info_list = []
                
                # image_filename_relative = image_list[0]
                for image_filename_relative in image_list:
                    
                    source_file = os.path.join(image_folder,image_filename_relative)
                    assert os.path.isfile(source_file)
                    
                    html_image_info = {}
                    detections = filename_to_results_im[image_filename_relative]['detections']
                    if len(detections) == 0:
                        max_conf = 0
                    else:
                        max_conf = max([d['conf'] for d in detections])
                    
                    title = '<b>Image</b>: {}, <b>Max conf</b>: {:0.3f}'.format(
                        image_filename_relative, max_conf)
                    image_link = 'images/' + os.path.basename(
                        _image_to_output_file(image_filename_relative,preview_images_folder))
                    html_image_info = {
                        'filename': image_link,
                        'title': title,
                        'linkTarget': source_file,
                        'textStyle':\
                         'font-family:verdana,arial,calibri;font-size:80%;' + \
                             'text-align:left;margin-top:20;margin-bottom:5'
                    }                
                    
                    html_image_info_list.append(html_image_info)
                    
                # ...for each image
                    
                write_html_image_list(
                    filename=target_html_file_abs,
                    images=html_image_info_list,
                    options=html_image_list_options)
    
                html_table += '<a href="{}">{}</a>\n'.format(target_html_file_relative,len(image_list))
            
            html_table += '</td>\n'
    
        # ...for each sub-page
            
        html_table += '</tr>\n'
    
    # ...for each category
        
    html_table += '</table>'        
    
    html = '<html>\n'
    
    style_header = """<head>
        <style type="text/css">
        a { text-decoration: none; }
        body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
        div.contentdiv { margin-left: 20px; }
        table.result-table { border:1px solid black; border-collapse: collapse; margin-left:50px;}
        td,th { padding:10px; }
        .rotate {    
          padding:0px;
          writing-mode:vertical-lr;
          -webkit-transform: rotate(-180deg);        
          -moz-transform: rotate(-180deg);            
          -ms-transform: rotate(-180deg);         
          -o-transform: rotate(-180deg);         
          transform: rotate(-180deg);
        }
        </style>
        </head>"""
        
    html += style_header + '\n'
    
    html += '<body>\n'
    
    html += '<h1>Results summary for {}</h1>\n'.format(job_name)
    
    if model_file is not None and len(model_file) > 0:
        html += '<p><b>Model file</b>: {}</p>'.format(os.path.basename(model_file))
    
    html += '<p><b>Confidence thresholds</b></p>'
    
    for c in confidence_thresholds.keys():
        html += '<p style="margin-left:15px;">{}: {}</p>'.format(c,confidence_thresholds[c])
    
    html += '<h2>Confusion matrix</h2>\n'
    
    html += '<p>...assuming a single category per image.</p>\n'
    
    html += '<img src="{}"/>\n'.format(cm_figure_fn_relative)
    
    html += '<h2>Confusion matrix (with links)</h2>\n'
    
    html += '<p>...assuming a single category per image.</p>\n'
    
    html += html_confusion_matrix
    
    html += '<h2>Per-class statistics</h2>\n'
    
    html += html_table
    
    html += '</body>\n'
    html += '<html>\n'
    
    target_html_file = os.path.join(preview_folder,'index.html')
    
    with open(target_html_file,'w') as f:
        f.write(html)
    
    
    ##%% Prepare return data
    
    confusion_matrix_info = {}
    confusion_matrix_info['html_file'] = target_html_file
    
    return confusion_matrix_info

# ...render_detection_confusion_matrix(...)
