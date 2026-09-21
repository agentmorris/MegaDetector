#%% Header

"""
Compare a variety of inference options (models, preprocessing, image sizes, etc.) on a folder
of images.
"""


#%% Imports and constants

import os

input_folder = 'c:/temp/batch-test-images'
test_folder_base = 'c:/temp/batch-comparisons'
tiling_folder = 'c:/temp/batch-tiles'

from megadetector.utils.path_utils import windows_path_to_wsl_path
from megadetector.utils.ct_utils import environment_is_wsl

if environment_is_wsl():
    input_folder = windows_path_to_wsl_path(input_folder)
    test_folder_base = windows_path_to_wsl_path(test_folder_base)
    tiling_folder = windows_path_to_wsl_path(tiling_folder)

json_output_folder = os.path.join(test_folder_base,'json_files')
visualization_folder = os.path.join(test_folder_base,'visualization')

model_names = ['mdv5a','mdv1000-redwood'] # ['mdv5a','mdv5b','mdv1000-redwood']
compatibility_modes = ['modern','classic'] # ['modern','classic']

model_name_to_model_file = \
    {'some-model-name':
     r"c:/some-model-file.pt"}

image_sizes = [None,1600,1920] # [None,1600,1920]
augmentation_states = ['noaug','aug'] # ['noaug','aug']
tiling_states = ['tiling','no-tiling'] # ['tiling','no-tiling']

n_gpus = 2

# process-based parallelism is not supported in this notebook on Windows
worker_type = 'thread'

if environment_is_wsl():
    worker_type = 'process'

tile_size_x = 1280
tile_size_y = 1280
tile_overlap = 0.5

rendering_threshold = 0.025

# Make sure rfdetr is available if we're using RFDETR models
for model_name in model_names:
    model_file = model_name
    if model_name in model_name_to_model_file:
        model_file = model_name_to_model_file[model_name]
    if 'detr' in model_file or 'detr' in model_name:
        import rfdetr

#%% Derived constants and support functions

from megadetector.detection.run_detector import load_and_run_detector
from megadetector.utils.path_utils import open_file

def parameters_to_name(parameters):
    fields = sorted(list(parameters.keys()))
    values = [str(parameters[field]) for field in fields]
    return '_'.join(values).lower()


#%% Enumerate images

from megadetector.utils.path_utils import find_images

images = find_images(input_folder,recursive=True)
print('Found {} images'.format(len(images)))


#%% Define jobs

from megadetector.detection import run_detector
from megadetector.detection.run_detector_batch import \
    load_and_run_detector_batch, write_results_to_file
from megadetector.detection.run_tiled_inference import run_tiled_inference

all_job_info = []

include_finished_jobs = True

# model_name = model_names[0]
for model_name in model_names:

    for compatibility_mode in compatibility_modes:

        for image_size in image_sizes:

            for augmentation_state in augmentation_states:

                for tiling_state in tiling_states:

                    params = {}
                    params['model_name'] = model_name
                    params['compatibility_mode'] = compatibility_mode
                    params['image_size'] = image_size
                    params['aug'] = augmentation_state
                    params['tiling'] = tiling_state

                    job_name = parameters_to_name(params)
                    job_output_file = os.path.join(json_output_folder,job_name + '.json')

                    if os.path.isfile(job_output_file) and (not include_finished_jobs):
                        print('{} exists, skipping'.format(job_output_file))
                        continue

                    job_info = {}
                    job_info['job_id'] = len(all_job_info)
                    job_info['job_name'] = job_name
                    job_info['job_output_file'] = job_output_file
                    job_info['params'] = params
                    all_job_info.append(job_info)

                # ...for each tiling size

            # ...for each augmentation state

        # ...for each inference size

    # ...for each compatibility mode

# ...for each model

print('Assembled {} jobs'.format(len(all_job_info)))
print('Total inference equivalent to {} images'.format(len(images)*len(all_job_info)))


#%% Create tiles if necessary

# We do this just once; the tiles will be re-used across all jobs where tiling
# is enabled.

overwrite_tiles = True

if ('tiling' in tiling_states) and (tiling_folder is not None):

    _ = run_tiled_inference(model_file=None,
                            image_folder=input_folder,
                            tiling_folder=tiling_folder,
                            output_file=None,
                            tile_size_x=tile_size_x,
                            tile_size_y=tile_size_y,
                            tile_overlap=tile_overlap,
                            checkpoint_path=None,
                            checkpoint_frequency=-1,
                            remove_tiles=False,
                            yolo_inference_options=None,
                            n_patch_extraction_workers=4,
                            overwrite_tiles=overwrite_tiles,
                            image_list=None,
                            augment=None,
                            detector_options=None,
                            use_image_queue=False, # Windows IPython issue
                            preprocess_on_image_queue=None,
                            inference_size=None,
                            pool_type='thread',
                            load_cached_tiles_if_available=False,
                            create_tiles_only=True)


#%% Run jobs (support functions)

overwrite_job_results = True

def run_job(job_info):

    job_id = job_info['job_id']
    job_name = job_info['job_name']
    job_output_file = job_info['job_output_file']
    model_name = job_info['params']['model_name']
    compatibility_mode = job_info['params']['compatibility_mode']
    image_size = job_info['params']['image_size']
    augmentation_state = job_info['params']['aug']
    tiling_state = job_info['params']['tiling']

    print('Processing job {} ({}) to {}'.format(job_id,job_name,job_output_file))

    augment = (augmentation_state == 'aug')

    assert tiling_state in ('no-tiling','tiling')

    if os.path.isfile(job_output_file) and (not overwrite_job_results):
        print('Output file exists, skipping')
        return

    detector_options = {'compatibility_mode':compatibility_mode}

    if 'gpu_index' in job_info:
        assert isinstance(job_info['gpu_index'],int)
        detector_options['device'] = 'cuda:' + str(job_info['gpu_index'])

    if model_name in model_name_to_model_file:
        model_file = model_name_to_model_file[model_name]
    else:
        model_file = model_name

    if tiling_state == 'no-tiling':

        r = load_and_run_detector_batch(model_file=model_file,
                                        image_file_names=images,
                                        checkpoint_path=None,
                                        confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                                        checkpoint_frequency=-1,
                                        results=None,
                                        n_cores=1,
                                        use_image_queue=False, # Windows IPython issue
                                        quiet=True,
                                        image_size=image_size,
                                        class_mapping_filename=None,
                                        include_image_size=False,
                                        include_image_timestamp=False,
                                        include_exif_tags=None,
                                        augment=augment,
                                        force_model_download=False,
                                        detector_options=detector_options,
                                        loader_workers=4,
                                        preprocess_on_image_queue=False,
                                        batch_size=1,
                                        verbose_output=False)

        write_results_to_file(results=r,
                              output_file=job_output_file,
                              relative_path_base=input_folder,
                              detector_file=model_name,
                              info=None,
                              include_max_conf=False,
                              custom_metadata=None,
                              force_forward_slashes=True)

    else:

        assert tiling_state == 'tiling'
        overwrite_tiles = False
        remove_tiles = (tiling_folder is None)
        load_cached_tiles_if_available = (tiling_folder is not None)

        _ = run_tiled_inference(model_file=model_file,
                                image_folder=input_folder,
                                tiling_folder=tiling_folder,
                                output_file=job_output_file,
                                tile_size_x=tile_size_x,
                                tile_size_y=tile_size_y,
                                tile_overlap=tile_overlap,
                                checkpoint_path=None,
                                checkpoint_frequency=-1,
                                remove_tiles=remove_tiles,
                                yolo_inference_options=None,
                                n_patch_extraction_workers=4,
                                overwrite_tiles=overwrite_tiles,
                                image_list=None,
                                augment=augment,
                                detector_options=detector_options,
                                use_image_queue=False, # Windows IPython issue
                                preprocess_on_image_queue=None,
                                inference_size=image_size,
                                pool_type='thread',
                                load_cached_tiles_if_available=load_cached_tiles_if_available,
                                create_tiles_only=False)

# ...def run_job(...)


#%% Run jobs (execution)

from concurrent.futures import ProcessPoolExecutor

import threading
import random

random.seed(0)

if n_gpus <= 1:

    # i_job = 0; job_info = all_job_info[i_job]
    for i_job,job_info in enumerate(all_job_info):
        print('Running job index {} of {}'.format(i_job,len(all_job_info)))
        run_job(job_info)

else:

    # Randomly permute the list "job_info", we do this so that the slower jobs (e.g. with larger
    # image sizes) don't all get assigned to the same GPU.
    random.shuffle(all_job_info)

    # Partition jobs by GPU index (round-robin)
    gpu_job_lists = [[] for _ in range(n_gpus)]
    for i_job,job_info in enumerate(all_job_info):
        gpu_index = i_job % n_gpus
        job_info['gpu_index'] = gpu_index
        gpu_job_lists[gpu_index].append(job_info)

    def run_gpu_jobs(gpu_jobs):
        for job_info in gpu_jobs:
            run_job(job_info)

    assert worker_type in ('thread','process'), \
        'Invalid worker_type: {}'.format(worker_type)

    if worker_type == 'thread':

        threads = []
        for gpu_index in range(n_gpus):
            t = threading.Thread(target=run_gpu_jobs,args=(gpu_job_lists[gpu_index],))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    else:

        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            futures = [executor.submit(run_gpu_jobs,gpu_job_lists[gpu_index]) \
                       for gpu_index in range(n_gpus)]
            for future in futures:
                future.result()

# ...if we need to parallelize execution


#%% Visualize output for every job

import html

from megadetector.visualization.visualize_detector_output import \
    visualize_detector_output
from megadetector.visualization.visualization_utils import \
    DEFAULT_BOX_THICKNESS, DEFAULT_LABEL_FONT_SIZE
from megadetector.utils import write_html_image_list

html_output_files = []

# job_info = all_job_info[0]
for i_job,job_info in enumerate(all_job_info):

    file_to_visualize = job_info['job_output_file']
    assert os.path.isfile(file_to_visualize)

    job_visualization_folder = os.path.join(visualization_folder,job_info['job_name'])
    html_output_file = os.path.join(job_visualization_folder,'index.html')
    job_info['html_output_file'] = html_output_file

    print('Visualizing job {} of {}: {}'.format(
        i_job,len(all_job_info),job_info['job_name']))

    html_options = write_html_image_list.write_html_image_list()
    html_options['pageTitle'] = 'Comparison results: {}'.format(job_info['job_name'])
    html_options['headerHtml'] = '<h2>Comparison results: {}</h2>\n'.format(job_info['job_name'])

    job_info['html_output_file'] = html_output_file

    _ = visualize_detector_output(detector_output_path=file_to_visualize,
                                  out_dir=job_visualization_folder,
                                  images_dir=input_folder,
                                  confidence_threshold=rendering_threshold,
                                  sample=1000,
                                  output_image_width=1200,
                                  random_seed=0,
                                  render_detections_only=False,
                                  classification_confidence_threshold=None,
                                  html_output_file=html_output_file,
                                  html_output_options=html_options,
                                  preserve_path_structure=False,
                                  parallelize_rendering=True,
                                  parallelize_rendering_n_cores=10,
                                  parallelize_rendering_with_threads=True,
                                  category_names_to_blur=None,
                                  link_images_to_originals=True,
                                  box_thickness=2,
                                  box_expansion=2,
                                  label_font_size=12)

    html_output_files.append(html_output_file)

# ...for each job


#%% Generate the comparison index page

comparison_index_page = os.path.join(visualization_folder,'index.html')

# Columns after the job name, as (parameter name, column header) tuples
index_page_columns = [('model_name','model name'),
                      ('compatibility_mode','compatibility mode'),
                      ('image_size','image size'),
                      ('aug','augmentation'),
                      ('tiling','tiling')]

index_page_head = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Comparison results</title>
<style>
body { font-family: sans-serif; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: left; }
th { background-color: #eee; cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { background-color: #ddd; }
tbody tr:nth-child(even) { background-color: #f6f6f6; }
tbody tr:hover { background-color: #e8f0fe; }
.sort_indicator { display: inline-block; width: 1.2em; text-align: right; }
</style>
<script>
var sortColumn = -1;
var sortAscending = true;

function sortTable(iColumn) {

    var table = document.getElementById('job_table');
    var tbody = table.tBodies[0];

    if (iColumn === sortColumn) {
        sortAscending = !sortAscending;
    } else {
        sortColumn = iColumn;
        sortAscending = true;
    }

    // Break ties using each row's current position, so the sort is stable regardless of
    // the browser's sort implementation
    var rows = Array.from(tbody.rows).map(function(row, i) { return [row, i]; });
    rows.sort(function(a, b) {
        var aText = a[0].cells[iColumn].textContent;
        var bText = b[0].cells[iColumn].textContent;
        var c = aText.localeCompare(bText, undefined, {numeric: true});
        if (!sortAscending) { c = -c; }
        return (c !== 0) ? c : (a[1] - b[1]);
    });
    rows.forEach(function(r) { tbody.appendChild(r[0]); });

    var headers = table.tHead.rows[0].cells;
    for (var i = 0; i < headers.length; i++) {
        var indicator = headers[i].getElementsByClassName('sort_indicator')[0];
        indicator.textContent = (i === iColumn) ? (sortAscending ? '▲' : '▼') : '';
    }
}
</script>
</head>
"""

with open(comparison_index_page,'w',encoding='utf-8') as f:

    f.write(index_page_head)
    f.write('<body>\n')
    f.write('<h2>Comparison results ({} jobs)</h2>\n'.format(len(all_job_info)))
    f.write('<p>Click a column header to sort</p>\n')
    f.write('<table id="job_table">\n<thead>\n<tr>\n')
    column_headers = ['job name'] + [c[1] for c in index_page_columns]

    for i_column,column_header in enumerate(column_headers):
        f.write('<th onclick="sortTable({})">{}<span class="sort_indicator"></span></th>\n'.format(
            i_column,column_header))

    f.write('</tr>\n</thead>\n<tbody>\n')

    for job_info in sorted(all_job_info,key=lambda x: x['job_name']):

        f.write('<tr><td><a href="{}">{}</a></td>'.format(
            html.escape(job_info['html_output_file']),
            html.escape(job_info['job_name'])))
        for param_name,_ in index_page_columns:
            f.write('<td>{}</td>'.format(
                html.escape(str(job_info['params'][param_name]).lower())))
        f.write('</tr>\n')

    # ...for each job

    f.write('</tbody>\n</table>\n</body>\n</html>\n')

# ...with open(...)

open_file(comparison_index_page)


#%% Create a page for each image, with results from every model

# This cell assumes the visualizations have already been generated.

import shutil
from tqdm import tqdm
from megadetector.utils.write_html_image_list import write_html_image_list

images_relative = find_images(input_folder,recursive=True, return_relative_paths=True)
print('Found {} images'.format(len(images)))

image_review_folder = os.path.join(test_folder_base,'image_review')
os.makedirs(image_review_folder,exist_ok=True)

image_fn_relative_to_image_index_file = {}

for i_image,image_fn_relative in tqdm(enumerate(images_relative),
                                      total=len(images_relative)):

    image_info = []

    # For each job
    # i_job = 0; job_info = all_job_info[i_job]
    for i_job,job_info in enumerate(all_job_info):

        image_review_folder_this_image = os.path.join(image_review_folder,
          image_fn_relative.replace('\\','/').replace('/','_'))
        os.makedirs(image_review_folder_this_image,exist_ok=True)
        job_visualization_folder = os.path.join(visualization_folder,
                                                job_info['job_name'])
        assert os.path.isdir(job_visualization_folder)

        # Find the annotated version of this image for this job
        annotated_images = os.listdir(job_visualization_folder)
        source_image = [fn for fn in annotated_images if image_fn_relative in fn]
        assert len(source_image) == 1
        source_image_relative = source_image[0]
        source_image_abs = os.path.join(job_visualization_folder,source_image_relative)
        assert os.path.isfile(source_image_abs)

        # Copy it to the output folder, with a filename that includes the job name
        job_name = job_info['job_name']

        # This is more readable, but leads to very long filenames
        # target_image_relative = os.path.splitext(source_image_relative)[0] + '_' + job_name + \
        #    os.path.splitext(source_image_relative)[1]
        target_image_relative = 'job_{}_image_{}{}'.format(
            i_job,i_image,os.path.splitext(source_image_relative)[1])
        target_image_abs = os.path.join(image_review_folder_this_image,
                                        target_image_relative)
        shutil.copyfile(source_image_abs,target_image_abs)

        im = {}
        image_info.append(im)
        im['filename'] = target_image_relative
        im['title'] = job_name

    # ...for each job

    # Create an index page for this image
    options = write_html_image_list()
    index_file_this_image = os.path.join(image_review_folder_this_image,'index.html')
    write_html_image_list(filename=index_file_this_image,
                          images=image_info,
                          options=options)
    image_fn_relative_to_image_index_file[image_fn_relative] = \
        index_file_this_image

# ...for each image


#%% Generate the master index page

master_index_page = os.path.join(image_review_folder,'index.html')

index_page_head = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Comparison results per image</title>
<style>
body { font-family: sans-serif; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: left; }
th { background-color: #eee; cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { background-color: #ddd; }
tbody tr:nth-child(even) { background-color: #f6f6f6; }
tbody tr:hover { background-color: #e8f0fe; }
.sort_indicator { display: inline-block; width: 1.2em; text-align: right; }
</style>
<body>"""

with open(master_index_page,'w') as f:
    f.write(index_page_head + '\n')

    for image_fn_relative in image_fn_relative_to_image_index_file:
        f.write('<a href="{}">{}</a><br/>\n'.format(
            image_fn_relative_to_image_index_file[image_fn_relative],
            image_fn_relative
        ))
    # ...for each image
    f.write('</body></html>')

open_file(master_index_page)


#%% Scrap

if False:

    pass

    #%% Find results files with failures in them

    import json

    jobs_with_failures = []

    # job_info = all_job_info[0]
    for i_job,job_info in enumerate(all_job_info):

        job_output_file = job_info['job_output_file']
        with open(job_output_file,'r') as f:
            d = json.load(f)
        for im in d['images']:
            if 'failure' in im:
                assert ('detections' not in im) or (im['detections'] is None)
                jobs_with_failures.append(job_info)
                break

    print('{} of {} files have failures'.format(
        len(jobs_with_failures),len(all_job_info)))


    #%% One-off fix for incorrect category IDs

    import json
    from megadetector.utils.ct_utils import write_json

    # job_info = all_job_info[0]
    for i_job,job_info in enumerate(all_job_info):

        job_output_file = job_info['job_output_file']
        with open(job_output_file,'r') as f:
            d = json.load(f)

        assert len(d['detection_categories']) == 3
        assert not ('0' in d['detection_categories'] and '3' in d['detection_categories'])

        categories_used = set()

        for im in d['images']:
            for det in im['detections']:
                categories_used.add(det['category'])
        assert not ('0' in categories_used and '3' in categories_used)
        offset_categories = ('0' in categories_used)

        if offset_categories:
            print('Offsetting categories in {}'.format(job_output_file))
            for im in d['images']:
                for det in im['detections']:
                    det['category'] = str(int(det['category'])+1)

        rewrite_category_mappings = '0' in d['detection_categories']

        if rewrite_category_mappings:
            print('Re-writing categories in {}'.format(job_output_file))
            d['detection_categories'] = {}
            d['detection_categories']['1'] = 'animal'
            d['detection_categories']['2'] = 'person'
            d['detection_categories']['3'] = 'vehicle'

        if rewrite_category_mappings or offset_categories:
            print('Re-writing output file {}'.format(job_output_file))
            write_json(job_output_file,d)

    # ...for each job


    #%% Review versions of one image

    # This cell assumes the visualizations have already been generated.

    import shutil

    image_fn_relative = 'IMG_0809.JPG'
    image_review_folder = os.path.join(test_folder_base,'image_review')
    os.makedirs(image_review_folder,exist_ok=True)

    image_info = []

    # For each job
    # job_info = all_job_info[0]
    for job_info in all_job_info:

        job_visualization_folder = os.path.join(visualization_folder,job_info['job_name'])
        assert os.path.isdir(job_visualization_folder)

        # Find the annotated version of this image for this job
        annotated_images = os.listdir(job_visualization_folder)
        source_image = [fn for fn in annotated_images if image_fn_relative in fn]
        assert len(source_image) == 1
        source_image_relative = source_image[0]
        source_image_abs = os.path.join(job_visualization_folder,source_image_relative)
        assert os.path.isfile(source_image_abs)

        # Copy it to the output folder, with a filename that includes the job name
        job_name = job_info['job_name']

        target_image_relative = os.path.splitext(source_image_relative)[0] + '_' + job_name + \
            os.path.splitext(source_image_relative)[1]
        target_image_abs = os.path.join(image_review_folder,target_image_relative)
        shutil.copyfile(source_image_abs,target_image_abs)

        im = {}
        image_info.append(im)
        im['filename'] = target_image_relative
        im['title'] = job_name

    # Create an index page
    from megadetector.utils.write_html_image_list import write_html_image_list

    options = write_html_image_list()
    index_file = os.path.join(image_review_folder,'index.html')
    write_html_image_list(filename=index_file,
                            images=image_info,
                            options=options)

    from megadetector.utils.path_utils import open_file
    open_file(index_file, attempt_to_open_in_wsl_host=True)


    #%% Open existing visualizations for specific jobs

    all_job_names = [x['job_name'] for x in all_job_info]

    def find_matching_jobs(html_group_params):

        matching_jobs = []
        for job_info in all_job_info:
            job_matches = True
            for param_name in html_group_params['params']:
                assert param_name in job_info['params'], \
                    'No value for {}'.format(param_name)
                # Empty lists for a parameter means that all jobs match for that
                # parameter
                if (html_group_params['params'][param_name] is None) or \
                    (len(html_group_params['params'][param_name]) == 0):
                    continue
                if job_info['params'][param_name] not in html_group_params['params'][param_name]:
                    job_matches = False
                    break
            if job_matches:
                matching_jobs.append(job_info)
        return matching_jobs

    html_groups = []

    html_group = {}
    html_group['name'] = 'test'
    html_group['params'] = {}
    html_group['params']['model_name'] = ['mdv5a']
    html_group['params']['image_size'] = None # [1600]
    html_group['params']['compatibility_mode'] = None # ['modern']
    html_group['params']['aug'] = None # ['noaug','aug']
    html_group['params']['tiling'] = ['tiling']
    html_group['matching_jobs'] = find_matching_jobs(html_group)

    # Choose a specific subset of jobs by name
    if False:
        html_group = {}
        html_group['name'] = 'no-tiling-best'
        html_group['matching_names'] = [
            'aug_modern_none_mdv1000-redwood_no-tiling',
            'aug_modern_none_mdv5a_no-tiling',
            'aug_classic_none_mdv1000-redwood_no-tiling',
            'aug_classic_1600_mdv1000-redwood_no-tiling',
            'noaug_modern_none_mdv1000-redwood_no-tiling'
        ]
        html_group['matching_jobs'] = [x for x in all_job_info if x['job_name'] in html_group['matching_names']]
        assert len(html_group['matching_jobs']) == len(html_group['matching_names'])


    print('Found {} matching jobs for {}'.format(
        len(html_group['matching_jobs']),
        html_group['name']))

    for matching_job in html_group['matching_jobs']:
        open_file(matching_job['html_output_file'],attempt_to_open_in_wsl_host=True)
