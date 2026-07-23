#%% Header

"""

After initiating .csv downloads from one or more Wildlife Insights projects, download the corresponding
images and convert labels to COCO.

This notebook expects a single base folder, with a subfolder called "csv_downloads"; unzip
WI .csv zipfiles there.  A parallel folder called "images" will be created for image downloads.

E.g.:

c:\temp\wi-test
  csv-downloads
    wildlife-insights_f108491f-4724-442c-8073-0b3ac74ac5d7_project-2013431_data
      projects.csv
      deployments.csv
      images_2013431.csv
  images

"""


#%% Recommended environment settings

"""
gcloud config set disable_usage_reporting true
gcloud config set core/disable_file_logging True
gcloud config set component_manager/disable_update_check true
gcloud auth login
"""

#%% Imports and constants

import os
import json

from tqdm import tqdm

from megadetector.utils.wi_platform_utils import read_images_from_download_bundle
from megadetector.utils.wi_platform_utils import write_download_commands
from megadetector.utils.wi_platform_utils import write_prefix_download_command
from megadetector.utils.ct_utils import is_empty

# Should we download individual images, or whole buckets?
download_individual_images = False

download_blank_images = True
download_unidentified_images = True

# This determines the parallelism of the download process.  Only meaningful if
# download_individual_images is False.
n_download_workers = 25

force_generate_download_commands = True
force_download = False

if os.name == 'nt':
    script_extension = '.bat'
else:
    script_extension = '.sh'

projects = []

project_base = 'e:/data/project-nanme'
assert os.path.isdir(project_base)

project_info_cache_file = os.path.join(project_base,'project_info.json')
image_base_folder = os.path.join(project_base,'images')
csv_base = os.path.join(project_base,'csv_downloads')

p = {}
p['name'] = 'Project One'
p['id'] = 2001111
projects.append(p)

p = {}
p['name'] = 'Project Two'
p['id'] = 2001112
projects.append(p)


#%% Find download folders

project_folders_relative = os.listdir(csv_base)

project_id_to_download_folder = {}

for folder_name in project_folders_relative:

    # E.g.:
    #
    # wildlife-insights_e81cf866-face-4722-9310-04d51768a23d_project-2003085_data
    project_id = int(folder_name.split('project-')[1].split('_')[0])
    assert project_id not in project_id_to_download_folder
    project_id_to_download_folder[project_id] = folder_name

for i_project,p in enumerate(projects):

    project_download_folder = project_id_to_download_folder[p['id']]
    p['project_download_folder'] = project_download_folder


#%% Prepare download scripts

unidentified_images = []
blank_mismatches = []
blank_images = []

if not download_blank_images:
    assert download_individual_images, \
        "Can't skip blank images if we're downloading whole buckets"

if not download_unidentified_images:
    assert download_individual_images, \
        "Can't skip unidentified images if we're downloading whole buckets"

# i_project = 1; p = projects[i_project]
for i_project,p in enumerate(projects):

    project_id = str(p['id'])

    print('Processing project {} of {} ({})'.format(
        i_project,len(projects),project_id))

    project_image_folder = os.path.join(image_base_folder,project_id)
    download_command_file = \
        os.path.join(project_image_folder,'download_images_{}{}'.format(
            project_id,script_extension))

    if os.path.isfile(download_command_file) and (not force_generate_download_commands):
        print('Download command file {} exists, skipping'.format(
            download_command_file))
        continue

    download_folder_relative = p['project_download_folder']
    download_folder_abs = os.path.join(csv_base,download_folder_relative)
    image_records = read_images_from_download_bundle(download_folder_abs)

    image_records_flattened = []
    for x in image_records.values():
        assert isinstance(x,list)
        image_records_flattened.extend(x)
    image_records = image_records_flattened

    image_records_to_download = []

    # r = image_records[0]
    for r in tqdm(image_records):

        if is_empty(r['identified_by']):
            unidentified_images.append(r)
            if not download_unidentified_images:
                continue

        is_blank = r['is_blank']
        assert is_blank in (0,1)

        # Sometimes common_name is NaN... this is a platform bug, there's no
        # good reason for this in the cases where I see this.
        if isinstance(r['common_name'],str):
            if ((is_blank == 1) and (r['common_name'].lower() != 'blank')) or \
            ((is_blank == 0) and (r['common_name'].lower() == 'blank')):
                blank_mismatches.append(r)

        if is_blank or \
            (isinstance(r['common_name'],str) and (r['common_name'].lower() == 'blank')):
            blank_images.append(r)
            # Optionally skip blanks
            if not download_blank_images:
                continue

        n = r['number_of_objects']
        assert isinstance(n,int) and (n >= 0)

        image_records_to_download.append(r)

    # ...for each record

    print('Found {} blank images (of {})'.format(
        len(blank_images),len(image_records)))
    print('Found {} unidentified images (of {})'.format(
        len(unidentified_images),len(image_records)))

    print('Downloading {} of {} images'.format(
        len(image_records_to_download),len(image_records)))

    os.makedirs(project_image_folder,exist_ok=True)

    image_records_file = os.path.join(project_image_folder,'image_records.json')
    with open(image_records_file,'w') as f:
        json.dump(image_records_to_download,f,indent=1)
    print('Wrote image records to {}'.format(image_records_file))

    image_urls_to_download = [r['location'] for r in image_records_to_download]

    p['image_urls_to_download'] = image_urls_to_download

    if download_individual_images:
        write_download_commands(image_records=image_records_to_download,
                                download_dir_base=project_image_folder,
                                force_download=False,
                                n_download_workers=n_download_workers)
    else:
        write_prefix_download_command(image_records=image_records_to_download,
                                      download_dir_base=project_image_folder,
                                      download_command_file=download_command_file)

# ...for each project


#%% Save or load download information

if os.path.isfile(project_info_cache_file):

    print('Loading project info from {}'.format(project_info_cache_file))
    with open(project_info_cache_file,'r') as f:
        projects = json.load(f)

else:

    with open(project_info_cache_file,'w') as f:
        json.dump(projects,f,indent=1)
    print('Wrote project cache to {}'.format(project_info_cache_file))


#%% Check download completion

from megadetector.utils.wi_platform_utils import url_to_relative_path
from megadetector.utils.path_utils import recursive_file_list

n_placeholders = 0

# i_project = 0; p = projects[i_project]
for i_project,p in enumerate(projects):

    project_id = p['id']
    project_image_folder_abs = os.path.join(image_base_folder,str(project_id))

    print('Enumerating files in {}'.format(project_image_folder_abs))
    downloaded_images_relative = recursive_file_list(project_image_folder_abs,
                                                     return_relative_paths=True)

    downloaded_images_relative = set(downloaded_images_relative)
    missing_files = []

    relative_paths_requested = set()

    # url = p['image_urls_to_download'][0]
    for url in p['image_urls_to_download']:
        if 'placeholder' in url:
            n_placeholders += 1
            continue
        relative_path = url_to_relative_path(url)
        relative_paths_requested.add(relative_path)
        if relative_path not in downloaded_images_relative:
            missing_files.append(relative_path)

    extra_files = []

    for relative_path in downloaded_images_relative:
        if relative_path not in relative_paths_requested:
            extra_files.append(relative_path)

    print('Found {} images for project {} ({}): {} missing, {} placeholder, {} extra'.format(
            len(downloaded_images_relative),
            i_project,
            project_id,
            len(missing_files),
            n_placeholders,
            len(extra_files)))

# ...for each project


#%% Delete redundant thumbnails (prep)

# Only necessary for whole-bucket downloads

from megadetector.utils.path_utils import recursive_file_list

files_to_delete = []

print('Enumerating files in {}'.format(project_base))
downloaded_images_relative = recursive_file_list(image_base_folder,
                                                 return_relative_paths=True)

downloaded_images_relative = set(downloaded_images_relative)

# i_file = 0; relative_path = downloaded_images_relative[i_file]
for i_file,relative_path in tqdm(enumerate(downloaded_images_relative),
                                    total=len(downloaded_images_relative)):
    if ('_500' in relative_path) and \
        (relative_path.replace('_500','') in downloaded_images_relative):
        absolute_path = os.path.join(image_base_folder,relative_path)
        assert os.path.isfile(absolute_path)
        files_to_delete.append(absolute_path)

print('Identified {} redundant thumbnails (of {} images)'.format(
    len(files_to_delete),
    len(downloaded_images_relative)))

for fn in files_to_delete:
    assert '_500' in fn


#%%  Delete redundant thumbnails (execution)

from megadetector.utils.path_utils import parallel_delete_files
parallel_delete_files(input_files=files_to_delete)


#%% Find image/csv folders

def is_int_string(s):
    try:
        _ = int(s)
        return True
    except Exception:
        return False

project_image_folders = os.listdir(image_base_folder)
project_image_folders = [fn for fn in project_image_folders if is_int_string(fn)]
project_image_folders = [os.path.join(image_base_folder,fn) for fn in project_image_folders]

project_csv_folders = os.listdir(csv_base)
project_csv_folders = [fn for fn in project_csv_folders if fn.endswith('_data')]
project_csv_folders = [os.path.join(csv_base,fn) for fn in project_csv_folders]

print('Found {} project image folders and {} project csv folders'.format(
    len(project_image_folders),
    len(project_csv_folders)
))

assert len(project_image_folders) == len(projects)
assert len(project_csv_folders) == len(projects)

for fn in project_image_folders:
    assert os.path.isdir(fn)

for fn in project_csv_folders:
    assert os.path.isdir(fn)


#%% Run COCO conversions

from megadetector.data_management.wi_download_csv_to_coco import wi_download_csv_to_coco

force_coco_conversion = True

# i_project = 0; project_image_folder = project_image_folders[i_project]
for i_project,project_image_folder in enumerate(project_image_folders):

    project_id = project_image_folder.split('/')[-1]
    _ = int(project_id)
    current_project_csv_folders = [fn for fn in project_csv_folders if project_id in fn]
    assert len(current_project_csv_folders) == 1
    project_csv_folder = current_project_csv_folders[0]

    project_coco_file = os.path.join(project_image_folder,project_id + '.coco.json')

    if os.path.exists(project_coco_file) and (not force_coco_conversion):
        print('{} exists, skipping'.format(project_coco_file))
        continue

    print('Processing project {} of {}: {}'.format(
        i_project,len(project_image_folders),project_id))

    _ = wi_download_csv_to_coco(csv_file_in=project_csv_folder,
                                coco_file_out=project_coco_file,
                                image_folder=project_image_folder,
                                exclude_missing_images=False,
                                image_flattening=None, # 'deployment',
                                verbose=True,
                                blank_disagreement_handling='trust_label',
                                include_blanks=True)

# ...for each project


#%% Create sequences

import json

from megadetector.data_management import cct_json_utils
from megadetector.data_management.cct_json_utils import SequenceOptions
from megadetector.utils.path_utils import insert_before_extension
from megadetector.utils.ct_utils import write_json

sequence_options = SequenceOptions()

# i_project = 0; project_image_folder = project_image_folders[i_project]
for i_project,project_image_folder in enumerate(project_image_folders):

    project_id = project_image_folder.split('/')[-1]
    _ = int(project_id)
    project_coco_file = os.path.join(project_image_folder,project_id + '.coco.json')
    assert os.path.isfile(project_coco_file)

    with open(project_coco_file,'r') as f:
        d = json.load(f)

    print('Assembling images into sequences')
    _ = cct_json_utils.create_sequences(d, options=sequence_options)

    project_coco_file_with_sequences = insert_before_extension(
        project_coco_file,'with_sequences')

    write_json(project_coco_file_with_sequences,d,serialize_datetimes=True)

# ...for each project


#%% Preview COCO conversions

from megadetector.visualization.visualize_db import \
    DbVizOptions, visualize_db

project_base = os.path.expanduser('~/tmp/wi-project-analysis')
preview_base = os.path.join(project_base,'coco-preview')
os.makedirs(preview_base,exist_ok=True)

viz_options = DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.viz_size = (1000, -1)
viz_options.html_options['maxFiguresPerHtmlFile'] = 1000
viz_options.sort_by_filename = True
viz_options.random_seed = 0
viz_options.classes_to_include = None
viz_options.classes_to_exclude = None
viz_options.multiple_categories_tag = '*multiple*'
viz_options.parallelize_rendering = True
viz_options.parallelize_rendering_with_threads = True
viz_options.parallelize_rendering_n_cores = 12
viz_options.create_category_pages = True

#: If this is None, we just sample images, and show images.  If this is
#: not None, we sample images, but we also show the other images in the sequences
#: containing our sampled images.  If this is <=0, there is no limit on the
#: number of images we'll show per sequences.  If this is >0, we will cap the number
#: of images shown per sequence; no guarantee is made about which images will
#: be selected in that case.  This only impacts the number of images added as
#: "sequence friends" of images that get sampled.
viz_options.max_sequence_length = 3

preview_filenames = []

# i_project = 0; project_image_folder = project_image_folders[i_project]
for i_project,project_image_folder in enumerate(project_image_folders):

    project_id = project_image_folder.split('/')[-1]
    _ = int(project_id)
    project_coco_file = os.path.join(project_image_folder,project_id + '.coco.with_sequences.json')
    assert os.path.isfile(project_coco_file)

    project_preview_dir = os.path.join(preview_base,project_id)

    print('Previewing COCO file {} to {}'.format(project_coco_file,
                                                 project_preview_dir))

    html_filename,_ = visualize_db(db_path=project_coco_file,
                                output_dir=project_preview_dir,
                                image_base_dir=project_image_folder,
                                options=viz_options)

    preview_filenames.append(html_filename)

# ...for each project


#%% Open preview visualizations

from megadetector.utils.path_utils import open_file
for fn in preview_filenames:
    open_file(fn)


#%% Sample images from each project for MD comparisons

import random

from collections import defaultdict

random.seed(0)
n_samples_per_project = 50
include_blanks_in_sample = False

absolute_filenames_to_copy = []

for i_project,project_image_folder in enumerate(project_image_folders):

    project_id = project_image_folder.split('/')[-1]
    _ = int(project_id)
    project_coco_file = os.path.join(project_image_folder,project_id + '.coco.with_sequences.json')
    assert os.path.isfile(project_coco_file)

    with open(project_coco_file,'r') as f:
        d = json.load(f)

    image_id_to_categories = defaultdict(set)

    category_id_to_name = {}
    for c in d['categories']:
        category_id_to_name[c['id']] = c['name']

    for ann in d['annotations']:
        image_id = ann['image_id']
        category_name = category_id_to_name[ann['category_id']]
        image_id_to_categories[image_id].add(category_name)

    relative_filenames_to_sample = []

    for im in d['images']:
        fn_relative = im['file_name']
        categories_this_image = image_id_to_categories[im['id']]
        if not include_blanks_in_sample:
            if len(categories_this_image) == 1 and 'empty' in categories_this_image:
                continue
        relative_filenames_to_sample.append(fn_relative)

    n_sample = min(n_samples_per_project, len(relative_filenames_to_sample))
    sampled_filenames = random.sample(relative_filenames_to_sample,n_sample)

    print('Sampled {} of {} candidates ({} total) in project {}'.format(
        len(sampled_filenames), len(relative_filenames_to_sample),
        len(d['images']), project_id))

    for fn_relative in sampled_filenames:
        fn_abs = os.path.join(project_image_folder, fn_relative)
        assert os.path.isfile(fn_abs)
        absolute_filenames_to_copy.append(fn_abs)

# ...for each project


#%% Copy samples

import shutil

sample_folder = os.path.join(project_base,'sample-images')
os.makedirs(sample_folder,exist_ok=True)

output_filenames_relative = set()

for fn_abs_in in tqdm(absolute_filenames_to_copy):
    fn_out_relative = os.path.basename(fn_abs_in)
    assert fn_out_relative not in output_filenames_relative
    output_filenames_relative.add(fn_out_relative)
    fn_out_abs = os.path.join(sample_folder,fn_out_relative)
    shutil.copyfile(fn_abs_in,fn_out_abs)
