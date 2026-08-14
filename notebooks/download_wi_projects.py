#%% Header

"""

After initiating .csv downloads from one or more Wildlife Insights projects, download the corresponding
images and convert labels to COCO.

This notebook expects a single base folder, with a subfolder called "csv_downloads"; unzip
WI .csv zipfiles there.  A parallel folder called "images" will be created for image downloads.

E.g.:

c:\temp\wi-test
  csv_downloads
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
from collections import defaultdict

from megadetector.utils.wi_platform_utils import read_images_from_download_bundle
from megadetector.utils.wi_platform_utils import read_sequences_from_download_bundle
from megadetector.utils.wi_platform_utils import write_download_commands
from megadetector.utils.wi_platform_utils import write_prefix_download_command
from megadetector.utils.ct_utils import is_empty

# Should we download individual images, or whole buckets?
download_individual_images = True

# All of these must be True if "download_individual_images" is False
download_blank_images = True
download_unidentified_images = True
download_identified_images = True

# This determines the parallelism of the download process.  Only meaningful if
# download_individual_images is True.  If download_individual_images is False, we rely on
# gcloud storage cp for parallelism.
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

print('Found {} projects'.format(len(projects)))


#%% Prepare download scripts

unidentified_images = []
skipped_identified_images = []
blank_mismatches = []
blank_images = []

if not download_blank_images:
    assert download_individual_images, \
        "Can't skip blank images if we're downloading whole buckets"

if not download_unidentified_images:
    assert download_individual_images, \
        "Can't skip unidentified images if we're downloading whole buckets"

if not download_identified_images:
    assert download_individual_images, \
        "Can't skip identified images if we're downloading whole buckets"

all_image_ids = set()

# i_project = 0; p = projects[i_project]
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

    # Dict mapping image IDs to image records
    image_records = read_images_from_download_bundle(download_folder_abs)

    # Dict mapping sequence IDs to sequence records
    sequence_records = read_sequences_from_download_bundle(download_folder_abs)

    image_records_flattened = []
    for x in image_records.values():
        assert isinstance(x,list)
        image_records_flattened.extend(x)
    image_records = image_records_flattened

    image_records_to_download = []

    missing_sequence_id_to_image_ids = defaultdict(list)

    # i_record = 0; r = image_records[i_record]
    for i_record,r in tqdm(enumerate(image_records),total=len(image_records)):

        all_image_ids.add(r['image_id'])

        # If this is a sequence-based project, find the sequence information
        # for this image
        sequence_record = None
        if ('sequence_id' in r) and (sequence_records is not None):
            if r['sequence_id'] not in sequence_records:
                print('Warning: sequence ID {} not found'.format(r['sequence_id']))
                missing_sequence_id_to_image_ids[r['sequence_id']].append(r['image_id'])
                continue
            sequence_record = sequence_records[r['sequence_id']][-1]

        # Optionally exclude unidentified images
        if 'identified_by' in r:
            identified_by = r['identified_by']
        else:
            identified_by = sequence_record['identified_by']

        # Is this a unidentified image?
        if is_empty(identified_by) or (identified_by.lower() == 'computer vision'):
            unidentified_images.append(r)
            if download_unidentified_images:
                image_records_to_download.append(r)
            continue

        # If we got this far, we have an identified image, so skip it if we're
        # not supposed to be downloading identified images.
        if not download_identified_images:
            skipped_identified_images.append(r)
            continue

        if 'is_blank' in r:
            is_blank = r['is_blank']
        else:
            is_blank = sequence_record['is_blank']

        assert is_blank in (0,1)

        # Sometimes common_name is NaN... this is a platform bug, there's no
        # good reason for this in the cases where I see this.
        if isinstance(r['common_name'],str):
            if ((is_blank == 1) and (r['common_name'].lower() != 'blank')) or \
            ((is_blank == 0) and (r['common_name'].lower() == 'blank')):
                blank_mismatches.append(r)

        # If either the "is_blank" field or the "common_name" field indicate that this image
        # is blank, treat it as blank (these can disagree sometimes).
        if is_blank or \
            (isinstance(r['common_name'],str) and (r['common_name'].lower() == 'blank')):
            blank_images.append(r)
            # Optionally skip blanks
            if not download_blank_images:
                continue

        if 'number_of_objects' in r:
            n = r['number_of_objects']
            assert isinstance(n,int) and (n >= 0)

        image_records_to_download.append(r)

    # ...for each record

    if len(missing_sequence_id_to_image_ids) > 0:
        n_missing_sequence_images = 0
        for seq_id in missing_sequence_id_to_image_ids:
            n_missing_sequence_images += len(missing_sequence_id_to_image_ids[seq_id])
        print('Warning: {} sequence IDs were missing ({} images)'.format(
            len(missing_sequence_id_to_image_ids),
            n_missing_sequence_images))

    print('Found {} unique image IDs'.format(len(all_image_ids)))

    print('Found {} unidentified image records (of {})'.format(
        len(unidentified_images),len(image_records)))

    if download_identified_images:
        assert len(skipped_identified_images) == 0
    else:
        print('Skipped {} identified image records (of {})'.format(
            len(skipped_identified_images),len(image_records)))

    print('Found {} blank image records (of {})'.format(
        len(blank_images),len(image_records)))

    print('Downloading {} of {} image records'.format(
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
from megadetector.utils.path_utils import is_image_file

n_placeholders = 0

# Don't count files as "extra downloaded files" if they were generated locally as
# part of the download process
ignore_tokens = ['download_wi_images','image_records']

# i_project = 0; p = projects[i_project]
for i_project,p in enumerate(projects):

    project_id = p['id']
    project_image_folder_abs = os.path.join(image_base_folder,str(project_id))

    print('Enumerating files in {}'.format(project_image_folder_abs))
    downloaded_files_relative = recursive_file_list(project_image_folder_abs,
                                                     return_relative_paths=True)
    downloaded_files_relative = set(downloaded_files_relative)
    missing_files = []
    matching_files = []

    relative_paths_requested = set()

    # url = p['image_urls_to_download'][0]
    for url in p['image_urls_to_download']:
        if 'placeholder' in url:
            n_placeholders += 1
            continue
        relative_path = url_to_relative_path(url)
        relative_paths_requested.add(relative_path)
        if relative_path in downloaded_files_relative:
            matching_files.append(relative_path)
        else:
            missing_files.append(relative_path)

    extra_files = []

    for relative_path in downloaded_files_relative:

        # Don't count files as "extra downloaded files" if they were generated locally as
        # part of the download process
        ignore_file = False
        for s in ignore_tokens:
            if s in relative_path:
                ignore_file = True
                break
        if ignore_file:
            continue

        if relative_path not in relative_paths_requested:
            extra_files.append(relative_path)

    print('Found {} files for project {} ({}):\n{} matching downloads, {} missing, {} placeholder, {} extra files'.format(
            len(downloaded_files_relative),
            i_project,
            project_id,
            len(matching_files),
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
                                image_flattening='deployment',
                                verbose=True,
                                blank_disagreement_handling='trust_label',
                                include_blanks=True)

# ...for each project


#%% Create sequences

import json
import shutil
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
    project_coco_file_with_sequences = insert_before_extension(
        project_coco_file,'with_sequences')

    assert os.path.isfile(project_coco_file) or \
        os.path.isfile(project_coco_file_with_sequences)

    if os.path.isfile(project_coco_file_with_sequences) and \
        (not os.path.isfile(project_coco_file)):
        print('Pre-sequence file already moved, skipping')
        continue

    with open(project_coco_file,'r') as f:
        d = json.load(f)

    n_images_with_sequence_information = 0

    for im in d['images']:
        if 'seq_id' in im:
            n_images_with_sequence_information += 1

    if n_images_with_sequence_information > 0:
        print('{} of {} images have sequence information, skipping sequence creation'.format(
            n_images_with_sequence_information,len(d['images'])))
        shutil.move(project_coco_file,project_coco_file_with_sequences)
        continue

    print('Assembling images into sequences')
    _ = cct_json_utils.create_sequences(d, options=sequence_options)

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
