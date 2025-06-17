"""

labelme_to_yolo.py

Create YOLO .txt files in a folder containing labelme .json files.

"""

#%% Imports

import os
import json

from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from tqdm import tqdm

from megadetector.utils.path_utils import recursive_file_list


#%% Main function

def labelme_file_to_yolo_file(labelme_file,
                              category_name_to_category_id,
                              yolo_file=None,
                              required_token=None,
                              overwrite_behavior='overwrite'):
    """
    Convert the single .json file labelme_file to yolo format, writing the results to the text
    file yolo_file (defaults to s/json/txt).

    If required_token is not None and the dict in labelme_file does not contain the key [required_token],
    this function no-ops (i.e., does not generate a YOLO file).

    overwrite_behavior should be 'skip' or 'overwrite' (default).

    Args:
        labelme_file (str): .json file to convert
        category_name_to_category_id (dict): category name --> ID mapping
        yolo_file (str, optional): output .txt file defaults to s/json/txt
        required_token (str, optional): only process filenames containing this token
        overwrite_behavior (str, optional): "skip" or "overwrite"
    """

    result = {}
    result['labelme_file'] = labelme_file
    result['status'] = 'unknown'

    assert os.path.isfile(labelme_file), 'Could not find labelme .json file {}'.format(labelme_file)
    assert labelme_file.endswith('.json'), 'Illegal labelme .json file {}'.format(labelme_file)

    if yolo_file is None:
        yolo_file = os.path.splitext(labelme_file)[0] + '.txt'

    if os.path.isfile(yolo_file):
        if overwrite_behavior == 'skip':
            result['status'] = 'skip-exists'
            return result
        else:
            assert overwrite_behavior == 'overwrite', \
                'Unrecognized overwrite behavior {}'.format(overwrite_behavior)

    with open(labelme_file,'r') as f:
        labelme_data = json.load(f)

    if required_token is not None and required_token not in labelme_data:
        result['status'] = 'skip-no-required-token'
        return result

    im_height = labelme_data['imageHeight']
    im_width = labelme_data['imageWidth']

    yolo_lines = []

    for shape in labelme_data['shapes']:

        assert shape['shape_type'] == 'rectangle', \
            'I only know how to convert rectangles to YOLO format'
        assert shape['label'] in category_name_to_category_id, \
            'Category {} not in category mapping'.format(shape['label'])
        assert len(shape['points']) == 2, 'Illegal rectangle'
        category_id = category_name_to_category_id[shape['label']]

        p0 = shape['points'][0]
        p1 = shape['points'][1]

        # Labelme: [[x0,y0],[x1,y1]] (arbitrarily sorted) (absolute coordinates)
        #
        # YOLO: [class, x_center, y_center, width, height] (normalized coordinates)
        minx_abs = min(p0[0],p1[0])
        maxx_abs = max(p0[0],p1[0])
        miny_abs = min(p0[1],p1[1])
        maxy_abs = max(p0[1],p1[1])

        if (minx_abs >= (im_width-1)) or (maxx_abs <= 0) or \
            (miny_abs >= (im_height-1)) or (maxy_abs <= 0):
                print('Skipping invalid shape in {}'.format(labelme_file))
                continue

        # Clip to [0,1]... it's not obvious that the YOLO format doesn't allow bounding
        # boxes to extend outside the image, but YOLOv5 and YOLOv8 get sad about boxes
        # that extend outside the image.
        maxx_abs = min(maxx_abs,im_width-1)
        maxy_abs = min(maxy_abs,im_height-1)
        minx_abs = max(minx_abs,0.0)
        miny_abs = max(miny_abs,0.0)

        minx_rel = minx_abs / (im_width-1)
        maxx_rel = maxx_abs / (im_width-1)
        miny_rel = miny_abs / (im_height-1)
        maxy_rel = maxy_abs / (im_height-1)

        assert maxx_rel >= minx_rel
        assert maxy_rel >= miny_rel

        xcenter_rel = (maxx_rel + minx_rel) / 2.0
        ycenter_rel = (maxy_rel + miny_rel) / 2.0
        w_rel = maxx_rel - minx_rel
        h_rel = maxy_rel - miny_rel

        yolo_line = '{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(category_id,
            xcenter_rel, ycenter_rel, w_rel, h_rel)
        yolo_lines.append(yolo_line)

    # ...for each shape

    with open(yolo_file,'w') as f:
        for s in yolo_lines:
            f.write(s + '\n')

    result['status'] = 'converted'
    return result


def labelme_folder_to_yolo(labelme_folder,
                           category_name_to_category_id=None,
                           required_token=None,
                           overwrite_behavior='overwrite',
                           relative_filenames_to_convert=None,
                           n_workers=1,
                           use_threads=True):
    """
    Given a folder with images and labelme .json files, convert the .json files
    to YOLO .txt format.  If category_name_to_category_id is None, first reads
    all the labels in the folder to build a zero-indexed name --> ID mapping.

    If required_token is not None and a labelme_file does not contain the key [required_token],
    it won't be converted.  Typically used to specify a field that indicates which files have
    been reviewed.

    If relative_filenames_to_convert is not None, this should be a list of .json (not image)
    files that should get converted, relative to the base folder.

    overwrite_behavior should be 'skip' or 'overwrite' (default).

    returns a dict with:
        'category_name_to_category_id', whether it was passed in or constructed
        'image_results': a list of results for each image (converted, skipped, error)

    Args:
        labelme_folder (str): folder of .json files to convert
        category_name_to_category_id (dict): category name --> ID mapping
        required_token (str, optional): only process filenames containing this token
        overwrite_behavior (str, optional): "skip" or "overwrite"
        relative_filenames_to_convert (list of str, optional): only process filenames on this list
        n_workers (int, optional): parallelism level
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallelism
    """

    if relative_filenames_to_convert is not None:
        labelme_files_relative = relative_filenames_to_convert
        assert all([fn.endswith('.json') for fn in labelme_files_relative]), \
            'relative_filenames_to_convert contains non-json files'
    else:
        labelme_files_relative = recursive_file_list(labelme_folder,return_relative_paths=True)
        labelme_files_relative = [fn for fn in labelme_files_relative if fn.endswith('.json')]

    if required_token is None:
        valid_labelme_files_relative = labelme_files_relative
    else:
        valid_labelme_files_relative = []

        # fn_relative = labelme_files_relative[-1]
        for fn_relative in labelme_files_relative:

            fn_abs = os.path.join(labelme_folder,fn_relative)

            with open(fn_abs,'r') as f:
                labelme_data = json.load(f)
                if required_token not in labelme_data:
                    continue

            valid_labelme_files_relative.append(fn_relative)

        print('{} of {} files are valid'.format(len(valid_labelme_files_relative),
                                                len(labelme_files_relative)))

    del labelme_files_relative

    if category_name_to_category_id is None:

        category_name_to_category_id = {}

        for fn_relative in valid_labelme_files_relative:

            fn_abs = os.path.join(labelme_folder,fn_relative)
            with open(fn_abs,'r') as f:
                labelme_data = json.load(f)
                for shape in labelme_data['shapes']:
                    label = shape['label']
                    if label not in category_name_to_category_id:
                        category_name_to_category_id[label] = len(category_name_to_category_id)
        # ...for each file

    # ...if we need to build a category mapping

    image_results = []

    n_workers = min(n_workers,len(valid_labelme_files_relative))

    if n_workers <= 1:
        for fn_relative in tqdm(valid_labelme_files_relative):

            fn_abs = os.path.join(labelme_folder,fn_relative)
            image_result = labelme_file_to_yolo_file(fn_abs,
                                      category_name_to_category_id,
                                      yolo_file=None,
                                      required_token=required_token,
                                      overwrite_behavior=overwrite_behavior)
            image_results.append(image_result)
        # ...for each file
    else:
        pool = None
        try:
            if use_threads:
                pool = ThreadPool(n_workers)
            else:
                pool = Pool(n_workers)

            valid_labelme_files_abs = [os.path.join(labelme_folder,fn_relative) for \
                                    fn_relative in valid_labelme_files_relative]

            image_results = list(tqdm(pool.imap(
                partial(labelme_file_to_yolo_file,
                        category_name_to_category_id=category_name_to_category_id,
                        yolo_file=None,
                        required_token=required_token,
                        overwrite_behavior=overwrite_behavior),
                        valid_labelme_files_abs),
                        total=len(valid_labelme_files_abs)))
        finally:
            pool.close()
            pool.join()
            print("Pool closed and joined for labelme conversion to YOLO")

    assert len(valid_labelme_files_relative) == len(image_results)

    print('Converted {} labelme .json files to YOLO'.format(
        len(valid_labelme_files_relative)))

    labelme_to_yolo_results = {}
    labelme_to_yolo_results['category_name_to_category_id'] = category_name_to_category_id
    labelme_to_yolo_results['image_results'] = image_results

    return labelme_to_yolo_results

# ...def labelme_folder_to_yolo(...)


#%% Interactive driver

if False:

    pass

    #%%

    labelme_file = os.path.expanduser('~/tmp/labels/x.json')
    required_token = 'saved_by_labelme'
    category_name_to_category_id = {'animal':0}
    labelme_folder = os.path.expanduser('~/tmp/labels')

    #%%

    category_name_to_category_id = \
        labelme_folder_to_yolo(labelme_folder,
                               category_name_to_category_id=category_name_to_category_id,
                               required_token=required_token,
                               overwrite_behavior='overwrite')

#%% Command-line driver

import sys
import argparse
import json # Already imported above, but good practice for clarity

def main():
    """
    Command-line interface to convert LabelMe JSON files to YOLO format.
    """
    parser = argparse.ArgumentParser(
        description='Convert a folder of LabelMe .json files to YOLO .txt format.'
    )
    parser.add_argument(
        'labelme_folder',
        type=str,
        help='Folder of LabelMe .json files to convert'
    )
    parser.add_argument(
        '--output_category_file',
        type=str,
        default=None,
        help='Path to save the generated category mapping (.json)'
    )
    parser.add_argument(
        '--required_token',
        type=str,
        default=None,
        help='Only process files containing this token in their LabelMe JSON data'
    )
    parser.add_argument(
        '--overwrite_behavior',
        type=str,
        default='overwrite',
        choices=['skip', 'overwrite'],
        help="Behavior if YOLO .txt files exist (default: 'overwrite')"
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=1,
        help='Number of workers for parallel processing (default: 1)'
    )
    parser.add_argument(
        '--use_threads',
        type=str,
        default='true',
        choices=['true', 'false'],
        help="Use threads ('true') or processes ('false') for parallelism (default: 'true')"
    )

    args = parser.parse_args()

    if args.use_threads.lower() == 'true':
        use_threads = True
    elif args.use_threads.lower() == 'false':
        use_threads = False
    else:
        # This case should ideally be caught by choices, but as a safeguard:
        print("Error: --use_threads must be 'true' or 'false'")
        sys.exit(1)

    results = labelme_folder_to_yolo(
        labelme_folder=args.labelme_folder,
        category_name_to_category_id=None,  # As per requirement
        required_token=args.required_token,
        overwrite_behavior=args.overwrite_behavior,
        relative_filenames_to_convert=None, # As per requirement
        n_workers=args.n_workers,
        use_threads=use_threads
    )

    if args.output_category_file:
        category_map = results['category_name_to_category_id']
        with open(args.output_category_file, 'w') as f:
            json.dump(category_map, f, indent=1)
        print(f"Saved category mapping to: {args.output_category_file}")

    print(f"Finished processing folder: {args.labelme_folder}")
    # Optionally, print more detailed results from results['image_results']
    # For example, count of converted vs. skipped files.
    converted_count = sum(1 for res in results['image_results'] if res['status'] == 'converted')
    skipped_exist_count = sum(1 for res in results['image_results'] if res['status'] == 'skip-exists')
    skipped_token_count = sum(1 for res in results['image_results'] if res['status'] == 'skip-no-required-token')
    print(f"  Converted files: {converted_count}")
    if skipped_exist_count > 0:
        print(f"  Skipped (already exist): {skipped_exist_count}")
    if skipped_token_count > 0:
        print(f"  Skipped (missing required token): {skipped_token_count}")


if __name__ == '__main__':
    main()
