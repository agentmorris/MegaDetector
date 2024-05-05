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

# TODO
