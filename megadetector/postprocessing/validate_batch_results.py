"""

validate_batch_results.py

Given a .json file containing MD results, validate that it's compliant with the format spec:

https://lila.science/megadetector-output-format

"""

#%% Constants and imports

import os
import sys
import json
import argparse

from megadetector.detection.video_utils import is_video_file
from megadetector.utils.ct_utils import args_to_object

typical_info_fields = ['detector','detection_completion_time',
                       'classifier','classification_completion_time',
                       'detection_metadata','classifier_metadata']
required_keys = ['info','images','detection_categories']
typical_keys = ['classification_categories']


#%% Classes

class ValidateBatchResultsOptions:
    """
    Options controlling the behavior of validate_bach_results()
    """
    
    def __init__(self):
        
        #: Should we verify that images exist?  If this is True, and the .json
        #: file contains relative paths, relative_path_base needs to be specified.
        self.check_image_existence = False
        
        #: If check_image_existence is True, where do the images live?
        #:
        #: If None, assumes absolute paths.
        self.relative_path_base = None
    
# ...class ValidateBatchResultsOptions



#%% Main function

def validate_batch_results(json_filename,options=None):
    """
    Verify that [json_filename] is a valid MD output file.  Currently errors on invalid files.
    
    Args: 
        json_filename (str): the filename to validate
        options (ValidateBatchResultsOptions, optionsl): all the parameters used to control this 
            process, see ValidateBatchResultsOptions for details
            
    Returns:
        bool: reserved; currently always errors or returns True.
    """
    
    if options is None:
        options = ValidateBatchResultsOptions()
    
    with open(json_filename,'r') as f:
        d = json.load(f)
    
    ## Info validation
    
    assert 'info' in d
    info = d['info']
    
    assert isinstance(info,dict)
    assert 'format_version' in info 
    format_version = float(info['format_version'])
    assert format_version >= 1.3, 'This validator can only be used with format version 1.3 or later'
            
    print('Validating a .json results file with format version {}'.format(format_version))
    
    ## Category validation
    
    assert 'detection_categories' in d
    for k in d['detection_categories'].keys():
        # Categories should be string-formatted ints
        assert isinstance(k,str)
        _ = int(k)
        assert isinstance(d['detection_categories'][k],str)
        
    if 'classification_categories' in d:
        for k in d['classification_categories'].keys():
            # Categories should be string-formatted ints
            assert isinstance(k,str)
            _ = int(k)
            assert isinstance(d['classification_categories'][k],str)
    
    
    ## Image validation
    
    assert 'images' in d
    assert isinstance(d['images'],list)
    
    # im = d['images'][0]
    for im in d['images']:
        
        assert isinstance(im,dict)
        assert 'file' in im
        
        file = im['file']
        
        if options.check_image_existence:
            if options.relative_path_base is None:
                file_abs = file
            else:
                file_abs = os.path.join(options.relative_path_base,file)
            assert os.path.isfile(file_abs), 'Cannot find file {}'.format(file_abs)
            
        if 'detections' not in im or im['detections'] is None:
            assert 'failure' in im and isinstance(im['failure'],str)
        else:
            assert isinstance(im['detections'],list)
            
        if is_video_file(im['file']) and (format_version >= 1.4):
            assert 'frame_rate' in im
            if 'detections' in im and im['detections'] is not None:
                for det in im['detections']:
                    assert 'frame_number' in det
            
    # ...for each image
        
    
    ## Checking on other keys
    
    for k in d.keys():
        if k not in typical_keys and k not in required_keys:
            print('Warning: non-standard key {} present at file level'.format(k))
                  
# ...def validate_batch_results(...)


#%% Interactive driver(s)

if False:

    #%%
    
    options = ValidateBatchResultsOptions()
    # json_filename = r'g:\temp\format.json'
    # json_filename = r'g:\temp\test-videos\video_results.json'
    json_filename = r'g:\temp\test-videos\image_results.json'
    options.check_image_existence = True
    options.relative_path_base = r'g:\temp\test-videos'
    validate_batch_results(json_filename,options)
    

#%% Command-line driver

def main():
    
    options = ValidateBatchResultsOptions()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'json_filename',
        help='path to .json file containing MegaDetector results')
    parser.add_argument(
        '--check_image_existence', action='store_true',
        help='check that all images referred to in the results file exist')
    parser.add_argument(
        '--relative_path_base', default=None,
        help='if --check_image_existence is specified and paths are relative, use this as the base folder')
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    
    args_to_object(args, options)    

    validate_batch_results(args.json_filename,options)
    
    
if __name__ == '__main__':
    main()
