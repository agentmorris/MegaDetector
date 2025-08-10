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

from tqdm import tqdm

from megadetector.detection.video_utils import is_video_file
from megadetector.utils.ct_utils import args_to_object, is_list_sorted # noqa

typical_info_fields = ['detector',
                       'detection_completion_time',
                       'classifier',
                       'classification_completion_time',
                       'detection_metadata',
                       'classifier_metadata']

required_keys = ['info',
                 'images',
                 'detection_categories']

typical_keys = ['classification_categories',
                'classification_category_descriptions']


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

        #: Should we return the loaded data, or just the validation results?
        self.return_data = False

        #: Enable additional debug output
        self.verbose = False

        #: Should we raise errors immediately (vs. just catching and reporting)?
        self.raise_errors = False

# ...class ValidateBatchResultsOptions


#%% Main function

def validate_batch_results(json_filename,options=None):
    """
    Verify that [json_filename] is a valid MD output file.  Currently errors on invalid files.

    Args:
        json_filename (str): the filename to validate
        options (ValidateBatchResultsOptions, optional): all the parameters used to control this
            process, see ValidateBatchResultsOptions for details

    Returns:
        dict: a dict with a field called "validation_results", which is itself a dict.  The reason
        it's a dict inside a dict is that if return_data is True, the outer dict also contains all
        the loaded data.  The "validation_results" dict contains fields called "errors", "warnings",
        and "filename".  "errors" and "warnings" are lists of strings, although "errors" will never
        be longer than N=1, since validation fails at the first error.

    """

    if options is None:
        options = ValidateBatchResultsOptions()

    if options.verbose:
        print('Loading results from {}'.format(json_filename))

    with open(json_filename,'r') as f:
        d = json.load(f)

    validation_results = {}
    validation_results['filename'] = json_filename
    validation_results['warnings'] = []
    validation_results['errors'] = []

    if not isinstance(d,dict):

        validation_results['errors'].append('Input data is not a dict')
        to_return = {}
        to_return['validation_results'] = validation_results
        return to_return

    try:

        ## Info validation

        if 'info' not in d:
            raise ValueError('Input does not contain info field')

        info = d['info']

        if not isinstance(info,dict):
            raise ValueError('Input contains invalid info field')

        if 'format_version' not in info :
            raise ValueError('Input does not specify format version')

        format_version = float(info['format_version'])
        if format_version < 1.3:
            raise ValueError('This validator can only be used with format version 1.3 or later')


        ## Category validation

        if 'detection_categories' not in d:
            raise ValueError('Input does not contain detection_categories field')

        for k in d['detection_categories'].keys():
            # Category ID should be string-formatted ints
            if not isinstance(k,str):
                raise ValueError('Invalid detection category ID: {}'.format(k))
            _ = int(k)
            if not isinstance(d['detection_categories'][k],str):
                raise ValueError('Invalid detection category name: {}'.format(
                    d['detection_categories'][k]))

        if 'classification_categories' in d:
            for k in d['classification_categories'].keys():
                # Categories should be string-formatted ints
                if not isinstance(k,str):
                    raise ValueError('Invalid classification category ID: {}'.format(k))
                _ = int(k)
                if not isinstance(d['classification_categories'][k],str):
                    raise ValueError('Invalid classification category name: {}'.format(
                        d['classification_categories'][k]))


        ## Image validation

        if 'images' not in d:
            raise ValueError('images field not present')
        if not isinstance(d['images'],list):
            raise ValueError('Invalid images field')

        if options.verbose:
            print('Validating images')

        # im = d['images'][0]
        for i_im,im in tqdm(enumerate(d['images']),total=len(d['images']),disable=(not options.verbose)):

            if not isinstance(im,dict):
                raise ValueError('Invalid image at index {}'.format(i_im))
            if 'file' not in im:
                raise ValueError('Image without filename at index {}'.format(i_im))

            file = im['file']

            if 'detections' in im and im['detections'] is not None:

                for det in im['detections']:

                    assert 'category' in det, 'Image {} has a detection with no category'.format(file)
                    assert 'conf' in det, 'Image {} has a detection with no confidence'.format(file)
                    assert isinstance(det['conf'],float), \
                        'Image {} has an illegal confidence value'.format(file)
                    assert 'bbox' in det, 'Image {} has a detection with no box'.format(file)
                    assert det['category'] in d['detection_categories'], \
                        'Image {} has a detection with an unmapped category {}'.format(
                            file,det['category'])

                    if 'classifications' in det and det['classifications'] is not None:
                        for c in det['classifications']:
                            assert isinstance(c[0],str), \
                                'Image {} has an illegal classification category: {}'.format(file,c[0])
                            try:
                                _ = int(c[0])
                            except Exception:
                                raise ValueError('Image {} has an illegal classification category: {}'.format(
                                    file,c[0]))
                            assert isinstance(c[1],float) or isinstance(c[1], int)

                # ...for each detection

            # ...if this image has a detections field

            if options.check_image_existence:

                if options.relative_path_base is None:
                    file_abs = file
                else:
                    file_abs = os.path.join(options.relative_path_base,file)
                if not os.path.isfile(file_abs):
                    raise ValueError('Cannot find file {}'.format(file_abs))

            if 'failure' in im:
                if im['failure'] is not None:
                    if not isinstance(im['failure'],str):
                        raise ValueError('Image {} has an illegal [failure] value: {}'.format(
                            im['file'],str(im['failure'])))
                    if 'detections' not in im:
                        s = 'Image {} has a failure value, should also have a null detections array'.format(
                            im['file'])
                        validation_results['warnings'].append(s)
                    elif im['detections'] is not None:
                        raise ValueError('Image {} has a failure value but a non-null detections array'.format(
                            im['file']))
            else:
                if not isinstance(im['detections'],list):
                    raise ValueError('Invalid detections list for image {}'.format(im['file']))

            if is_video_file(im['file']) and (format_version >= 1.5):

                if 'frames_processed' not in im:
                    raise ValueError('Video without frames_processed field: {}'.format(im['file']))

            if is_video_file(im['file']) and (format_version >= 1.4):

                if 'frame_rate' not in im:
                    raise ValueError('Video without frame rate: {}'.format(im['file']))
                if im['frame_rate'] < 0:
                    if 'failure' not in im:
                        raise ValueError('Video with illegal frame rate {}: {}'.format(
                            str(im['frame_rate']),im['file']))
                if 'detections' in im and im['detections'] is not None:
                    for det in im['detections']:
                        if 'frame_number' not in det:
                            raise ValueError('Frame without frame number in video {}'.format(
                                im['file']))
                    frame_numbers = [det['frame_number'] for det in im['detections']] # noqa
                    # assert is_list_sorted(frame_numbers)

        # ...for each image


        ## Validation of other keys

        for k in d.keys():
            if (k not in typical_keys) and (k not in required_keys):
                validation_results['warnings'].append(
                    'Warning: non-standard key {} present at file level'.format(k))

    except Exception as e:

        if options.raise_errors:
            raise
        else:
            validation_results['errors'].append(str(e))

    # ...try/except

    if options.return_data:
        to_return = d
    else:
        to_return = {}

    to_return['validation_results'] = validation_results

    return to_return

# ...def validate_batch_results(...)


#%% Interactive driver(s)

if False:

    #%% Validate all .json files in the MD test suite

    from megadetector.utils.path_utils import recursive_file_list
    filenames = recursive_file_list(os.path.expanduser('~/AppData/Local/Temp/md-tests'))
    filenames = [fn for fn in filenames if fn.endswith('.json')]
    filenames = [fn for fn in filenames if 'detectionIndex' not in fn]

    options = ValidateBatchResultsOptions()
    options.check_image_existence = False
    options.relative_path_base = None # r'g:\temp\test-videos'

    for json_filename in filenames:
        results = validate_batch_results(json_filename,options)
        if len(results['validation_results']['warnings']) > 0:
            print('Warnings in file {}:'.format(json_filename))
            for s in results['validation_results']['warnings']:
                print(s)
            print('')
        assert len(results['validation_results']['errors']) == 0


#%% Command-line driver

def main(): # noqa

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
