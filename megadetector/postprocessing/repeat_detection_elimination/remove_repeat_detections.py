"""

remove_repeat_detections.py

Used after running find_repeat_detections, then manually filtering the results,
to create a final filtered output file.

If you want to use this script, we recommend that you read the RDE user's guide:

https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination

"""

#%% Constants and imports

import argparse
import os
import sys

from megadetector.postprocessing.repeat_detection_elimination import repeat_detections_core


#%% Main function

def remove_repeat_detections(input_file,output_file,filtering_dir):
    """
    Given an index file that was produced in a first pass through find_repeat_detections,
    and a folder of images (from which the user has deleted images they don't want removed),
    remove the identified repeat detections from a set of MD results and write to a new file.

    Args:
        input_file (str): .json file of MD results, from which we should remove repeat detections
        output_file (str): output .json file to which we should write MD results (with repeat
            detections removed)
        filtering_dir (str): the folder produced by find_repeat_detections, containing a
            detectionIndex.json file
    """

    assert os.path.isfile(input_file), "Can't find file {}".format(input_file)
    assert os.path.isdir(filtering_dir), "Can't find folder {}".format(filtering_dir)
    options = repeat_detections_core.RepeatDetectionOptions()
    if os.path.isfile(filtering_dir):
        options.filterFileToLoad = filtering_dir
    else:
        assert os.path.isdir(filtering_dir), '{} is not a valid folder'.format(filtering_dir)
        options.filterFileToLoad = \
            os.path.join(filtering_dir,repeat_detections_core.detection_index_file_name_base)
    repeat_detections_core.find_repeat_detections(input_file, output_file, options)


#%% Interactive driver

if False:

    #%%

    input_file = r''
    output_file = r''
    filtering_dir = r''
    remove_repeat_detections(input_file,output_file,filtering_dir)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='.json file containing the original, unfiltered API results')
    parser.add_argument('output_file', help='.json file to which you want to write the final, ' + \
                        'filtered API results')
    parser.add_argument('filtering_dir', help='directory where you looked at lots of images and ' + \
                        'decided which ones were really false positives')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    remove_repeat_detections(args.input_file, args.output_file, args.filtering_dir)

if __name__ == '__main__':
    main()
