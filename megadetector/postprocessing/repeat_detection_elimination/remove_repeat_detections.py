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

from megadetector.postprocessing.repeat_detection_elimination import repeat_detections_core


#%% Main function

def remove_repeat_detections(inputFile,outputFile,filteringDir):
    """
    Given an index file that was produced in a first pass through find_repeat_detections,
    and a folder of images (from which the user has deleted images they don't want removed),
    remove the identified repeat detections from a set of MD results and write to a new file.
    
    Args:
        inputFile (str): .json file of MD results, from which we should remove repeat detections
        outputFile (str): output .json file to which we should write MD results (with repeat 
            detections removed)
        filteringDir (str): the folder produced by find_repeat_detections, containing a 
            detectionIndex.json file        
    """
    
    assert os.path.isfile(inputFile), "Can't find file {}".format(inputFile)
    assert os.path.isdir(filteringDir), "Can't find folder {}".format(filteringDir)
    options = repeat_detections_core.RepeatDetectionOptions()
    if os.path.isfile(filteringDir):
        options.filterFileToLoad = filteringDir
    else:
        assert os.path.isdir(filteringDir), '{} is not a valid folder'.format(filteringDir)
        options.filterFileToLoad = \
            os.path.join(filteringDir,repeat_detections_core.detection_index_file_name_base)
    repeat_detections_core.find_repeat_detections(inputFile, outputFile, options)


#%% Interactive driver

if False:
    
    #%%
    
    inputFile = r''
    outputFile = r''
    filteringDir = r''
    remove_repeat_detections(inputFile,outputFile,filteringDir)


#%% Command-line driver

import sys

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', help='.json file containing the original, unfiltered API results')
    parser.add_argument('outputFile', help='.json file to which you want to write the final, ' + \
                        'filtered API results')
    parser.add_argument('filteringDir', help='directory where you looked at lots of images and ' + \
                        'decided which ones were really false positives')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    remove_repeat_detections(args.inputFile, args.outputFile, args.filteringDir)

if __name__ == '__main__':
    main()
