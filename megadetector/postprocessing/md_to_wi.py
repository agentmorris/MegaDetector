"""

md_to_wi.py

Converts the MD .json format to the WI predictions.json format.

"""

#%% Imports and constants

import sys
import argparse
from megadetector.utils.wi_taxonomy_utils import generate_predictions_json_from_md_results


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument('md_results_file', action='store', type=str,
                        help='output file in MD format to convert')
    parser.add_argument('predictions_json_file', action='store', type=str,
                        help='.json file to write in predictions.json format')
    parser.add_argument('--base_folder', action='store', type=str, default=None,
                        help='folder name to prepend to each path in md_results_file, ' + \
                             'to convert relative paths to absolute paths.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    generate_predictions_json_from_md_results(args.md_results_file,
                                              args.predictions_json_file,
                                              base_folder=None)

if __name__ == '__main__':
    main()
