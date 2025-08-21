"""

speciesnet_to_md.py

Converts the WI (SpeciesNet) predictions.json format to MD .json format.  This is just a
command-line wrapper around utils.wi_taxonomy_utils.generate_md_results_from_predictions_json.

"""

#%% Imports and constants

import sys
import argparse
from megadetector.utils.wi_taxonomy_utils import generate_md_results_from_predictions_json


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_json_file', action='store', type=str,
                        help='.json file to convert from SpeciesNet predictions.json format to MD format')
    parser.add_argument('md_results_file', action='store', type=str,
                        help='output file to write in MD format')
    parser.add_argument('--base_folder', action='store', type=str, default=None,
                        help='leading string to remove from each path in the predictions.json ' + \
                            'file (to convert from absolute to relative paths)')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    generate_md_results_from_predictions_json(args.predictions_json_file,
                                              args.md_results_file,
                                              args.base_folder)

if __name__ == '__main__':
    main()
