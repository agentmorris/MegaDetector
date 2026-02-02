"""

threshold_coco_dataset.py

Given a COCO-formatted dataset that stores confidence in the semi-standard "score"
field, remove annotations below a threshold.

"""

#%% Imports and constants

import json
import sys
import argparse

from tqdm import tqdm

from megadetector.utils.ct_utils import write_json


#%% Functions

def threshold_coco_dataset(input_filename,
                           confidence_threshold=0.0,
                           output_filename=None,
                           confidence_field='score',
                           missing_confidence_handling='error'):
    """
    Given a COCO-formatted dataset that stores confidence in the semi-standard "score"
    field, remove annotations below a threshold.


    Args:
        input_filename (str): the (input) COCO-formatted .json file containing annotations
        confidence_threshold (float, optional): discard annotations below this confidence value
        output_filename (str, optional): write the thresholded output here
        confidence_field (str, optional): the field within annotations that
            represents confidence values
        missing_confidence_handling: what to do if a confidence value is missing
            (should be 'error' or 'warning')

    Returns:
        dict: the thresholded COCO database
    """

    # Validate arguments

    assert missing_confidence_handling in ('error','warning'), \
        f'Illegal missing confidence handling {missing_confidence_handling}'

    # Read input data
    with open(input_filename,'r') as f:
        d = json.load(f)

    annotations_to_keep = []
    annotations = d['annotations']

    for ann in tqdm(annotations):

        if confidence_field not in ann:
            ann_id = 'unknown'
            if 'id' in ann:
                ann_id = ann['id']
            s = 'annotation {} is missing field {}'.format(ann_id,confidence_field)
            if missing_confidence_handling == 'error':
                raise ValueError(s)
            else:
                assert missing_confidence_handling == 'warning'
                print('Warning: {}'.format(s))
                continue

        confidence = ann[confidence_field]
        if confidence >= confidence_threshold:
            annotations_to_keep.append(ann)

    # ...for each annotation

    print('Keeping {} of {} annotations'.format(
        len(annotations_to_keep),len(d['annotations'])))
    d['annotations'] = annotations_to_keep

    if output_filename is not None:
        write_json(output_filename,d)

    return d

# ...def threshold_coco_dataset(...)


#%% Command-line driver

def main():
    """
    Command-line driver for threshold_coco_dataset
    """

    parser = argparse.ArgumentParser(
        description='Threshold a COCO dataset, write the results to a new file'
    )
    parser.add_argument(
        'input_filename',
        type=str,
        help='Path to the input COCO .json file'
    )
    parser.add_argument(
        'output_filename',
        type=str,
        help='Path to the .json file where thresholded data will be saved'
    )
    parser.add_argument(
        'confidence_threshold',
        type=float,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--confidence_field',
        type=str,
        default='score',
        help='Field to use for confidence values, default "score"'
    )
    parser.add_argument(
        '--missing_confidence_handling',
        type=str,
        default='error',
        choices=['error', 'warning'],
        help='Whether to error on annotations that are missing the confidence field'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    threshold_coco_dataset(
        input_filename=args.input_filename,
        confidence_threshold=args.confidence_threshold,
        output_filename=args.output_filename,
        confidence_field=args.confidence_field,
        missing_confidence_handling=args.missing_confidence_handling
    )


if __name__ == '__main__':
    main()
