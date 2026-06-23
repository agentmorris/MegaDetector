"""

Regression tests for postprocessing utility behavior.

"""

#%% Imports

import json

import pandas as pd

from megadetector.postprocessing.combine_batch_outputs import combine_batch_output_dictionaries
from megadetector.postprocessing.generate_csv_report import generate_csv_report
from megadetector.postprocessing.remap_detection_categories import remap_detection_categories


#%% Support functions

def _write_json(filename, data):
    """
    Writes test JSON data to [filename].
    """

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f)


#%% Tests

def test_generate_csv_report_accepts_descending_classifications(tmp_path):
    """
    Classification lists are required to be sorted in descending confidence order.
    """

    results = {
        'info': {
            'format_version': '1.6',
            'detector': 'test-detector'
        },
        'detection_categories': {
            '1': 'animal'
        },
        'classification_categories': {
            '10': 'deer',
            '11': 'elk'
        },
        'images': [
            {
                'file': 'camera/image001.jpg',
                'detections': [
                    {
                        'category': '1',
                        'conf': 0.9,
                        'bbox': [0.1, 0.2, 0.3, 0.4],
                        'classifications': [
                            ['10', 0.8],
                            ['11', 0.2]
                        ]
                    }
                ]
            }
        ]
    }

    input_file = tmp_path / 'results.json'
    output_file = tmp_path / 'report.csv'
    _write_json(input_file, results)

    generate_csv_report(
        md_results_file=str(input_file),
        output_file=str(output_file),
        detection_confidence_threshold=0.1,
        classification_confidence_threshold=0.1,
        verbose=False
    )

    df = pd.read_csv(output_file)
    assert len(df) == 1
    assert df.iloc[0]['classification_category'] == 'deer'
    assert df.iloc[0]['max_classification_confidence'] == 0.8


def test_remap_detection_categories_adds_unknown_category(tmp_path):
    """
    Unknown category handling should produce a valid output category map.
    """

    input_data = {
        'info': {
            'format_version': '1.6',
            'detector': 'test-detector'
        },
        'detection_categories': {
            '1': 'animal'
        },
        'images': [
            {
                'file': 'image001.jpg',
                'detections': [
                    {
                        'category': '7',
                        'conf': 0.9,
                        'bbox': [0.1, 0.2, 0.3, 0.4]
                    }
                ]
            }
        ]
    }

    input_file = tmp_path / 'input.json'
    output_file = tmp_path / 'output.json'
    _write_json(input_file, input_data)

    remap_detection_categories(
        input_file=str(input_file),
        output_file=str(output_file),
        target_category_map={'1': 'animal'},
        input_category_name_to_output_category_name={'animal': 'animal'},
        invalid_category_handling='unknown'
    )

    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    assert output_data['detection_categories'] == {
        '1': 'animal',
        '2': 'unknown'
    }
    assert output_data['images'][0]['detections'][0]['category'] == '2'


def test_combine_batch_outputs_replaces_previous_null_detection_failure():
    """
    A later successful duplicate should replace an earlier failed result.
    """

    failed = {
        'info': {
            'format_version': '1.6',
            'detector': 'test-detector'
        },
        'detection_categories': {
            '1': 'animal'
        },
        'images': [
            {
                'file': 'camera\\image001.jpg',
                'failure': 'image access failure',
                'detections': None
            }
        ]
    }
    successful = {
        'info': {
            'format_version': '1.6',
            'detector': 'test-detector'
        },
        'detection_categories': {
            '1': 'animal'
        },
        'images': [
            {
                'file': 'camera/image001.jpg',
                'detections': [
                    {
                        'category': '1',
                        'conf': 0.9,
                        'bbox': [0.1, 0.2, 0.3, 0.4]
                    }
                ]
            }
        ]
    }

    merged = combine_batch_output_dictionaries(
        [failed, successful],
        require_uniqueness=False
    )

    assert failed['images'][0]['file'] == 'camera\\image001.jpg'
    assert len(merged['images']) == 1
    assert merged['images'][0]['file'] == 'camera/image001.jpg'
    assert 'failure' not in merged['images'][0]
    assert merged['images'][0]['detections'][0]['conf'] == 0.9
