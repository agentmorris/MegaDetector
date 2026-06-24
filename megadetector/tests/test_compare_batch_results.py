"""

Tests for compare_batch_results threshold handling.

"""

#%% Imports

import pytest

from megadetector.postprocessing.compare_batch_results import (
    BatchComparisonOptions,
    PairwiseBatchComparisonOptions,
    _normalize_detection_thresholds,
    _pairwise_compare_batch_results,
)


#%% Tests

def test_normalize_detection_thresholds_float():
    """
    A single floating-point threshold should apply to all categories.
    """

    assert _normalize_detection_thresholds(0.2) == {'default': 0.2}


def test_normalize_detection_thresholds_int():
    """
    A single integer threshold should apply to all categories as a float.
    """

    assert _normalize_detection_thresholds(1) == {'default': 1.0}


def test_normalize_detection_thresholds_dict_preserved():
    """
    Existing category-to-threshold dictionaries should be preserved unchanged.
    """

    thresholds = {'animal': 0.2, 'person': 0.4, 'default': 0.1}

    assert _normalize_detection_thresholds(thresholds) is thresholds


def test_normalize_detection_thresholds_invalid_type():
    """
    Non-numeric, non-dictionary thresholds should fail clearly.
    """

    with pytest.raises(TypeError, match='dict or a numeric value'):
        _normalize_detection_thresholds('0.2')


def test_pairwise_comparison_normalizes_numeric_thresholds_before_validation():
    """
    Direct pairwise options with numeric thresholds should not fail at .values().
    """

    options = BatchComparisonOptions()
    options.pairwise_options = None
    options.image_folder = 'unused'

    pairwise_options = PairwiseBatchComparisonOptions()
    pairwise_options.results_filename_a = 'missing-a.json'
    pairwise_options.results_filename_b = 'missing-b.json'
    pairwise_options.detection_thresholds_a = 0.15
    pairwise_options.detection_thresholds_b = 1

    with pytest.raises(AssertionError, match="Can't find results file"):
        _pairwise_compare_batch_results(options, 0, pairwise_options)

    assert pairwise_options.detection_thresholds_a == {'default': 0.15}
    assert pairwise_options.detection_thresholds_b == {'default': 1.0}
