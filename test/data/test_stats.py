import numpy as np
import pandas as pd

from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.stype import (
    categorical,
    multicategorical,
    numerical,
    sequence_numerical,
)


def test_compute_col_stats_all_numerical():
    ser = pd.Series([1, 2, 3])
    stype = numerical
    assert compute_col_stats(ser, stype) == {
        StatType.MEAN: 2.0,
        StatType.STD: 0.816496580927726,
        StatType.QUANTILES: [1.0, 1.5, 2.0, 2.5, 3.0],
    }


def test_compute_col_stats_all_categorical():
    ser = pd.Series(['a', 'a', 'a', 'b', 'c'])
    stype = categorical
    assert compute_col_stats(ser, stype) == {
        StatType.COUNT: (['a', 'b', 'c'], [3, 1, 1]),
    }


def test_compute_col_stats_all_multi_categorical():
    ser = pd.Series(['a|b', 'a|c', 'c|a', 'a|b|c', np.nan])
    stype = multicategorical
    assert compute_col_stats(ser, stype, sep='|') == {
        StatType.MULTI_COUNT: (['a', 'c', 'b'], [4, 3, 2]),
    }


def test_compute_col_stats_all_sequence_numerical():
    ser = pd.Series([[1, 2, 3], [4, 5, 6]])
    stype = sequence_numerical
    assert compute_col_stats(ser, stype) == {
        StatType.MEAN: 3.5,
        StatType.STD: 1.707825127659933,
        StatType.QUANTILES: [1.0, 2.25, 3.5, 4.75, 6.0],
    }


def test_compute_col_stats_all_sequence_numerical_with_nan_inf():
    ser = pd.Series([[1, 2, 3, np.nan], [4, 5, 6]])
    expected_col_stats = {
        StatType.MEAN: 3.5,
        StatType.STD: 1.707825127659933,
        StatType.QUANTILES: [1.0, 2.25, 3.5, 4.75, 6.0],
    }
    stype = sequence_numerical
    assert compute_col_stats(ser, stype) == expected_col_stats
    ser = pd.Series([[1, 2, 3, np.inf], [4, 5, 6]])
    assert compute_col_stats(ser, stype) == expected_col_stats


def test_compute_col_stats_all_sequence_numerical_with_all_nan():
    ser = pd.Series([[np.nan, np.nan, np.nan, np.inf],
                     [np.nan, np.nan, np.nan]])
    stype = sequence_numerical
    assert compute_col_stats(ser, stype) == {
        StatType.MEAN: np.nan,
        StatType.STD: np.nan,
        StatType.QUANTILES: [np.nan, np.nan, np.nan, np.nan, np.nan],
    }


def test_compute_col_stats_numerical_with_inf():
    ser = pd.Series([1, 2, 3, np.inf, -np.inf])
    stype = numerical
    assert compute_col_stats(ser, stype) == {
        StatType.MEAN: 2.0,
        StatType.STD: 0.816496580927726,
        StatType.QUANTILES: [1.0, 1.5, 2.0, 2.5, 3.0],
    }


def test_compute_col_stats_numerical_with_nan():
    ser = pd.Series([1, 2, 3, np.nan])
    stype = numerical
    assert compute_col_stats(ser, stype) == {
        StatType.MEAN: 2.0,
        StatType.STD: 0.816496580927726,
        StatType.QUANTILES: [1.0, 1.5, 2.0, 2.5, 3.0],
    }


def test_compute_col_stats_numerical_all_nan():
    ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
    stype = numerical
    assert compute_col_stats(ser, stype) == {
        StatType.MEAN: np.nan,
        StatType.STD: np.nan,
        StatType.QUANTILES: [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
