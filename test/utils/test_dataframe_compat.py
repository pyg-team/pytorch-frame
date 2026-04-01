"""Tests for the DataFrame compatibility layer (pandas path).

cuDF-specific tests live in test/data/test_cudf_integration.py and are
skipped when cuDF is not installed.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from torch_frame.utils.dataframe_compat import (
    array_to_numpy,
    df_concat,
    df_merge,
    flatten_series_values,
    is_bool_dtype,
    is_cudf_available,
    is_cudf_object,
    is_float_dtype,
    is_numeric_dtype,
    is_string_dtype,
    make_series,
    series_to_tensor,
    to_datetime,
)


# ---------- backend detection ----------


def test_is_cudf_object_pandas():
    ser = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": [1]})
    assert not is_cudf_object(ser)
    assert not is_cudf_object(df)
    assert not is_cudf_object(42)
    assert not is_cudf_object("hello")


# ---------- dtype helpers ----------


def test_is_numeric_dtype():
    assert is_numeric_dtype(pd.Series([1, 2, 3]))
    assert is_numeric_dtype(pd.Series([1.0, 2.0]))
    assert not is_numeric_dtype(pd.Series(["a", "b"]))


def test_is_bool_dtype():
    assert is_bool_dtype(pd.Series([True, False]))
    assert not is_bool_dtype(pd.Series([1, 2]))


def test_is_float_dtype():
    assert is_float_dtype(pd.Series([1.0, 2.0]))
    assert not is_float_dtype(pd.Series([1, 2]))


def test_is_string_dtype():
    assert is_string_dtype(pd.Series(["a", "b"]))
    assert not is_string_dtype(pd.Series([1, 2]))


# ---------- conversion helpers ----------


def test_to_datetime():
    ser = pd.Series(["2023-01-01", "2023-06-15"])
    result = to_datetime(ser, format="%Y-%m-%d")
    assert result.dtype == "datetime64[ns]"
    assert result.iloc[0].year == 2023


def test_series_to_tensor_float():
    ser = pd.Series([1.0, 2.0, 3.0])
    tensor = series_to_tensor(ser, dtype=np.dtype("float32"))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_series_to_tensor_int():
    ser = pd.Series([1, 2, 3])
    tensor = series_to_tensor(ser)
    assert isinstance(tensor, torch.Tensor)
    assert torch.equal(tensor, torch.tensor([1, 2, 3]))


def test_series_to_tensor_no_dtype():
    ser = pd.Series([1.5, 2.5])
    tensor = series_to_tensor(ser)
    assert tensor.dtype == torch.float64


def test_series_to_tensor_device():
    ser = pd.Series([1.0, 2.0])
    tensor = series_to_tensor(ser, dtype=np.dtype("float32"), device="cpu")
    assert tensor.device == torch.device("cpu")


def test_array_to_numpy_passthrough():
    arr = np.array([1, 2, 3])
    result = array_to_numpy(arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, arr)


def test_flatten_series_values_scalar():
    ser = pd.Series([1.0, 2.0, 3.0])
    result = flatten_series_values(ser)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_flatten_series_values_lists():
    ser = pd.Series([[1.0, 2.0], [3.0, 4.0, 5.0]])
    result = flatten_series_values(ser)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])


# ---------- merge / concat helpers ----------


def test_df_merge():
    left = pd.DataFrame({"key": ["a", "b", "c"], "val": [1, 2, 3]})
    right = pd.DataFrame({"key": ["a", "c"], "extra": [10, 30]})
    result = df_merge(left, right, on="key", how="inner")
    assert len(result) == 2
    assert set(result["key"]) == {"a", "c"}


def test_df_merge_series_index():
    # Mirrors CategoricalTensorMapper merge pattern
    categories = pd.Series(
        index=["cat_a", "cat_b", "cat_c"],
        data=pd.RangeIndex(0, 3),
        name="index",
    )
    ser = pd.Series(["cat_b", "cat_a", "cat_c", "cat_a"], name="data")
    result = df_merge(
        ser,
        categories,
        how="left",
        left_on="data",
        right_index=True,
    )
    assert list(result["index"]) == [1, 0, 2, 0]


def test_df_concat():
    s1 = pd.Series([0])
    s2 = pd.Series([1, 2, 3])
    result = df_concat([s1, s2])
    assert len(result) == 4


def test_make_series():
    ser = make_series(data=[1, 2, 3], name="test")
    assert isinstance(ser, pd.Series)
    assert ser.name == "test"
    assert list(ser) == [1, 2, 3]


def test_make_series_with_index():
    ser = make_series(
        data=[10, 20],
        index=["a", "b"],
        like=pd.Series(),
    )
    assert isinstance(ser, pd.Series)
    assert ser["a"] == 10
