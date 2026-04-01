r"""Compatibility layer for pandas and cuDF DataFrames.

This module provides helper functions that dispatch to the correct
backend (pandas or cuDF) based on the type of the input object.
When cuDF is not installed, all helpers fall back to pandas.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def is_cudf_available() -> bool:
    r"""Returns :obj:`True` if cuDF is installed."""
    try:
        import cudf  # noqa: F401
        return True
    except ImportError:
        return False


def is_cudf_object(obj: Any) -> bool:
    r"""Returns :obj:`True` if *obj* is a cuDF DataFrame or Series.

    Uses module-name inspection so that cuDF is never imported when it
    is not installed.
    """
    mod = type(obj).__module__ or ""
    return mod.startswith("cudf.")


def _get_df_lib(obj: Any):
    r"""Returns the :mod:`cudf` or :mod:`pandas` module matching *obj*."""
    if is_cudf_object(obj):
        import cudf
        return cudf
    return pd


# --------------- dtype helpers ---------------


def is_numeric_dtype(ser_or_dtype: Any) -> bool:
    r"""Backend-agnostic wrapper for
    :func:`pandas.api.types.is_numeric_dtype`."""
    if is_cudf_object(ser_or_dtype):
        import cudf.api.types as ctypes
        return ctypes.is_numeric_dtype(ser_or_dtype)
    import pandas.api.types as ptypes
    return ptypes.is_numeric_dtype(ser_or_dtype)


def is_bool_dtype(ser_or_dtype: Any) -> bool:
    r"""Backend-agnostic wrapper for
    :func:`pandas.api.types.is_bool_dtype`."""
    if is_cudf_object(ser_or_dtype):
        import cudf.api.types as ctypes
        return ctypes.is_bool_dtype(ser_or_dtype)
    import pandas.api.types as ptypes
    return ptypes.is_bool_dtype(ser_or_dtype)


def is_float_dtype(ser_or_dtype: Any) -> bool:
    r"""Backend-agnostic wrapper for
    :func:`pandas.api.types.is_float_dtype`."""
    if is_cudf_object(ser_or_dtype):
        import cudf.api.types as ctypes
        return ctypes.is_float_dtype(ser_or_dtype)
    import pandas.api.types as ptypes
    return ptypes.is_float_dtype(ser_or_dtype)


def is_string_dtype(ser_or_dtype: Any) -> bool:
    r"""Backend-agnostic wrapper for
    :func:`pandas.api.types.is_string_dtype`."""
    if is_cudf_object(ser_or_dtype):
        import cudf.api.types as ctypes
        return ctypes.is_string_dtype(ser_or_dtype)
    import pandas.api.types as ptypes
    return ptypes.is_string_dtype(ser_or_dtype)


# --------------- conversion helpers ---------------


def to_datetime(
    ser: Any,
    format: str | None = None,
    errors: str = "coerce",
) -> Any:
    r"""Backend-agnostic wrapper for :func:`pandas.to_datetime`."""
    lib = _get_df_lib(ser)
    return lib.to_datetime(ser, format=format, errors=errors)


def series_to_tensor(
    ser: Any,
    dtype: np.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Converts a pandas or cuDF Series to a :class:`torch.Tensor`.

    For pandas, uses :func:`torch.from_numpy`.  For cuDF, uses DLPack
    via :func:`torch.as_tensor` on the underlying cupy array, avoiding
    a GPU-to-CPU round-trip.
    """
    if is_cudf_object(ser):
        values = ser.values
        if dtype is not None:
            values = values.astype(dtype)
        tensor = torch.as_tensor(values)
    else:
        values = ser.values
        if dtype is not None:
            values = values.astype(dtype)
        tensor = torch.from_numpy(values)
    return tensor.to(device) if device is not None else tensor


def array_to_numpy(values: Any) -> np.ndarray:
    r"""Converts a numpy or cupy array to a numpy array.

    If the input is already a numpy array, it is returned as-is.
    For cupy arrays, :meth:`cupy.ndarray.get` is used.
    """
    mod = type(values).__module__ or ""
    if mod.startswith("cupy"):
        return values.get()
    return np.asarray(values)


def flatten_series_values(ser: Any) -> np.ndarray:
    r"""Flattens a Series of scalars or lists into a 1D numpy array.

    Used by :class:`~torch_frame.data.stats.StatType` for computing
    ``MEAN``, ``STD``, and ``QUANTILES`` over numerical and
    sequence-numerical columns.
    """
    if is_cudf_object(ser):
        # cuDF list columns support .explode(); scalar columns pass
        # through.  Always bring to CPU for numpy stats.
        try:
            flat = ser.explode().dropna()
        except TypeError:
            flat = ser
        return array_to_numpy(flat.values)
    return np.hstack(np.hstack(ser.values))


# --------------- merge / concat helpers ---------------


def df_merge(
    left: Any,
    right: Any,
    **kwargs: Any,
) -> Any:
    r"""Backend-agnostic merge that converts the smaller side to match
    the larger side's backend."""
    lib = _get_df_lib(left)
    if is_cudf_object(left) and not is_cudf_object(right):
        import cudf
        if isinstance(right, pd.Series):
            right = cudf.Series(right)
        elif isinstance(right, pd.DataFrame):
            right = cudf.DataFrame(right)
    return lib.merge(left, right, **kwargs)


def df_concat(
    objs: list[Any],
    like: Any | None = None,
    **kwargs: Any,
) -> Any:
    r"""Backend-agnostic wrapper for :func:`pandas.concat`."""
    ref = like if like is not None else objs[0]
    lib = _get_df_lib(ref)
    if is_cudf_object(ref):
        import cudf
        converted = []
        for o in objs:
            if isinstance(o, (pd.Series, pd.DataFrame)):
                converted.append(
                    cudf.Series(o)
                    if isinstance(o, pd.Series) else cudf.DataFrame(o))
            else:
                converted.append(o)
        return lib.concat(converted, **kwargs)
    return pd.concat(objs, **kwargs)


def make_series(
    data: Any = None,
    index: Any = None,
    name: str | None = None,
    like: Any | None = None,
    **kwargs: Any,
) -> Any:
    r"""Creates a :class:`pandas.Series` or :class:`cudf.Series`
    depending on *like*."""
    if like is not None and is_cudf_object(like):
        import cudf
        return cudf.Series(data=data, index=index, name=name, **kwargs)
    return pd.Series(data=data, index=index, name=name, **kwargs)
