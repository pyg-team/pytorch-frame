from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from dateutil.parser import ParserError

from torch_frame import stype
from torch_frame.data.mapper import MultiCategoricalTensorMapper
from torch_frame.typing import DataFrame, Series

POSSIBLE_SEPS = ["|", ","]
POSSIBLE_TIME_FORMATS = [None, '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']


def _is_timestamp(ser: Series) -> bool:
    is_timestamp = False
    for time_format in POSSIBLE_TIME_FORMATS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(ser, format=time_format)
            is_timestamp = True
        except (ValueError, ParserError, TypeError):
            pass
    return is_timestamp


def _lst_is_all_type(
    lst: list[Any],
    types: tuple[type, ...] | type,
) -> bool:
    assert isinstance(lst, list)
    return all(isinstance(x, types) for x in lst)


def _lst_is_free_of_nan_and_inf(lst: list[Any]):
    assert isinstance(lst, list)
    return all(not math.isnan(x) and not math.isinf(x) for x in lst)


def _min_count(ser: Series) -> int:
    return ser.value_counts().min()


def infer_series_stype(ser: Series) -> stype | None:
    """Infer :obj:`stype` given :class:`Series` object. The inference may not
    be always correct/best for your data. We recommend you double-checking the
    correctness yourself before actually using it.

    Args:
        ser (Series): Input series.

    Returns:
        Optional[stype]: Inferred :obj:`stype`. Returns :obj:`None` if
            inference failed.
    """
    has_nan = ser.isna().any()
    if has_nan:
        ser = ser.dropna()

    if len(ser) == 0:
        return None

    # Categorical minimum counting threshold. If the count of the most minor
    # categories is larger than this value, we treat the column as categorical.
    cat_min_count_thresh = 4

    if isinstance(ser.iloc[0], list):
        # Candidates: embedding, sequence_numerical, multicategorical

        # True if all elements in all lists are numerical
        is_all_numerical = True
        # True if all elements in all lists are string
        is_all_string = True
        # True if all lists are of the same length and all elements are float
        # and free of nans.
        is_embedding = True

        length = len(ser.iloc[0])
        for lst in ser:
            if not isinstance(lst, list):
                return None
            if _lst_is_all_type(lst, (int, float)):
                if not (length == len(lst) and _lst_is_all_type(lst, float)
                        and _lst_is_free_of_nan_and_inf(lst)):
                    is_embedding = False
            else:
                is_all_numerical = False
            if not _lst_is_all_type(lst, str):
                is_all_string = False

        if is_all_numerical:
            if is_embedding:
                return stype.embedding
            else:
                return stype.sequence_numerical
        elif is_all_string:
            return stype.multicategorical
        else:
            return None
    else:
        # Candidates: numerical, categorical, multicategorical, and
        # text_(embedded/tokenized)

        if ptypes.is_numeric_dtype(ser):

            if ptypes.is_bool_dtype(ser):
                return stype.categorical
            # Candidates: numerical, categorical
            if ptypes.is_float_dtype(ser) and not (has_nan and
                                                   (ser % 1 == 0).all()):
                return stype.numerical
            else:
                if _min_count(ser) > cat_min_count_thresh:
                    return stype.categorical
                else:
                    return stype.numerical
        else:
            # Candidates: timestamp, categorical, multicategorical,
            # text_(embedded/tokenized), embedding
            if _is_timestamp(ser):
                return stype.timestamp

            # Candates: categorical, multicategorical,
            # text_(embedded/tokenized), embedding
            if _min_count(ser) > cat_min_count_thresh or ptypes.is_bool_dtype(
                    ser):
                return stype.categorical

            # Candates: multicategorical, text_(embedded/tokenized), embedding
            if not ptypes.is_string_dtype(ser):
                if _min_count(ser) > cat_min_count_thresh:
                    return stype.multicategorical
                else:
                    return stype.embedding

            # Try different possible seps and mick the largest min_count.
            if isinstance(ser.iloc[0], list) or isinstance(
                    ser.iloc[0], np.ndarray):
                max_min_count = _min_count(ser.explode())
            else:
                min_count_list = []
                for sep in POSSIBLE_SEPS:
                    try:
                        min_count_list.append(
                            _min_count(
                                ser.apply(
                                    lambda row: MultiCategoricalTensorMapper.
                                    split_by_sep(row, sep)).explode()))
                    except Exception as e:
                        logging.warn(
                            "Mapping series into multicategorical stype "
                            f"with separator {sep} raised an exception {e}")
                        continue
                max_min_count = max(min_count_list or [0])

            if max_min_count > cat_min_count_thresh:
                return stype.multicategorical
            else:
                return stype.text_embedded


def infer_df_stype(df: DataFrame) -> dict[str, stype]:
    """Infer :obj:`col_to_stype` given :class:`DataFrame` object.

    Args:
        df (DataFrame): Input data frame.

    Returns:
        col_to_stype: Inferred :obj:`col_to_stype`, mapping a column name to
            its inferred :obj:`stype`.
    """
    col_to_stype = {}
    for col in df.columns:
        stype = infer_series_stype(df[col])
        if stype is not None:
            col_to_stype[col] = stype
    return col_to_stype
