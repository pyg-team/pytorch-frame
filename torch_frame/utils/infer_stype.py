import math
from typing import Any, Dict, List, Optional

import pandas as pd
import pandas.api.types as ptypes
from dateutil.parser import ParserError

from torch_frame import stype
from torch_frame.typing import DataFrame, Series

POSSIBLE_SEPS = ["|", ","]
POSSIBLE_TIME_FORMATS = [None, '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m-%d']


def _is_timestamp(ser: Series) -> bool:
    is_timestamp = False
    for time_format in POSSIBLE_TIME_FORMATS:
        try:
            pd.to_datetime(ser, format=time_format)
            is_timestamp = True
        except (ValueError, ParserError):
            pass
    return is_timestamp


def _is_multicategorical(ser: Series) -> bool:
    ','.join(ser)
    import pdb
    pdb.set_trace()
    return True


def _lst_is_all_numeric(lst: List[Any]) -> bool:
    assert isinstance(lst, list)
    return all(isinstance(x, (int, float)) for x in lst)


def _lst_is_free_of_nan_and_inf(lst: List[Any]):
    assert isinstance(lst, list)
    return all(not math.isnan(x) and not math.isinf(x) for x in lst)


def infer_series_stype(ser: Series) -> Optional[stype]:
    """Infer :obj:`stype` given :class:`Series` object. The inference may not
    be always correct/best for your data. We recommend you double-checking the
    correctness yourself before actually using it.

    Args:
        ser (Series): Input series.

    Returns:
        Optional[stype]: Inferred :obj:`stype`. Returns :obj:`None` if
            inference failed.
    """
    ser = ser.dropna()

    if len(ser) == 0:
        # Cannot infer stype if ser is completely empty
        return None

    if isinstance(ser[0], list):
        # Candidates: embedding, sequence_categorical
        length = len(ser[0])
        is_embedding = True
        for lst in ser:
            if not isinstance(lst, list):
                return None
            if not _lst_is_all_numeric(lst):
                return None
            # Check if lst is qualified for embedding or not.
            if not (length == len(lst) and _lst_is_free_of_nan_and_inf(lst)):
                is_embedding = False
        if is_embedding:
            return stype.embedding
        else:
            return stype.sequence_numerical
    else:
        # Candidates: numerical, categorical, multicategorical,
        # text_(embedded/tokenized)
        if ptypes.is_numeric_dtype(ser):
            # Candidates: numerical, categorical
            if ptypes.is_float_dtype(ser):
                return stype.numerical
            elif ptypes.is_integer_dtype(ser):
                if ser.nunique() < 10:
                    # Heuristics: If the number of unique values is less than
                    # 10, then we infer the column as categorical
                    return stype.categorical
                else:
                    return stype.numerical
            else:
                return None
        else:
            # Candidates: categorical, multicategorical,
            # text_(embedded/tokenized)
            if _is_timestamp(ser):
                return stype.timestamp
            elif _is_multicategorical(ser):
                # TODO: Add more logic
                return stype.multicategorical


def infer_df_stype(df: DataFrame) -> Dict[str, stype]:
    """Infer :obj:`col_to_stype` given :class:`DataFrame` object.

    Args:
        df (DataFrame): Input data frame.

    Returns:
        col_to_stype: Inferred :obj:`col_to_stype`, mapping a column name to
            its inferred :obj:`stype`.
    """
    col_to_stype = {}
    for col in df.columns:
        col_to_stype[col] = infer_series_stype(df[col])
    return col_to_stype
