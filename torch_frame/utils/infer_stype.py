from typing import Dict

import pandas.api.types as ptypes

from torch_frame import stype
from torch_frame.typing import DataFrame, Series


def infer_series_stype(ser: Series) -> stype:
    """Infer :obj:`stype` given :class:`Series` object.

    Args:
        ser (Series): Input series.

    Returns:
        stype: Inferred :obj:`stype`.
    """
    ser = ser.dropna()
    if ptypes.is_float_dtype(ser):
        return stype.numerical
    else:
        # TODO: Add more logic
        return stype.categorical


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
