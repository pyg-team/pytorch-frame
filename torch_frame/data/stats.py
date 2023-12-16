from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import torch

import torch_frame
from torch_frame.data.mapper import (
    MultiCategoricalTensorMapper,
    TimestampTensorMapper,
)
from torch_frame.typing import Series


class StatType(Enum):
    r"""The different types for column statistics.

    Attributes:
        MEAN: The average value of a numerical column.
        STD: The standard deviation of a numerical column.
        QUANTILES: The minimum, first quartile, median, third quartile,
            and the maximum of a numerical column.
        COUNT: The count of each category in a categorical column.
        MULTI_COUNT: The count of each category in a multi-categorical
            column.
        YEAR_RANGE: The range of years in a timestamp column.
    """

    # Numerical:
    MEAN = "MEAN"
    STD = "STD"
    QUANTILES = "QUANTILES"

    # categorical:
    COUNT = "COUNT"

    # multicategorical:
    MULTI_COUNT = "MULTI_COUNT"

    # timestamp
    YEAR_RANGE = "YEAR_RANGE"
    OLDEST_TIME = "OLDEST_TIME"
    NEWEST_TIME = "NEWEST_TIME"
    MEDIAN_TIME = "MEDIAN_TIME"

    # text_embedded (Also, embedding)
    # Note: For text_embedded, this stats is computed in
    # dataset._update_col_stats, not here.
    EMB_DIM = "EMB_DIM"

    @staticmethod
    def stats_for_stype(stype: torch_frame.stype) -> list[StatType]:
        stats_type = {
            torch_frame.numerical: [
                StatType.MEAN,
                StatType.STD,
                StatType.QUANTILES,
            ],
            torch_frame.categorical: [StatType.COUNT],
            torch_frame.multicategorical: [StatType.MULTI_COUNT],
            torch_frame.sequence_numerical: [
                StatType.MEAN,
                StatType.STD,
                StatType.QUANTILES,
            ],
            torch_frame.timestamp: [
                StatType.YEAR_RANGE,
                StatType.NEWEST_TIME,
                StatType.OLDEST_TIME,
                StatType.MEDIAN_TIME,
            ],
            torch_frame.embedding: [
                StatType.EMB_DIM,
            ]
        }
        return stats_type.get(stype, [])

    def compute(
        self,
        ser: Series,
        sep: str | None = None,
    ) -> Any:
        if self == StatType.MEAN:
            flattened = np.hstack(np.hstack(ser.values))
            finite_mask = np.isfinite(flattened)
            if not finite_mask.any():
                # NOTE: We may just error out here if eveything is NaN
                return np.nan
            return np.mean(flattened[finite_mask]).item()

        elif self == StatType.STD:
            flattened = np.hstack(np.hstack(ser.values))
            finite_mask = np.isfinite(flattened)
            if not finite_mask.any():
                return np.nan
            return np.std(flattened[finite_mask]).item()

        elif self == StatType.QUANTILES:
            flattened = np.hstack(np.hstack(ser.values))
            finite_mask = np.isfinite(flattened)
            if not finite_mask.any():
                return [np.nan, np.nan, np.nan, np.nan, np.nan]
            return np.quantile(
                flattened[finite_mask],
                q=[0, 0.25, 0.5, 0.75, 1],
            ).tolist()

        elif self == StatType.COUNT:
            count = ser.value_counts(ascending=False)
            return count.index.tolist(), count.values.tolist()

        elif self == StatType.MULTI_COUNT:
            ser = ser.apply(lambda row: MultiCategoricalTensorMapper.
                            split_by_sep(row, sep))
            ser = ser.explode().dropna()
            count = ser.value_counts(ascending=False)
            return count.index.tolist(), count.values.tolist()

        elif self == StatType.YEAR_RANGE:
            year_range = ser.dt.year.values
            return [min(year_range), max(year_range)]

        elif self == StatType.NEWEST_TIME:
            return TimestampTensorMapper.to_tensor(pd.Series(
                ser.iloc[-1])).squeeze(0)

        elif self == StatType.OLDEST_TIME:
            return TimestampTensorMapper.to_tensor(pd.Series(
                ser.iloc[0])).squeeze(0)

        elif self == StatType.MEDIAN_TIME:
            return TimestampTensorMapper.to_tensor(
                pd.Series(ser.iloc[len(ser) // 2])).squeeze(0)

        elif self == StatType.EMB_DIM:
            return len(ser[0])


_default_values = {
    StatType.MEAN: np.nan,
    StatType.STD: np.nan,
    StatType.QUANTILES: [np.nan, np.nan, np.nan, np.nan, np.nan],
    StatType.COUNT: ([], []),
    StatType.MULTI_COUNT: ([], []),
    StatType.YEAR_RANGE: [-1, -1],
    StatType.NEWEST_TIME: torch.tensor([-1, -1, -1, -1, -1, -1, -1]),
    StatType.OLDEST_TIME: torch.tensor([-1, -1, -1, -1, -1, -1, -1]),
    StatType.MEDIAN_TIME: torch.tensor([-1, -1, -1, -1, -1, -1, -1]),
    StatType.EMB_DIM: -1,
}


def compute_col_stats(
    ser: Series,
    stype: torch_frame.stype,
    sep: str | None = None,
    time_format: str | None = None,
) -> dict[StatType, Any]:
    if stype == torch_frame.numerical:
        ser = ser.mask(ser.isin([np.inf, -np.inf]), np.nan)
        if not ptypes.is_numeric_dtype(ser):
            raise TypeError("Numerical series contains invalid entries. "
                            "Please make sure your numerical series "
                            "contains only numerical values or nans.")
    if ser.isnull().all():
        # NOTE: We may just error out here if eveything is NaN
        stats = {
            stat_type: _default_values[stat_type]
            for stat_type in StatType.stats_for_stype(stype)
        }
    else:
        if stype == torch_frame.timestamp:
            ser = pd.to_datetime(ser, format=time_format)
            ser = ser.sort_values()
        stats = {
            stat_type: stat_type.compute(ser.dropna(), sep)
            for stat_type in StatType.stats_for_stype(stype)
        }

    return stats
