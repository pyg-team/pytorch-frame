from enum import Enum
from typing import Any, Dict, List

import numpy as np

import torch_frame
from torch_frame.typing import Series


class StatType(Enum):
    r"""The different types for column statistics.

    Attributes:
        MEAN: The average value of a numerical column.
        STD: The standard deviation of a numerical column.
        QUANTILES: The minimum, first quartile, median, third quartile,
            and the maximum of a numerical column.
        COUNT: The count of each category in a categorical column.
    """
    # Numerical:
    MEAN = 'MEAN'
    STD = 'STD'
    QUANTILES = 'QUANTILES'

    # Categorical:
    COUNT = 'COUNT'

    @staticmethod
    def stats_for_stype(stype: torch_frame.stype) -> List['StatType']:
        if stype == torch_frame.numerical:
            return [
                StatType.MEAN,
                StatType.STD,
                StatType.QUANTILES,
            ]
        elif stype == torch_frame.categorical:
            return [
                StatType.COUNT,
            ]
        elif stype == torch_frame.text_embedded:
            return []

        raise NotImplementedError(f"Invalid semantic type '{stype.value}'")

    def compute(self, ser: Series) -> Any:
        if self == StatType.MEAN:
            return np.mean(ser.values).item()

        elif self == StatType.STD:
            return np.std(ser.values).item()

        elif self == StatType.QUANTILES:
            return np.quantile(ser.values, [0, 0.25, 0.5, 0.75, 1]).tolist()

        elif self == StatType.COUNT:
            count = ser.value_counts(ascending=False)
            return count.index.tolist(), count.values.tolist()

        raise NotImplementedError(f"Invalid stat type '{self.value}'")


def compute_col_stats(
    ser: Series,
    stype: torch_frame.stype,
) -> Dict[StatType, Any]:

    if stype == torch_frame.numerical:
        ser = ser.mask(ser.isin([np.inf, -np.inf]), np.nan)

    ser = ser.dropna()

    return {
        stat_type: stat_type.compute(ser)
        for stat_type in StatType.stats_for_stype(stype)
    }
