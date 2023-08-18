from enum import Enum
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import torch_frame
from torch_frame.typing import Series


class StatType(Enum):
    # Numerical:
    MEAN = 'MEAN'
    STD = 'STD'

    # Categorical:
    CATEGORY_COUNTS = 'CATEGORY_COUNTS'

    @staticmethod
    def stats_for_stype(stype: torch_frame.stype) -> List['StatType']:
        if stype == torch_frame.numerical:
            return [
                StatType.MEAN,
                StatType.STD,
            ]
        elif stype == torch_frame.categorical:
            return [
                StatType.CATEGORY_COUNTS,
            ]

        raise NotImplementedError(f"Invalid semantic type '{stype.value}'")

    def compute(self, ser: Series) -> Any:
        if self == StatType.MEAN:
            return np.mean(ser.values).item()

        elif self == StatType.STD:
            return np.std(ser.values).item()

        elif self == StatType.CATEGORY_COUNTS:
            counts = ser.value_counts(ascending=False)
            return counts.index.tolist(), counts.values.tolist()

        raise NotImplementedError(f"Invalid stat type '{self.value}'")


def compute_col_stats(
    ser: Series,
    stype: torch_frame.stype,
) -> Dict[StatType, Any]:

    with pd.option_context('mode.use_inf_as_na', True):
        ser = ser.dropna()

    return {
        stat_type: stat_type.compute(ser)
        for stat_type in StatType.stats_for_stype(stype)
    }
