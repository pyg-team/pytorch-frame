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
    COUNT = 'COUNT'

    @staticmethod
    def stats_for_stype(stype: torch_frame.stype) -> List['StatType']:
        if stype == torch_frame.numerical:
            return [
                StatType.MEAN,
                StatType.STD,
            ]
        elif stype == torch_frame.categorical:
            return [
                StatType.COUNT,
            ]

        raise NotImplementedError(f"Invalid semantic type '{stype.value}'")

    def compute(self, ser: Series) -> Any:
        if self == StatType.MEAN:
            return np.mean(ser.values).item()

        elif self == StatType.STD:
            return np.std(ser.values).item()

        elif self == StatType.COUNT:
            count = ser.value_counts(ascending=True)
            return count.index.tolist(), count.values.tolist()

        raise NotImplementedError(f"Invalid stat type '{self.value}'")


def compute_col_stats(
    ser: Series,
    stype: torch_frame.stype,
) -> Dict[StatType, Any]:

    if stype == torch_frame.numerical:
        with pd.option_context('mode.use_inf_as_na', True):
            ser = ser.dropna()
    else:
        ser = ser.dropna()

    return {
        stat_type: stat_type.compute(ser)
        for stat_type in StatType.stats_for_stype(stype)
    }
