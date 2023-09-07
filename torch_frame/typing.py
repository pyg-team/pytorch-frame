from enum import Enum
from typing import List, Union

import pandas as pd
from torch import Tensor


class TaskType(Enum):
    REGRESSION = 'regression'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'


Series = pd.Series
DataFrame = pd.DataFrame

IndexSelectType = Union[int, List[int], range, slice, Tensor]
ColumnSelectType = Union[str, List[str]]
