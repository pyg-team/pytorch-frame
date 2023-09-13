from enum import Enum
from typing import List, Union

import pandas as pd
from torch import Tensor


class TaskType(Enum):
    REGRESSION = 'regression'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    BINARY_CLASSIFICATION = 'binary_classification'
    MULTILABEL_CLASSIFICATION = 'multilabel_classification'

    @property
    def is_classification(self):
        return self in (TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION)

    @property
    def is_regression(self):
        return self == TaskType.REGRESSION


class ImputingStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MOST_FREQUENT = 'most_frequent'
    ZEROS = 'zeros'

    @property
    def is_categorical_strategy(self):
        return self in [ImputingStrategy.MOST_FREQUENT, ImputingStrategy.ZEROS]

    @property
    def is_numerical_strategy(self):
        return self in [ImputingStrategy.MEAN, ImputingStrategy.ZEROS]


Series = pd.Series
DataFrame = pd.DataFrame

IndexSelectType = Union[int, List[int], range, slice, Tensor]
ColumnSelectType = Union[str, List[str]]
