from enum import Enum
from typing import List, Union

import pandas as pd
from torch import Tensor


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multiclass classification task.
        BINARY_CLASSIFICATION: Binary classification task.
    """
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


class NAStrategy(Enum):
    r"""Strategy for dealing with NaN values in columns.

    Attributes:
        MEAN: Replacing NaN values with mean
        (numerical :class:`torch_frame.stype`).
        ZEROS: Replacing NaN values with zeros
        (numerical :class:`torch_frame.stype`).
        MOST_FREQUENT: Replacing NaN values with most frequent
            (categorical :class:`torch_frame.stype`).
    """
    MEAN = 'mean'
    MOST_FREQUENT = 'most_frequent'
    ZEROS = 'zeros'

    @property
    def is_categorical_strategy(self):
        return self == NAStrategy.MOST_FREQUENT

    @property
    def is_numerical_strategy(self):
        return self in [NAStrategy.MEAN, NAStrategy.ZEROS]


Series = pd.Series
DataFrame = pd.DataFrame

IndexSelectType = Union[int, List[int], range, slice, Tensor]
ColumnSelectType = Union[str, List[str]]
