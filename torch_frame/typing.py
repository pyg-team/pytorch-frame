from enum import Enum
from typing import Dict, List, Mapping, Union

import pandas as pd
from torch import Tensor

from torch_frame.data.multi_nested_tensor import MultiNestedTensor


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
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
        MEAN: Replaces NaN values with the mean of a
            :obj:`torch_frame.numerical` column.
        ZEROS: Replaces NaN values with zeros in a
            :obj:`torch_frame.numerical` column.
        MOST_FREQUENT: Replaces NaN values with the most frequent category of a
            :obj:`torch_frame.categorical` column.
    """
    MEAN = 'mean'
    MOST_FREQUENT = 'most_frequent'
    ZEROS = 'zeros'

    @property
    def is_categorical_strategy(self):
        return self == NAStrategy.MOST_FREQUENT

    @property
    def is_multicategorical_strategy(self):
        return self == NAStrategy.ZEROS

    @property
    def is_numerical_strategy(self):
        return self in [NAStrategy.MEAN, NAStrategy.ZEROS]


Series = pd.Series
DataFrame = pd.DataFrame

IndexSelectType = Union[int, List[int], range, slice, Tensor]
ColumnSelectType = Union[str, List[str]]
TextTokenizationOutputs = Union[List[Mapping[str, Tensor]], Mapping[str,
                                                                    Tensor]]
TensorData = Union[Tensor, MultiNestedTensor, Dict[str, MultiNestedTensor]]
