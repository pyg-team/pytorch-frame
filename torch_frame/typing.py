from __future__ import annotations

from enum import Enum
from typing import Dict, List, Mapping, Union

import pandas as pd
import torch
from torch import Tensor

from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor

WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
WITH_PT24 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 4


class Metric(Enum):
    r"""The metric.

    Attributes:
        ACCURACY: accuracy
        ROCAUC: rocauc
        RMSE: rmse
        MAE: mae
    """
    ACCURACY = 'accuracy'
    ROCAUC = 'rocauc'
    RMSE = 'rmse'
    MAE = 'mae'
    R2 = 'r2'

    def supports_task_type(self, task_type: TaskType) -> bool:
        return self in task_type.supported_metrics


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
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION)

    @property
    def is_regression(self) -> bool:
        return self == TaskType.REGRESSION

    @property
    def supported_metrics(self) -> list[Metric]:
        if self == TaskType.REGRESSION:
            return [Metric.RMSE, Metric.MAE, Metric.R2]
        elif self == TaskType.BINARY_CLASSIFICATION:
            return [Metric.ACCURACY, Metric.ROCAUC]
        elif self == TaskType.MULTICLASS_CLASSIFICATION:
            return [Metric.ACCURACY]
        else:
            return []


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
    OLDEST_TIMESTAMP = 'oldest_timestamp'
    NEWEST_TIMESTAMP = 'newest_timestamp'
    MEDIAN_TIMESTAMP = 'median_timestamp'

    @property
    def is_categorical_strategy(self) -> bool:
        return self == NAStrategy.MOST_FREQUENT

    @property
    def is_multicategorical_strategy(self) -> bool:
        return self == NAStrategy.ZEROS

    @property
    def is_numerical_strategy(self) -> bool:
        return self in [NAStrategy.MEAN, NAStrategy.ZEROS]

    @property
    def is_timestamp_strategy(self) -> bool:
        return self in [
            NAStrategy.NEWEST_TIMESTAMP,
            NAStrategy.OLDEST_TIMESTAMP,
            NAStrategy.MEDIAN_TIMESTAMP,
        ]


Series = pd.Series
DataFrame = pd.DataFrame

IndexSelectType = Union[int, List[int], range, slice, Tensor]
ColumnSelectType = Union[str, List[str]]
TextTokenizationMapping = Mapping[str, Tensor]
TextTokenizationOutputs = Union[List[TextTokenizationMapping],
                                TextTokenizationMapping]
TensorData = Union[
    Tensor,
    MultiNestedTensor,
    MultiEmbeddingTensor,
    Dict[str, MultiNestedTensor],
]
