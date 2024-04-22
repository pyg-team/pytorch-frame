from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torch_frame import NAStrategy, TaskType, TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.transforms import FittableBaseTransform


class MutualInformationSort(FittableBaseTransform):
    r"""A transform that sorts the numerical features of input
        :class:`TensorFrame` object based on mutual information.

    Args:
        task_type (TaskType): The task type.
        na_strategy (NAStrategy): Strategy used for imputing NaN values
            in numerical features.
    """
    def __init__(self, task_type: TaskType,
                 na_strategy: NAStrategy = NAStrategy.MEAN):
        super().__init__()

        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )

        if task_type in [
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.BINARY_CLASSIFICATION
        ]:
            self.mi_func = mutual_info_classif
        elif task_type == TaskType.REGRESSION:
            self.mi_func = mutual_info_regression
        else:
            raise ValueError(
                f"'{self.__class__.__name__}' can be only used on binary "
                "classification,  multiclass classification or regression "
                f"task, but got {task_type}.")
        if not na_strategy.is_numerical_strategy:
            raise RuntimeError(
                f"Cannot use {na_strategy} for numerical features.")
        self.na_strategy = na_strategy

    def _fit(self, tf_train: TensorFrame, col_stats: dict[str, dict[StatType,
                                                                    Any]]):
        if tf_train.y is None:
            raise RuntimeError(
                "'{self.__class__.__name__}' cannot be used when target column"
                " is None.")
        if (stype.categorical in tf_train.col_names_dict
                and len(tf_train.col_names_dict[stype.categorical]) != 0):
            raise ValueError(f"'{self.__class__.__name__}' can be only used"
                             " on TensorFrame with numerical only features.")
        feat_train = tf_train.feat_dict[stype.numerical]
        y_train = tf_train.y
        if torch.isnan(feat_train).any():
            feat_train = self._replace_nans(feat_train, self.na_strategy)
        if torch.isnan(tf_train.y).any():
            not_nan_indices = ~torch.isnan(y_train)
            if not not_nan_indices.any():
                raise ValueError(f"'{self.__class__.__name__}' cannot be"
                                 "performed when all target values are"
                                 " nan.")
            y_train = y_train[not_nan_indices]
            feat_train = feat_train[not_nan_indices]
        mi_scores = self.mi_func(feat_train.cpu(), y_train.cpu())
        self.mi_ranks = np.argsort(-mi_scores)
        self.mi_scores = mi_scores[self.mi_ranks]
        col_names = tf_train.col_names_dict[stype.numerical]
        ranks = {col_names[self.mi_ranks[i]]: i for i in range(len(col_names))}
        self.reordered_col_names = tf_train.col_names_dict[
            stype.numerical].copy()

        for col, rank in ranks.items():
            self.reordered_col_names[rank] = col
        self._transformed_stats = col_stats

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if tf.col_names_dict.keys() != {stype.numerical}:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")

        tf.feat_dict[stype.numerical] = tf.feat_dict[
            stype.numerical][:, self.mi_ranks]

        tf.col_names_dict[stype.numerical] = self.reordered_col_names

        # set lazy attribute for meta features
        tf.mi_scores = torch.tensor(self.mi_scores, dtype=torch.float32,
                                    device=tf.device)

        return tf
