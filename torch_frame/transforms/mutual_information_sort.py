import numpy as np
import torch
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from torch import Tensor

from torch_frame import NAStrategy, TaskType, TensorFrame, stype
from torch_frame.transforms import FittableBaseTransform


class MutualInformationSort(FittableBaseTransform):
    r"""Sorts the numerical features of input :obj:`TensorFrame` based
        on mutual information.

    Args:
        task_type (TaskType): The task type.
    """
    def __init__(self, task_type: TaskType,
                 na_strategy: NAStrategy = NAStrategy.MEAN):
        super().__init__()

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

        self.na_strategy = na_strategy

    def _replace_nans(self, x: Tensor):
        r"""Replace NaNs based on Imputing Strategy.

        Args:
            x (Tensor): Input :obj:`Tensor` whose NaN values are to be
                replaced.
            strategy (ImputingStrategy): Strategy used for imputing NaN values
                in numerical features.
        Returns:
            x (Tensor): Output :obj:`Tensor` with NaN values replaced.
        """
        x = x.clone()
        for col in range(x.size(1)):
            column_data = x[:, col]
            nan_mask = torch.isnan(column_data)
            if not nan_mask.any():
                continue
            valid_data = column_data[~nan_mask]
            if self.na_strategy == NAStrategy.MEAN:
                fill_value = valid_data.mean()
            elif self.na_strategy == NAStrategy.ZEROS:
                fill_value = torch.tensor(0.)
            column_data[nan_mask] = fill_value
        return x

    def _fit(self, tf_train: TensorFrame):
        if tf_train.y is None:
            raise RuntimeError(
                "'{self.__class__.__name__}' cannot be used when target column"
                " is None.")
        if stype.categorical in tf_train.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        x = tf_train.x_dict[stype.numerical]
        if torch.isnan(x).any():
            x = self._replace_nans(x)
        mi_scores = self.mi_func(x.cpu(), tf_train.y.cpu())
        self.mi_ranks = np.argsort(-mi_scores)
        col_names = tf_train.col_names_dict[stype.numerical]
        ranks = {col_names[self.mi_ranks[i]]: i for i in range(len(col_names))}
        self.reordered_col_names = tf_train.col_names_dict[
            stype.numerical].copy()

        for col, rank in ranks.items():
            self.reordered_col_names[rank] = col

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if stype.categorical in tf.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")

        tf.x_dict[stype.numerical] = tf.x_dict[stype.numerical][:,
                                                                self.mi_ranks]

        tf.col_names_dict[stype.numerical] = self.reordered_col_names

        return tf
