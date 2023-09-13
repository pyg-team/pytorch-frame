import numpy as np
import torch
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from torch import Tensor

from torch_frame import ImputingStrategy, TaskType, TensorFrame, stype
from torch_frame.transforms import FittableBaseTransform


class MutualInformationSort(FittableBaseTransform):
    r"""Sorts the numerical features of input :obj:`TensorFrame` based
        on mutual information.

    Args:
        task_type (TaskType): The task type.
    """
    def __init__(self, task_type: TaskType,
                 strategy: ImputingStrategy = ImputingStrategy.MEAN):
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

        if (not strategy.is_numerical_strategy):
            raise RuntimeError(
                f"Cannot use {strategy} for imputing missing numerical "
                "features.")
        self.strategy = strategy

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
        for col in range(x.size(1)):
            column_data = x[:, col]
            nan_mask = torch.isnan(column_data)
            valid_data = column_data[~nan_mask]
            num_nans = nan_mask.sum().item()
            if num_nans == 0:
                continue
            if self.strategy == ImputingStrategy.MEAN:
                fill_value = valid_data.mean()
            elif self.strategy == ImputingStrategy.ZEROS:
                fill_value = torch.tensor(0.)
            column_data[nan_mask] = fill_value
            x[:, col] = column_data
        return x

    def _fit(self, tf_train: TensorFrame):
        if tf_train.y is None:
            raise RuntimeError(
                "'{self.__class__.__name__}' cannot be used when target column"
                " is None.")
        if stype.categorical in tf_train.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        x = tf_train.x_dict[stype.numerical].clone()
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
