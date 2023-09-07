import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from torch_frame import TaskType, TensorFrame, stype
from torch_frame.transforms import FittableBaseTransform


class MutualInformationSort(FittableBaseTransform):
    r"""Sorts the numerical features of input :obj:`TensorFrame` based
        on mutual information.

    Args:
        tf_train (TensorFrame): Input :obj:`TensorFrame` containing the
            training data.
        task_type (TaskType): The task type.
    """
    def __init__(self, task_type: TaskType):
        if task_type in [
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.BINARY_CLASSIFICATION
        ]:
            self.mi_func = mutual_info_classif
        elif task_type == TaskType.REGRESSION:
            self.mi_func = mutual_info_regression
        else:
            raise ValueError(
                "MutualInformationSort can be only used on binary "
                "classification,  multiclass classification or regression "
                f"task, but got {task_type}.")

    def fit(self, tf_train: TensorFrame):
        if tf_train.y is None:
            raise RuntimeError(
                "MutualInformationSort cannot be used when target column"
                " is None.")
        if stype.categorical in tf_train.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        mi_scores = self.mi_func(tf_train.x_dict[stype.numerical], tf_train.y)
        mi_ranks = np.argsort(-mi_scores)
        num_cols = tf_train.col_names_dict[stype.numerical]
        self.ranks = {num_cols[mi_ranks[i]]: i for i in range(len(num_cols))}
        col_names = tf_train.col_names_dict[stype.numerical]
        col_idx = {name: index for index, name in enumerate(col_names)}
        idx_rank = {col_idx[key]: value for key, value in self.ranks.items()}
        self.cols = sorted(idx_rank, key=idx_rank.get)
        self.reordered_col_names = tf_train.col_names_dict[
            stype.numerical].copy()

        for col, rank in self.ranks.items():
            self.reordered_col_names[rank] = col

    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame`.

        Returns:
            tf (TensorFrame): Input :obj:`TensorFrame` with numerical
                features sorted based on mutual information.
        """
        if tf.col_names_dict[stype.categorical]:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        if self.ranks is None:
            raise RuntimeError("The transform has not been fitted yet, "
                               "please fit the transform on train data.")

        tf.x_dict[stype.numerical] = tf.x_dict[stype.numerical][:, self.cols]

        tf.col_names_dict[stype.numerical] = self.reordered_col_names

        return tf
