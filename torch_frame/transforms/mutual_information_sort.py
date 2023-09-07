import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from torch_frame import TaskType, TensorFrame, stype
from torch_frame.transforms import BaseTransform


class MutualInformationSort(BaseTransform):
    r"""Sorts the numerical features of input :obj:`TensorFrame` based
        on mutual information.

        Args:
            tf_train (TensorFrame): Input :obj:`TensorFrame` containing the
                training data.
            task_type (TaskType): TaskType.REGRESSION or
                TaskType.CLASSIFICATION
    """
    def __init__(self, tf_train: TensorFrame, task_type: TaskType):
        if tf_train.y is None:
            raise RuntimeError(
                "MutualInformationSort cannot be used when target column"
                " is None.")
        if task_type == "classification":
            self.mi_func = mutual_info_classif
        elif task_type == "regression":
            self.mi_func = mutual_info_regression
        if stype.categorical in tf_train.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        mi_scores = self.mi_func(tf_train.x_dict[stype.numerical], tf_train.y)
        mi_ranks = np.argsort(-mi_scores)
        num_cols = tf_train.col_names_dict[stype.numerical]
        self.ranks = {num_cols[mi_ranks[i]]: i for i in range(len(num_cols))}

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
        col_names = tf.col_names_dict[stype.numerical]
        col_idx = {name: index for index, name in enumerate(col_names)}
        idx_rank = {col_idx[key]: value for key, value in self.ranks.items()}
        cols = sorted(idx_rank, key=idx_rank.get)

        tf.x_dict[stype.numerical] = tf.x_dict[stype.numerical][:, cols]

        for col, rank in self.ranks.items():
            tf.col_names_dict[stype.numerical][rank] = col

        return tf
