import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from torch_frame import TensorFrame, stype
from torch_frame.transforms import BaseTransform


class MutualInformationSort(BaseTransform):
    r"""Class that sort the numerical features of a TensorFrame obj based
        on mutual information.
    """
    def __init__(self, tf_train: TensorFrame, task_type):
        r"""Calculate the mutual information score of training data. The
        input tf_train :TensorFrame: should not contain categorical features.

        Args:
            tf_train (TensorFrame): TensorFrame containing the training data.
                :Tensorframe:`[train_size, num_cols]`.
        """
        if task_type == "classification":
            self.mi_func = mutual_info_classif
        elif task_type == "regression":
            self.mi_func = mutual_info_regression
        else:
            raise RuntimeError("task_type can only be classification or "
                               f"regression, but got {task_type}")
        if stype.categorical in tf_train.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        mi_scores = self.mi_func(tf_train.x_dict[stype.numerical], tf_train.y)
        mi_ranks = np.argsort(-mi_scores)
        num_cols = tf_train.col_names_dict[stype.numerical]
        self.ranks = {num_cols[mi_ranks[i]]: i for i in range(len(num_cols))}
        return

    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.

        Args:
            tf (TensorFrame): Input TensorFrame obj

        Returns:
            tf (TensorFrame): Input TensorFrame obj but with numerical
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

        tf.col_names_dict[stype.numerical] = sorted(
            tf.col_names_dict[stype.numerical], key=self.ranks.get)
        return tf
