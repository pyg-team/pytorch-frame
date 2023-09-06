import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from torch_frame import TensorFrame, stype
from torch_frame.transforms import BaseTransform


class SortFeaturesBasedOnMutualInformation(BaseTransform):
    r"""Base class for transform that transforms the input tensorflow
    to output tensorflow."""
    def __init__(self, tf_train: TensorFrame, task_type):
        if task_type == "classification":
            self.mi_func = mutual_info_regression
        elif task_type == "regression":
            self.mi_func = mutual_info_classif
        else:
            raise RuntimeError("task_type can only be classification or "
                               f"regression, but got {task_type}")
        if tf_train.col_names_dict[stype.categorical]:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        mi_scores = self.mi_func(tf_train.x_dict[stype.numerical], tf_train.y)
        mi_ranks = np.argsort(-mi_scores)
        print("sorted mi_ranks")
        self.ranks = {
            col_name: rank
            for col_name, rank in sorted(
                zip(tf_train.col_names_dict[stype.numerical], mi_ranks))
        }

        return

    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process column-wise 3-dimensional tensor into another column-wise
        3-dimensional tensor.

        Args:
            tf (TensorFrame): Input column-wise tensor of shape
                :Tensorframe:`[batch_size, num_cols, hidden_channels]`.
        """
        if tf.col_names_dict[stype.categorical]:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        col_names = tf.col_names_dict[stype.numerical]
        col_idx = {name: index for index, name in enumerate(col_names)}
        idx_rank = {col_idx[key]: value for key, value in self.ranks.items()}
        cols = sorted(idx_rank, key=idx_rank.get)
        tf.x_dict[stype.numerical] = tf.x_dict[:, cols]
        tf.col_names_dict[stype.numerical] = sorted(
            tf.col_names_dict[stype.numerical], key=col_idx.get)
        return tf
