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
        task_type (TaskType): The task type.
    """
    def __init__(self, task_type: TaskType):
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

    def _fit(self, tf_train: TensorFrame):
        if tf_train.y is None:
            raise RuntimeError(
                "MutualInformationSort cannot be used when target column"
                " is None.")
        if stype.categorical in tf_train.col_names_dict:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")
        mi_scores = self.mi_func(tf_train.x_dict[stype.numerical], tf_train.y)
        self.mi_ranks = np.argsort(-mi_scores)
        col_names = tf_train.col_names_dict[stype.numerical]
        ranks = {col_names[self.mi_ranks[i]]: i for i in range(len(col_names))}

        self.reordered_col_names = tf_train.col_names_dict[
            stype.numerical].copy()

        for col, rank in ranks.items():
            self.reordered_col_names[rank] = col

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if tf.col_names_dict[stype.categorical]:
            raise ValueError("The transform can be only used on TensorFrame"
                             " with numerical only features.")

        tf.x_dict[stype.numerical] = tf.x_dict[stype.numerical][:,
                                                                self.mi_ranks]

        tf.col_names_dict[stype.numerical] = self.reordered_col_names

        return tf
