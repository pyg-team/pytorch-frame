from abc import abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_frame import TaskType, TensorFrame, stype


class GradientBoostingDecisionTrees():
    r""" Base class for GBDT models used as strong baseline.

        Args:
            task_type (TaskType): The task type.
    """
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.obj = "multi:softmax"
            self.eval_metric = "mlogloss"
        elif task_type == TaskType.REGRESSION:
            self.obj = 'reg:squarederror'
            self.eval_metric = 'rmse'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            self.obj = 'binary:logistic'
            self.eval_metric = 'auc'
        else:
            raise ValueError(
                f"{self.__class__.__name__} is not supported for {task_type}.")
        self._is_fitted: bool = False

    def _tensor_frame_to_numpy(self, tf: TensorFrame) -> np.ndarray:
        r""" Convert :obj:`TensorFrame` into numpy array

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.
        Returns:
            out (np.ndarry): Output numpy array by concatenating tensor
                of numerical and categorical features of the input
                :obj:`TensorFrame` object.
        """

        if stype.categorical in tf.x_dict and stype.numerical in tf.x_dict:
            out = torch.cat(
                (tf.x_dict[stype.numerical], tf.x_dict[stype.categorical]),
                dim=1).cpu().numpy()
        elif stype.categorical in tf.x_dict:
            out = tf.x_dict[stype.categorical].cpu().numpy()
        elif stype.numerical in tf.x_dict:
            out = tf.x_dict[stype.numerical].cpu().numpy()
        else:
            raise ValueError("The input TensorFrame object is empty.")
        return out

    @abstractmethod
    def _fit_tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
                  num_trials: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, tf_train: TensorFrame) -> Tensor:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        r"""Whether the GBDT is already fitted."""
        return self._is_fitted

    def fit_tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
                 num_trials: int, *args, **kwargs):
        r""" Fit the model by performing hyperparameter tuning using Optuna.
            The number of trials is specified by
            num_trials.

        Args:
            tf_train (TensorFrame): The train data.
            tf_val (TensorFrame): The validation data.
            num_trials (int): Number of trials to perform hyper-
                parameter search.
        """
        self._fit_tune(tf_train, tf_val, num_trials=num_trials, *args,
                       **kwargs)
        self._is_fitted = True

    def eval(self, tf_test: TensorFrame) -> float:
        r""" Evaluate the trained model on the given test data.

        Returns:
            metric (float): The metric on test data, mean squared error
                for regression task and accuracy for classification task.
        """
        preds = self.predict(tf_test)
        return self.compute_metric(tf_test, preds)

    def predict(self, tf_test: TensorFrame) -> Tensor:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}' is not yet fitted."
                "Please run `fit_tune()` first before attempting "
                "to predict.")
        return self._predict(tf_test)

    def compute_metric(self, target: Union[TensorFrame, Tensor],
                       pred: Tensor) -> float:
        r""" Computes evaluation metric given test labels
            :obj:`Union[TensorFrame, Tensor]` and pred :obj:`Tensor`.

        Returns:
            metric (float): The metric on test data, mean squared error
                for regression task and accuracy for classification task.
        """
        if isinstance(target, TensorFrame):
            test_y = target.y
        else:
            test_y = target
        if self.task_type == TaskType.REGRESSION:
            metric = nn.MSELoss()
            metric_score = metric(pred, test_y)
        else:
            total_correct = (test_y == pred).sum().item()
            test_size = len(test_y)
            metric_score = total_correct / test_size
        return metric_score
