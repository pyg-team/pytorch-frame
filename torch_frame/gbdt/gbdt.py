from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor

from torch_frame import TaskType, TensorFrame


class GBDT:
    r"""Base class for GBDT (Gradient Boosting Decision Trees) models used as
    strong baseline.

    Args:
        task_type (TaskType): The task type.
        num_classes (int, optional): If the task is multiclass classification,
            an optional num_classes can be used to specify the number of
            classes. Otherwise, we infer the value from the train data.
    """
    def __init__(self, task_type: TaskType, num_classes: Optional[int] = None):
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
        self._num_classes = num_classes

    @abstractmethod
    def _tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
              num_trials: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, tf_train: TensorFrame) -> Tensor:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        r"""Whether the GBDT is already fitted."""
        return self._is_fitted

    def tune(self, tf_train: TensorFrame, tf_val: TensorFrame, num_trials: int,
             *args, **kwargs):
        r"""Fit the model by performing hyperparameter tuning using Optuna. The
        number of trials is specified by num_trials.

        Args:
            tf_train (TensorFrame): The train data in :obt:`TensorFrame`.
            tf_val (TensorFrame): The validation data in :obj:`TensorFrame`.
            num_trials (int): Number of trials to perform hyper-
                parameter search.
        """
        if tf_train.y is None:
            raise RuntimeError("tf_train.y must be a Tensor, but None given.")
        if tf_val.y is None:
            raise RuntimeError("tf_val.y must be a Tensor, but None given.")
        self._tune(tf_train, tf_val, num_trials=num_trials, *args, **kwargs)
        self._is_fitted = True

    def predict(self, tf_test: TensorFrame) -> Tensor:
        r"""Predict the labels/values of the test data on the fitted model.

        Returns:
            pred (Tensor): The prediction output :obj:`Tensor` on the fitted
                model.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}' is not yet fitted. "
                "Please run `tune()` first before attempting "
                "to predict.")
        return self._predict(tf_test)

    @torch.no_grad()
    def compute_metric(self, target: Tensor, pred: Tensor) -> float:
        r"""Compute evaluation metric given test target labels :obj:`Tensor`
        and pred :obj:`Tensor`. Target contains the target values or labels;
        pred contains the prediction output from calling `predict()` function.

        Returns:
            metric (float): The metric on test data, root mean squared error
                for regression task and accuracy for classification task.
        """
        if self.task_type == TaskType.REGRESSION:
            metric_score = (pred - target).square().mean().sqrt().item()
        else:
            total_correct = (target == pred).sum().item()
            test_size = len(target)
            metric_score = total_correct / test_size
        return metric_score
