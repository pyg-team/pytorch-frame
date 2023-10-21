from abc import abstractmethod
from typing import Dict, Optional

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
            tf_train (TensorFrame): The train data in :class:`TensorFrame`.
            tf_val (TensorFrame): The validation data in :class:`TensorFrame`.
            num_trials (int): Number of trials to perform hyper-parameter
                search.
        """
        if tf_train.y is None:
            raise RuntimeError("tf_train.y must be a Tensor, but None given.")
        if tf_val.y is None:
            raise RuntimeError("tf_val.y must be a Tensor, but None given.")
        self._tune(tf_train, tf_val, num_trials=num_trials, *args, **kwargs)
        self._is_fitted = True

    def predict(self, tf_test: TensorFrame) -> Tensor:
        r"""Predict the labels/values of the test data on the fitted model and
        returns its predictions:

        - :obj:`TaskType.REGRESSION`: Returns raw numerical values.

        - :obj:`TaskType.BINARY_CLASSIFICATION`: Returns the probability of
          being positive.

        - :obj:`TaskType.MULTICLASS_CLASSIFICATION`: Returns the class label
          predictions.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}' is not yet fitted. Please run "
                f"`tune()` first before attempting to predict.")
        pred = self._predict(tf_test)
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            assert pred.ndim == 2
        else:
            assert pred.ndim == 1
        assert len(pred) == len(tf_test)
        return pred

    @property
    def metric(self) -> str:
        r"""Metric to compute for different tasks. root mean squared error for
        regression, ROC-AUC for binary classification, and accuracy for
        multi-label classification task."""
        if self.task_type == TaskType.REGRESSION:
            return 'rmse'
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            return 'rocauc'
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return 'acc'
        else:
            raise ValueError(f"metric is not defined for {self.task_type}")

    @torch.no_grad()
    def compute_metric(self, target: Tensor, pred: Tensor) -> Dict[str, float]:
        r"""Compute evaluation metric given target labels :obj:`Tensor` and
        pred :obj:`Tensor`. Target contains the target values or labels; pred
        contains the prediction output from calling `predict()` function.

        Returns:
            metric (Dict[str, float]): A dictionary containing the metric name
                and the metric value.
        """
        if self.metric == 'rmse':
            metric = {
                self.metric: (pred - target).square().mean().sqrt().item()
            }
        elif self.metric == 'rocauc':
            from sklearn.metrics import roc_auc_score
            metric = {self.metric: roc_auc_score(target.cpu(), pred.cpu())}
        elif self.metric == 'acc':
            total_correct = (target == pred).sum().item()
            test_size = len(target)
            metric = {self.metric: total_correct / test_size}
        return metric
