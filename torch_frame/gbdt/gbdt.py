from __future__ import annotations

import os
from abc import abstractmethod

import torch
from torch import Tensor

from torch_frame import Metric, TaskType, TensorFrame

DEFAULT_METRIC = {
    TaskType.REGRESSION: Metric.RMSE,
    TaskType.BINARY_CLASSIFICATION: Metric.ROCAUC,
    TaskType.MULTICLASS_CLASSIFICATION: Metric.ACCURACY,
}


class GBDT:
    r"""Base class for GBDT (Gradient Boosting Decision Trees) models used as
    strong baseline.

    Args:
        task_type (TaskType): The task type.
        num_classes (int, optional): If the task is multiclass classification,
            an optional num_classes can be used to specify the number of
            classes. Otherwise, we infer the value from the train data.
        metric (Metric, optional): Metric to optimize for, e.g.,
            :obj:`Metric.MAE`. If :obj:`None`, it will default to
            :obj:`Metric.RMSE` for regression, :obj:`Metric.ROCAUC` for binary
            classification, and :obj:`Metric.ACCURACY` for multi-
            class classification. (default: :obj:`None`).
    """
    def __init__(
        self,
        task_type: TaskType,
        num_classes: int | None = None,
        metric: Metric | None = None,
    ):
        self.task_type = task_type
        self._is_fitted: bool = False
        self._num_classes = num_classes

        # Set up metric
        self.metric = DEFAULT_METRIC[task_type]
        if metric is not None:
            if metric.supports_task_type(task_type):
                self.metric = metric
            else:
                raise ValueError(
                    f"{task_type} does not support {metric}. Please choose "
                    f"from {task_type.supported_metrics}.")

    @abstractmethod
    def _tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
              num_trials: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, tf_train: TensorFrame) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _load(self, path: str) -> None:
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
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if tf_train.y is None:
            raise RuntimeError("tf_train.y must be a Tensor, but None given.")
        if tf_val.y is None:
            raise RuntimeError("tf_val.y must be a Tensor, but None given.")
        self._tune(tf_train, tf_val, num_trials=num_trials, *args, **kwargs)
        self._is_fitted = True

    def predict(self, tf_test: TensorFrame) -> Tensor:
        r"""Predict the labels/values of the test data on the fitted model and
        returns its predictions.

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

    def save(self, path: str) -> None:
        r"""Save the model.

        Args:
            path (str): The path to save tuned GBDTs model.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not yet fitted. Please run "
                f"`tune()` first before attempting to save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        r"""Load the model.

        Args:
            path (str): The path to load tuned GBDTs model.
        """
        self._load(path)
        self._is_fitted = True

    @torch.no_grad()
    def compute_metric(
        self,
        target: Tensor,
        pred: Tensor,
    ) -> float:
        r"""Compute evaluation metric given target labels :obj:`Tensor` and
        pred :obj:`Tensor`. Target contains the target values or labels; pred
        contains the prediction output from calling `predict()` function.

        Returns:
            score (float): Computed metric score.
        """
        if self.metric == Metric.RMSE:
            score = (pred - target).square().mean().sqrt().item()
        elif self.metric == Metric.MAE:
            score = (pred - target).abs().mean().item()
        elif self.metric == Metric.ROCAUC:
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(target.cpu(), pred.cpu())
        elif self.metric == Metric.ACCURACY:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = pred > 0.5
            total_correct = (target == pred).sum().item()
            test_size = len(target)
            score = total_correct / test_size
        elif self.metric == Metric.R2:
            from sklearn.metrics import r2_score
            score = r2_score(target.cpu(), pred.cpu())
        else:
            raise ValueError(f'{self.metric} is not supported.')
        return score
