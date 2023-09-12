from abc import abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_frame import TaskType, TensorFrame, stype


class GBDT:
    r"""Base class for GBDT (Gradient Boosting Decision Trees)
            models used as strong baseline.

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

    def _to_xgboost_input(self,
                          tf: TensorFrame) -> Tuple[Tensor, Tensor, List[str]]:
        r""" Convert :obj:`TensorFrame` into XGBoost-compatible input
            (test_x, test_y, feat_types).

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.
        Returns:
            test_x (Tensor): Output :obj:`Tensor` by concatenating tensors
                of numerical and categorical features of the input
                :obj:`TensorFrame`.
            test_y (Tensor): Prediction target :obj:`Tensor`.
            feature_types (List[str]): List of feature types: "q" for numerical
                features and "c" for categorical features. The abbreviation
                aligns with xgboost tutorial.
                <https://github.com/dmlc/xgboost/blob/master/doc/
                tutorials/categorical.rst#using-native-interface>
        """
        test_y = tf.y
        if stype.categorical in tf.x_dict and stype.numerical in tf.x_dict:
            test_x = torch.cat(
                (tf.x_dict[stype.numerical], tf.x_dict[stype.categorical]),
                dim=1)
            feature_types = ["q"] * len(tf.col_names_dict[stype.numerical]) + [
                "c"
            ] * len(tf.col_names_dict[stype.categorical])
        elif stype.categorical in tf.x_dict:
            test_x = tf.x_dict[stype.categorical]
            feature_types = ["c"] * len(tf.col_names_dict[stype.categorical])
        elif stype.numerical in tf.x_dict:
            test_x = tf.x_dict[stype.numerical]
            feature_types = ["q"] * len(tf.col_names_dict[stype.numerical])
        else:
            raise ValueError("The input TensorFrame object is empty.")
        return test_x, test_y, feature_types

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
        r""" Fit the model by performing hyperparameter tuning using Optuna.
            The number of trials is specified by
            num_trials.

        Args:
            tf_train (TensorFrame): The train data in :obt:`TensorFrame`.
            tf_val (TensorFrame): The validation data in :obj:`TensorFrame`.
            num_trials (int): Number of trials to perform hyper-
                parameter search.
        """
        self._tune(tf_train, tf_val, num_trials=num_trials, *args, **kwargs)
        self._is_fitted = True

    def predict(self, tf_test: TensorFrame) -> Tensor:
        r""" Predicts the label/result of the test data on the fitted model.

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

    def compute_metric(self, target: Tensor, pred: Tensor) -> float:
        r""" Computes evaluation metric given test target labels
                :obj:`Tensor` and pred :obj:`Tensor`. Target contains
                the target values or labels; pred contains the prediction
                output from calling predict() function.

        Returns:
            metric (float): The metric on test data, mean squared error
                for regression task and accuracy for classification task.
        """
        with torch.no_grad():
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
