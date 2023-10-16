from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torch_frame import DataFrame, TaskType, TensorFrame, stype
from torch_frame.gbdt import GBDT


class CatBoost(GBDT):
    r"""A CatBoost model implementation with hyper-parameter tuning using
    Optuna.

    This implementation extends GBDT and aims to find optimal hyperparameters
    by optimizing the given objective function.
    """
    def _to_catboost_input(
            self, tf) -> Tuple[DataFrame, np.ndarray, Optional[np.ndarray]]:
        r"""Convert :obj:`TensorFrame` into CatBoost-compatible input format:
        :obj:`(x, y, cat_features)`.

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.
        Returns:
            x (DataFrame): Output :obj:`Dataframe` by
                concatenating tensors of categorical and numerical features of
                the input :obj:`TensorFrame`.
            y (numpy.ndarray): Prediction target :obj:`numpy.ndarray`.
            cat_features (numpy.ndarray, optional): Array containing indexes of
                categorical features :obj:`numpy.ndarray`.
        """
        tf = tf.cpu()
        y = tf.y
        assert y is not None
        if (stype.categorical in tf.feat_dict
                and stype.numerical in tf.feat_dict):
            categorical_df = pd.DataFrame(
                tf.feat_dict[stype.categorical],
                columns=tf.col_names_dict[stype.categorical])
            numerical_df = pd.DataFrame(
                tf.feat_dict[stype.numerical],
                columns=tf.col_names_dict[stype.numerical])
            df = pd.concat([categorical_df, numerical_df], axis=1)
            cat_features = np.arange(tf.feat_dict[stype.categorical].shape[1])
        elif stype.categorical in tf.feat_dict:
            df = pd.DataFrame(tf.feat_dict[stype.categorical],
                              columns=tf.col_names_dict[stype.categorical])
            cat_features = np.arange(tf.feat_dict[stype.categorical].shape[1])
        elif stype.numerical in tf.feat_dict:
            df = pd.DataFrame(tf.feat_dict[stype.numerical],
                              columns=tf.col_names_dict[stype.numerical])
            cat_features = None
        else:
            raise ValueError("The input TensorFrame object is empty.")
        return df, y.numpy(), cat_features

    def _predict_helper(
        self,
        model: Any,  # catboost.CatBoost
        x: DataFrame,
    ) -> np.ndarray:
        r"""A helper function that applies the catboost model on DataFrame
        :obj:`x`.

        Args:
            model (catboost.CatBoost): The catboost model.
            x (DataFrame): The input`DataFrame.

        Returns:
            pred (np.nparray): The prediction output.
        """
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            prediction_type = "Probability"
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            prediction_type = "Class"
        else:
            prediction_type = "RawFormulaVal"

        pred = model.predict(x, prediction_type=prediction_type)

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            # Get the positive probability
            pred = pred[:, 1]
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Flatten (num_data, 1) into (num_data,)
            pred = pred.flatten()

        return pred

    def objective(
        self,
        trial,
        tf_train: TensorFrame,
        tf_val: TensorFrame,
        num_boost_round: int,
    ):
        r""" Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            tf_train (TensorFrame): Train data.
            tf_val (TensorFrame): Validation data.
            num_boost_round (int): Number of boosting round.

        Returns:
            float: Best objective value. Root mean squared error for
            regression task and accuracy for classification task.
        """
        import catboost

        self.params = {
            "iterations":
            num_boost_round,
            "depth":
            trial.suggest_int("depth", 3, 11),
            "boosting_type":
            trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bagging_temperature":
            trial.suggest_float("bagging_temperature", 0, 1),
            "colsample_bylevel":
            trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "leaf_estimation_iterations":
            trial.suggest_int("leaf_estimation_iterations", 1, 11),
            "l2_leaf_reg":
            trial.suggest_float("l2_leaf_reg", 1, 11, log=True),
            "eta":
            trial.suggest_float("eta", 1e-6, 1.0, log=True),
        }
        if self.task_type == TaskType.REGRESSION:
            self.params["objective"] = trial.suggest_categorical(
                "objective", ["RMSE", "MAE"])
            self.params["eval_metric"] = "RMSE"
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.params["objective"] = trial.suggest_categorical(
                "objective", ["CrossEntropy", "Logloss"])
            self.params["eval_metric"] = "AUC"
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["objective"] = "MultiClass"
            self.params["eval_metric"] = "Accuracy"
            self.params["classes_count"] = self._num_classes or len(
                np.unique(tf_train.y.cpu().numpy()))
        else:
            raise ValueError(f"{self.__class__.__name__} is not supported for "
                             f"{self.task_type}.")
        train_x, train_y, cat_features = self._to_catboost_input(tf_train)
        eval_x, eval_y, _ = self._to_catboost_input(tf_val)
        boost = catboost.CatBoost(self.params)
        boost = boost.fit(train_x, train_y, cat_features=cat_features,
                          eval_set=[(eval_x, eval_y)],
                          early_stopping_rounds=50, logging_level="Silent")
        pred = self._predict_helper(boost, eval_x)
        score = self.compute_metric(torch.from_numpy(eval_y),
                                    torch.from_numpy(pred))[self.metric]
        return score

    def _tune(
        self,
        tf_train: TensorFrame,
        tf_val: TensorFrame,
        num_trials: int,
        num_boost_round=2000,
    ):
        import catboost
        import optuna

        if self.task_type == TaskType.REGRESSION:
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, tf_train, tf_val,
                                         num_boost_round), num_trials)
        self.params.update(study.best_params)
        train_x, train_y, cat_features = self._to_catboost_input(tf_train)
        eval_x, eval_y, _ = self._to_catboost_input(tf_val)
        self.model = catboost.CatBoost(self.params)
        self.model.fit(train_x, train_y, cat_features=cat_features,
                       eval_set=[(eval_x, eval_y)], early_stopping_rounds=50,
                       logging_level="Silent")

    def _predict(self, tf_test: TensorFrame) -> Tensor:
        device = tf_test.device
        test_x, _, _ = self._to_catboost_input(tf_test)
        pred = self._predict_helper(self.model, test_x)
        return torch.from_numpy(pred).to(device)
