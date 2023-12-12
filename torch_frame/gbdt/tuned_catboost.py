from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torch_frame import DataFrame, Metric, TaskType, TensorFrame, stype
from torch_frame.gbdt import GBDT


class CatBoost(GBDT):
    r"""A CatBoost model implementation with hyper-parameter tuning using
    Optuna.

    This implementation extends GBDT and aims to find optimal hyperparameters
    by optimizing the given objective function.
    """
    def _to_catboost_input(
        self,
        tf,
    ) -> tuple[DataFrame, np.ndarray | None, np.ndarray]:
        r"""Convert :class:`TensorFrame` into CatBoost-compatible input format:
        :obj:`(x, y, cat_features)`.

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.

        Returns:
            x (DataFrame): Output :obj:`Dataframe` by
                concatenating tensors of categorical and numerical features of
                the input :class:`TensorFrame`.
            y (numpy.ndarray, optional): Prediction label.
            cat_features (numpy.ndarray): Array containing indexes of
                categorical features.
        """
        tf = tf.cpu()
        y = tf.y
        if y is not None:
            y: np.ndarray = y.numpy()

        dfs: list[DataFrame] = []
        cat_features: list[np.ndarray] = []
        offset: int = 0

        if stype.categorical in tf.feat_dict:
            feat = tf.feat_dict[stype.categorical].numpy()
            arange = np.arange(offset, offset + feat.shape[1])
            dfs.append(pd.DataFrame(feat, columns=arange))
            cat_features.append(arange)
            offset += feat.shape[1]

        if stype.numerical in tf.feat_dict:
            feat = tf.feat_dict[stype.numerical].numpy()
            arange = np.arange(offset, offset + feat.shape[1])
            dfs.append(pd.DataFrame(feat, columns=arange))
            offset += feat.shape[1]

        if stype.embedding in tf.feat_dict:
            feat = tf.feat_dict[stype.embedding]
            feat = feat.values
            feat = feat.view(feat.size(0), -1).numpy()
            arange = np.arange(offset, offset + feat.shape[1])
            dfs.append(pd.DataFrame(feat, columns=arange))
            offset += feat.shape[1]

        # TODO Add support for other stypes.

        if len(dfs) == 0:
            raise ValueError("The input TensorFrame object is empty.")

        df = pd.concat(dfs, axis=1)
        cat_features = np.concatenate(
            cat_features, axis=0) if len(cat_features) else np.array([])
        return df, y, cat_features

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
        trial: Any,  # optuna.trial.Trial
        train_x: DataFrame,
        train_y: np.ndarray,
        val_x: DataFrame,
        val_y: np.ndarray,
        cat_features: np.ndarray,
        num_boost_round: int,
    ) -> float:
        r"""Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            train_x (DataFrame): Train data.
            train_y (numpy.ndarray): Train label.
            val_x (DataFrame): Validation data.
            val_y (numpy.ndarray): Validation label.
            cat_features (numpy.ndarray): Array containing indexes of
                categorical features.
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
            if self.metric == Metric.RMSE:
                self.params["objective"] = "RMSE"
                self.params["eval_metric"] = "RMSE"
            elif self.metric == Metric.MAE:
                self.params["objective"] = "MAE"
                self.params["eval_metric"] = "MAE"
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.params["objective"] = "Logloss"
            if self.metric == Metric.ROCAUC:
                self.params["eval_metric"] = "AUC"
            elif self.metric == Metric.ACCURACY:
                self.params["eval_metric"] = "Accuracy"
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["objective"] = "MultiClass"
            self.params["eval_metric"] = "Accuracy"
            self.params["classes_count"] = self._num_classes or len(
                np.unique(train_y))
        else:
            raise ValueError(f"{self.__class__.__name__} is not supported for "
                             f"{self.task_type}.")
        boost = catboost.CatBoost(self.params)
        boost = boost.fit(train_x, train_y, cat_features=cat_features,
                          eval_set=[(val_x, val_y)], early_stopping_rounds=50,
                          logging_level="Silent")
        pred = self._predict_helper(boost, val_x)
        score = self.compute_metric(torch.from_numpy(val_y),
                                    torch.from_numpy(pred))
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
        train_x, train_y, cat_features = self._to_catboost_input(tf_train)
        val_x, val_y, _ = self._to_catboost_input(tf_val)
        assert train_y is not None
        assert val_y is not None
        study.optimize(
            lambda trial: self.objective(trial, train_x, train_y, val_x, val_y,
                                         cat_features, num_boost_round),
            num_trials)
        self.params.update(study.best_params)

        self.model = catboost.CatBoost(self.params)
        self.model.fit(train_x, train_y, cat_features=cat_features,
                       eval_set=[(val_x, val_y)], early_stopping_rounds=50,
                       logging_level="Silent")

    def _predict(self, tf_test: TensorFrame) -> Tensor:
        device = tf_test.device
        test_x, _, _ = self._to_catboost_input(tf_test)
        pred = self._predict_helper(self.model, test_x)
        return torch.from_numpy(pred).to(device)

    def _load(self, path: str) -> None:
        import catboost

        self.model = catboost.CatBoost()
        self.model.load_model(path)
