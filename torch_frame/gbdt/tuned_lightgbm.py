from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torch_frame import DataFrame, Metric, TaskType, TensorFrame, stype
from torch_frame.gbdt import GBDT


class LightGBM(GBDT):
    r"""LightGBM implementation with hyper-parameter tuning using Optuna.

    This implementation extends GBDT and aims to find optimal hyperparameters
    by optimizing the given objective function.
    """
    def _to_lightgbm_input(
        self,
        tf: TensorFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        r"""Convert :class:`TensorFrame` into LightGBM-compatible input format:
        :obj:`(feat, y, cat_features)`.

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.

        Returns:
            feat (numpy.ndarray): Output :obj:`numpy.ndarray` by
                concatenating tensors of numerical and categorical features of
                the input :class:`TensorFrame`.
            y (numpy.ndarray): Prediction target :obj:`numpy.ndarray`.
            cat_features (List[int]): Array containing indexes of
                categorical features :obj:`List[int]`.
        """
        tf = tf.cpu()
        y = tf.y
        assert y is not None

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
            cat_features, axis=0).tolist() if len(cat_features) else []

        return df, y.numpy(), cat_features

    def _predict_helper(
        self,
        model: Any,
        x: DataFrame,
    ) -> np.ndarray:
        r"""A helper function that applies the lightgbm model on DataFrame
        :obj:`x`.

        Args:
            model (lightgbm.Booster): The lightgbm model.
            x (DataFrame): The input `DataFrame`.

        Returns:
            pred (np.ndarray): The prediction output.
        """
        pred = model.predict(x)
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(axis=1)

        return pred

    def objective(
        self,
        trial: Any,
        tf_train: TensorFrame,
        tf_val: TensorFrame,
        num_boost_round: int,
    ):
        r"""Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            tf_train (TensorFrame): Train data.
            tf_val (TensorFrame): Validation data.
            num_boost_round (int): Number of boosting round.

        Returns:
            float: Best objective value. Mean absolute error for
            regression task and accuracy for classification task.
        """
        import lightgbm

        self.params = {
            "verbosity":
            -1,
            "bagging_freq":
            1,
            "max_depth":
            trial.suggest_int("max_depth", 3, 11),
            "learning_rate":
            trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves":
            trial.suggest_int("num_leaves", 2, 2**10),
            "subsample":
            trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree":
            trial.suggest_float("colsample_bytree", 0.05, 1.0),
            'lambda_l1':
            trial.suggest_float('lambda_l1', 1e-9, 10.0, log=True),
            'lambda_l2':
            trial.suggest_float('lambda_l2', 1e-9, 10.0, log=True),
            "min_data_in_leaf":
            trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        if self.task_type == TaskType.REGRESSION:
            if self.metric == Metric.RMSE:
                self.params["objective"] = "regression"
                self.params["metric"] = "rmse"
            elif self.metric == Metric.MAE:
                self.params["objective"] = "regression_l1"
                self.params["metric"] = "mae"
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.params["objective"] = "binary"
            if self.metric == Metric.ROCAUC:
                self.params["metric"] = "auc"
            elif self.metric == Metric.ACCURACY:
                self.params["metric"] = "binary_error"
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["objective"] = "multiclass"
            self.params["metric"] = "multi_error"
            self.params["num_class"] = self._num_classes or len(
                np.unique(tf_train.y.cpu().numpy()))
        else:
            raise ValueError(f"{self.__class__.__name__} is not supported for "
                             f"{self.task_type}.")

        train_x, train_y, cat_features = self._to_lightgbm_input(tf_train)
        eval_x, eval_y, _ = self._to_lightgbm_input(tf_val)
        train_data = lightgbm.Dataset(train_x, label=train_y)
        eval_data = lightgbm.Dataset(eval_x, label=eval_y)

        boost = lightgbm.train(
            self.params, train_data, num_boost_round=num_boost_round,
            categorical_feature=cat_features, valid_sets=[eval_data],
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=2000)
            ])
        pred = self._predict_helper(boost, eval_x)
        score = self.compute_metric(torch.from_numpy(eval_y),
                                    torch.from_numpy(pred))
        return score

    def _tune(
        self,
        tf_train: TensorFrame,
        tf_val: TensorFrame,
        num_trials: int,
        num_boost_round=2000,
    ):
        import lightgbm
        import optuna

        if self.task_type == TaskType.REGRESSION:
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: self.objective(trial, tf_train, tf_val,
                                         num_boost_round), num_trials)
        self.params.update(study.best_params)

        train_x, train_y, cat_features = self._to_lightgbm_input(tf_train)
        eval_x, eval_y, _ = self._to_lightgbm_input(tf_val)
        train_data = lightgbm.Dataset(train_x, label=train_y)
        eval_data = lightgbm.Dataset(eval_x, label=eval_y)
        self.model = lightgbm.train(
            self.params, train_data, num_boost_round=num_boost_round,
            categorical_feature=cat_features, valid_sets=[eval_data],
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=2000)
            ])

    def _predict(self, tf_test: TensorFrame) -> Tensor:
        device = tf_test.device
        test_x, _, _ = self._to_lightgbm_input(tf_test)
        pred = self._predict_helper(self.model, test_x)
        return torch.from_numpy(pred).to(device)
