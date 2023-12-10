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
    ) -> tuple[DataFrame, np.ndarray, list[int]]:
        r"""Convert :class:`TensorFrame` into LightGBM-compatible input format:
        :obj:`(feat, y, cat_features)`.

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.

        Returns:
            df (DataFrame): :obj:`DataFrame` that concatenates tensors of
                numerical and categorical features of the input
                :class:`TensorFrame`.
            y (numpy.ndarray, optional): Prediction label.
            cat_features (list[int]): Array containing indexes of
                categorical features.
        """
        tf = tf.cpu()
        y = tf.y
        if y is not None:
            y: np.ndarray = y.numpy()

        dfs: list[DataFrame] = []
        cat_features_list: list[np.ndarray] = []
        offset: int = 0

        if stype.categorical in tf.feat_dict:
            feat = tf.feat_dict[stype.categorical].numpy()
            arange = np.arange(offset, offset + feat.shape[1])
            dfs.append(pd.DataFrame(feat, columns=arange))
            cat_features_list.append(arange)
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
        cat_features: list[int] = np.concatenate(
            cat_features_list,
            axis=0).tolist() if len(cat_features_list) else []

        return df, y, cat_features

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
            pred (numpy.ndarray): The prediction output.
        """
        pred = model.predict(x)
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(axis=1)

        return pred

    def objective(
        self,
        trial: Any,  # optuna.trial.Trial
        train_data: Any,  # lightgbm.Dataset
        eval_data: Any,  # lightgbm.Dataset
        cat_features: list[int],
        num_boost_round: int,
    ) -> float:
        r"""Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            train_data (lightgbm.Dataset): Train data.
            eval_data (lightgbm.Dataset): Validation data.
            cat_features (list[int]): Array containing indexes of
                categorical features.
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
                np.unique(train_data.label))
        else:
            raise ValueError(f"{self.__class__.__name__} is not supported for "
                             f"{self.task_type}.")

        boost = lightgbm.train(
            self.params, train_data, num_boost_round=num_boost_round,
            categorical_feature=cat_features, valid_sets=[eval_data],
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=2000)
            ])
        pred = self._predict_helper(boost, eval_data.data)
        score = self.compute_metric(torch.from_numpy(eval_data.label),
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

        train_x, train_y, cat_features = self._to_lightgbm_input(tf_train)
        val_x, val_y, _ = self._to_lightgbm_input(tf_val)
        assert train_y is not None
        assert val_y is not None
        train_data = lightgbm.Dataset(train_x, label=train_y,
                                      free_raw_data=False)
        eval_data = lightgbm.Dataset(val_x, label=val_y, free_raw_data=False)

        study.optimize(
            lambda trial: self.objective(trial, train_data, eval_data,
                                         cat_features, num_boost_round),
            num_trials)
        self.params.update(study.best_params)

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

    def _load(self, path: str) -> None:
        import lightgbm

        self.model = lightgbm.Booster(model_file=path)
