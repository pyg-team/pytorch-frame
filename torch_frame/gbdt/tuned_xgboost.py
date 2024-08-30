from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import Tensor

from torch_frame import Metric, TaskType, TensorFrame, stype
from torch_frame.gbdt import GBDT


def neg_to_nan(x: Tensor) -> Tensor:
    r"""Convert -1 category back to NaN that can be handled by GBDT.

    Args:
        x (Tensor): Input categ. feature, where `-1` represents `NaN`.

    Returns:
        x (Tensor): Output categ. feature, where `-1` is replaced with `NaN`
    """
    is_neg = x == -1
    if is_neg.any():
        x = copy.copy(x).to(torch.float32)
        x[is_neg] = torch.nan
    return x


class XGBoost(GBDT):
    r"""An XGBoost model implementation with hyper-parameter tuning using
    Optuna.

    This implementation extends GBDT and aims to find optimal hyperparameters
    by optimizing the given objective function.
    """
    def _to_xgboost_input(
        self,
        tf: TensorFrame,
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        r"""Convert :class:`TensorFrame` into XGBoost-compatible input format:
        :obj:`(feat, y, feat_types)`.

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.

        Returns:
            feat (numpy.ndarray): Output :obj:`numpy.ndarray` by
                concatenating tensors of numerical and categorical features of
                the input :class:`TensorFrame`.
            y (numpy.ndarray, optional): Prediction target.
            feature_types (List[str]): List of feature types: "q" for numerical
                features and "c" for categorical features. The abbreviation
                aligns with xgboost tutorial.
                <https://github.com/dmlc/xgboost/blob/master/doc/
                tutorials/categorical.rst#using-native-interface>
        """
        tf = tf.cpu()
        y = tf.y
        if y is not None:
            y: np.ndarray = y.numpy()

        feats: list[Tensor] = []
        types: list[str] = []

        if stype.categorical in tf.feat_dict:
            feats.append(neg_to_nan(tf.feat_dict[stype.categorical]))
            types.extend(['c'] * len(tf.col_names_dict[stype.categorical]))

        if stype.numerical in tf.feat_dict:
            feats.append(tf.feat_dict[stype.numerical])
            types.extend(['q'] * len(tf.col_names_dict[stype.numerical]))

        if stype.embedding in tf.feat_dict:
            feat = tf.feat_dict[stype.embedding]
            feat = feat.values
            feat = feat.view(feat.size(0), -1)
            feats.append(feat)
            types.extend(['q'] * feat.size(-1))

        # TODO Add support for other stypes.

        if len(feats) == 0:
            raise ValueError("The input TensorFrame object is empty.")

        feat = torch.cat(feats, dim=-1).numpy()

        return feat, y, types

    def objective(
        self,
        trial: Any,  # optuna.trial.Trial
        dtrain: Any,  # xgboost.DMatrix
        dvalid: Any,  # xgboost.DMatrix
        num_boost_round: int,
        early_stopping_rounds: int,
    ) -> float:
        r"""Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            dtrain (xgboost.DMatrix): Train data.
            dvalid (xgboost.DMatrix): Validation data.
            num_boost_round (int): Number of boosting round.
            early_stopping_rounds (int): Number of early stopping
                rounds.

        Returns:
            float: Best objective value. Root mean squared error for
            regression task and accuracy for classification task.
        """
        import optuna
        import xgboost

        self.params = {
            "booster":
            trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda":
            (0.0 if not trial.suggest_categorical('use_lambda', [True, False])
             else trial.suggest_float('lambda', 1e-8, 1e2, log=True)),
            "alpha":
            (0.0 if not trial.suggest_categorical('use_alpha', [True, False])
             else trial.suggest_float('alpha', 1e-8, 1e2, log=True))
        }
        if self.params["booster"] == "gbtree" or self.params[
                "booster"] == "dart":
            self.params["max_depth"] = trial.suggest_int("max_depth", 3, 11)
            self.params["min_child_weight"] = trial.suggest_float(
                "min_child_weight", 1e-8, 1e5, log=True)
            self.params["subsample"] = trial.suggest_float(
                "subsample", 0.5, 1.0)
            self.params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0)
            self.params["colsample_bylevel"] = trial.suggest_float(
                "colsample_bylevel", 0.5, 1.0)
            self.params["gamma"] = (0.0 if not trial.suggest_categorical(
                'use_gamma', [True, False]) else trial.suggest_float(
                    'gamma', 1e-8, 1e2, log=True))
            self.params["eta"] = trial.suggest_float('learning_rate', 1e-6,
                                                     1.0, log=True)
        if self.params["booster"] == "dart":
            self.params["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"])
            self.params["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"])
            self.params["rate_drop"] = trial.suggest_float(
                "rate_drop", 1e-8, 1.0, log=True)
            self.params["skip_drop"] = trial.suggest_float(
                "skip_drop", 1e-8, 1.0, log=True)

        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["objective"] = "multi:softmax"
            self.params["eval_metric"] = "merror"
        elif self.task_type == TaskType.REGRESSION:
            if self.metric == Metric.RMSE:
                self.params["objective"] = "reg:squarederror"
                self.params["eval_metric"] = "rmse"
            elif self.metric == Metric.MAE:
                self.params["objective"] = "reg:absoluteerror"
                self.params["eval_metric"] = "mae"
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.params["objective"] = "binary:logistic"
            if self.metric == Metric.ROCAUC:
                self.params["eval_metric"] = "auc"
            elif self.metric == Metric.ACCURACY:
                self.params["eval_metric"] = "error"
        else:
            raise ValueError(f"{self.__class__.__name__} is not supported for "
                             f"{self.task_type}.")

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, f"validation-{self.params['eval_metric']}")
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["num_class"] = self._num_classes or len(
                np.unique(dtrain.get_label()))

        boost = xgboost.train(self.params, dtrain,
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=False, evals=[
                                  (dvalid, 'validation')
                              ], callbacks=[pruning_callback])
        if boost.best_iteration:
            iteration_range = (0, boost.best_iteration + 1)
        else:
            iteration_range = None
        pred = boost.predict(dvalid, iteration_range)

        # If xgboost early stops on multiclass classification
        # task, then the output shape would be (batch_size, num_classes).
        # We need to take argmax to get the final prediction output.
        if (boost.best_iteration
                and self.task_type == TaskType.MULTICLASS_CLASSIFICATION):
            assert pred.shape[1] == self.params["num_class"]
            pred = torch.argmax(torch.from_numpy(pred), dim=1)
        else:
            pred = torch.from_numpy(pred)
        score = self.compute_metric(torch.from_numpy(dvalid.get_label()), pred)
        return score

    def _tune(
        self,
        tf_train: TensorFrame,
        tf_val: TensorFrame,
        num_trials: int,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
    ):
        import optuna
        import xgboost

        if self.task_type == TaskType.REGRESSION:
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="maximize")
        train_feat, train_y, train_feat_type = self._to_xgboost_input(tf_train)
        val_feat, val_y, val_feat_type = self._to_xgboost_input(tf_val)
        assert train_y is not None
        assert val_y is not None
        dtrain = xgboost.DMatrix(train_feat, label=train_y,
                                 feature_types=train_feat_type,
                                 enable_categorical=True)
        dvalid = xgboost.DMatrix(val_feat, label=val_y,
                                 feature_types=val_feat_type,
                                 enable_categorical=True)
        study.optimize(
            lambda trial: self.objective(
                trial, dtrain, dvalid, num_boost_round, early_stopping_rounds),
            num_trials)
        self.params.update(study.best_params)

        self.model = xgboost.train(self.params, dtrain,
                                   num_boost_round=num_boost_round,
                                   early_stopping_rounds=early_stopping_rounds,
                                   verbose_eval=False,
                                   evals=[(dvalid, 'validation')])

    def _predict(self, tf_test: TensorFrame) -> Tensor:
        import xgboost

        device = tf_test.device
        test_feat, test_y, test_feat_type = self._to_xgboost_input(tf_test)
        dtest = xgboost.DMatrix(test_feat, label=test_y,
                                feature_types=test_feat_type,
                                enable_categorical=True)
        if self.model.best_iteration is not None:
            iteration_range = self.model.best_iteration
        else:
            iteration_range = None
        pred = self.model.predict(dtest, iteration_range)

        # If xgboost early stops on multiclass classification
        # task, then the output shape would be (batch_size, num_classes).
        # We need to take argmax to get the final prediction output.
        if (self.model.best_iteration
                and self.task_type == TaskType.MULTICLASS_CLASSIFICATION):
            assert pred.shape[1] == self._num_classes
            pred = torch.argmax(torch.from_numpy(pred), dim=1)
        else:
            pred = torch.from_numpy(pred)
        return pred.to(device)

    def _load(self, path: str) -> None:
        import xgboost

        self.model = xgboost.Booster(model_file=path)
