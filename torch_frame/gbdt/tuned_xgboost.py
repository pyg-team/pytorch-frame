import copy
from typing import List, Tuple

import numpy as np
import optuna
import torch
import xgboost
from torch import Tensor

from torch_frame import TaskType, TensorFrame, stype
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

    This implementation extends GradientBoostingDecisionTrees and aims to find
    optimal hyperparameters by optimizing the given objective function.
    """
    def _to_xgboost_input(
        self,
        tf: TensorFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        r"""Convert :obj:`TensorFrame` into XGBoost-compatible input format:
        :obj:`(test_x, test_y, feat_types)`.

        Args:
            tf (Tensor Frame): Input :obj:TensorFrame object.
        Returns:
            test_x (numpy.ndarray): Output :obj:`numpy.ndarray` by
                concatenating tensors of numerical and categorical features of
                the input :obj:`TensorFrame`.
            test_y (numpy.ndarray): Prediction target :obj:`numpy.ndarray`.
            feature_types (List[str]): List of feature types: "q" for numerical
                features and "c" for categorical features. The abbreviation
                aligns with xgboost tutorial.
                <https://github.com/dmlc/xgboost/blob/master/doc/
                tutorials/categorical.rst#using-native-interface>
        """
        tf = tf.cpu()
        test_y = tf.y
        assert test_y is not None
        if stype.categorical in tf.x_dict and stype.numerical in tf.x_dict:
            x_cat = neg_to_nan(tf.x_dict[stype.categorical])
            test_x = torch.cat([tf.x_dict[stype.numerical], x_cat], dim=1)
            feature_types = ["q"] * len(tf.col_names_dict[stype.numerical]) + [
                "c"
            ] * len(tf.col_names_dict[stype.categorical])
        elif stype.categorical in tf.x_dict:
            test_x = neg_to_nan(tf.x_dict[stype.categorical])
            feature_types = ["c"] * len(tf.col_names_dict[stype.categorical])
        elif stype.numerical in tf.x_dict:
            test_x = tf.x_dict[stype.numerical]
            feature_types = ["q"] * len(tf.col_names_dict[stype.numerical])
        else:
            raise ValueError("The input TensorFrame object is empty.")
        return test_x.numpy(), test_y.numpy(), feature_types

    def objective(self, trial: optuna.trial.Trial, dtrain: xgboost.DMatrix,
                  dvalid: xgboost.DMatrix, num_boost_round: int) -> float:
        r""" Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            dtrain (xgboost.DMatrix): Train data.
            dvalid (xgboost.DMatrix): Validation data.
            num_boost_round (int): Number of boosting round.

        Returns:
            score (float): Best objective value. Root mean squared error for
                regression task and accuracy for classification task.
        """
        self.params = {
            "objective":
            self.obj,
            "eval_metric":
            self.eval_metric,
            "booster":
            trial.suggest_categorical("booster",
                                      ["gbtree", "gblinear", "dart"]),
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

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, f"validation-{self.eval_metric}")
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["num_class"] = len(np.unique(dtrain.get_label()))
        boost = xgboost.train(self.params, dtrain,
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=50, verbose_eval=False,
                              evals=[(dvalid, 'validation')],
                              callbacks=[pruning_callback])
        pred = boost.predict(dvalid)
        score = self.compute_metric(torch.from_numpy(dvalid.get_label()),
                                    torch.from_numpy(pred))
        return score

    def _tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
              num_trials: int, num_boost_round: int = 2000):
        if self.task_type == TaskType.REGRESSION:
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="maximize")
        train_x, train_y, train_ft = self._to_xgboost_input(tf_train)
        val_x, val_y, val_ft = self._to_xgboost_input(tf_val)
        dtrain = xgboost.DMatrix(train_x, label=train_y,
                                 feature_types=train_ft,
                                 enable_categorical=True)
        dvalid = xgboost.DMatrix(val_x, label=val_y, feature_types=val_ft,
                                 enable_categorical=True)
        study.optimize(
            lambda trial: self.objective(trial, dtrain, dvalid, num_boost_round
                                         ), num_trials)
        self.params.update(study.best_params)

        self.model = xgboost.train(self.params, dtrain,
                                   num_boost_round=num_boost_round,
                                   early_stopping_rounds=50,
                                   verbose_eval=False,
                                   evals=[(dvalid, 'validation')])

    def _predict(self, tf_test: TensorFrame) -> Tensor:
        device = tf_test.device
        test_x, test_y, test_ft = self._to_xgboost_input(tf_test)
        dtest = xgboost.DMatrix(test_x, label=test_y, feature_types=test_ft,
                                enable_categorical=True)
        pred = self.model.predict(dtest)
        return torch.from_numpy(pred).to(device)
