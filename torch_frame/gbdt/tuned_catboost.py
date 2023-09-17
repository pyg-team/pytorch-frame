import copy

import catboost
import numpy as np
import optuna
import pandas as pd
import torch
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


class CatBoost(GBDT):
    def _to_catboost_input(self, tf):
        tf = tf.cpu()
        y = tf.y
        assert y is not None
        if stype.categorical in tf.x_dict and stype.numerical in tf.x_dict:
            x_cat = neg_to_nan(tf.x_dict[stype.categorical])
            categorical_df = pd.DataFrame(
                x_cat, columns=tf.col_names_dict[stype.categorical])
            numerical_df = pd.DataFrame(
                tf.x_dict[stype.numerical],
                columns=tf.col_names_dict[stype.numerical])
            df = pd.concat([categorical_df, numerical_df], axis=1)
            cat_features = np.arange(len(x_cat))
        elif stype.categorical in tf.x_dict:
            x_cat = neg_to_nan(tf.x_dict[stype.categorical])
            df = pd.DataFrame(x_cat,
                              columns=tf.col_names_dict[stype.categorical])
            cat_features = np.arange(len(x_cat))
        elif stype.numerical in tf.x_dict:
            df = pd.DataFrame(x_cat,
                              columns=tf.col_names_dict[stype.numerical])
        else:
            raise ValueError("The input TensorFrame object is empty.")
        return df, y.numpy(), cat_features

    def objective(self, trial, tf_train: TensorFrame, tf_val: TensorFrame):
        self.params = {
            "depth":
            trial.suggest_int("depth", 3, 11),
            "boosting_type":
            trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type":
            trial.suggest_categorical("bootstrap_type",
                                      ["Bayesian", "Bernoulli", "MVS"]),
            "colsample_bylevel":
            trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "leaf_estimation_iterations":
            trial.suggest_int("leaf_estimation_iterations", 1, 11),
            "l2_leaf_reg":
            trial.suggest_float("l2_leaf_reg", 1, 11, log=True)
        }
        if self.task_type == TaskType.REGRESSION:
            self.params["objective"] = trial.suggest_categorical(
                "objective", ["RMSE", "MAE", "MAPE"])
        else:
            self.params["objective"] = trial.suggest_categorical(
                "objective", ["CrossEntropy", "AUC"])
        if self.params["bootstrap_type"] == "Bayesian":
            self.params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 1)
        elif self.params["bootstrap_type"] == "Bernoulli":
            self.params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        train_x, train_y, cat_features = self._to_catboost_input(tf_train)
        eval_x, eval_y, _ = self._to_catboost_input(tf_val)
        boost = catboost.CatBoost(self.params)
        print("cat features are ", cat_features)
        boost = boost.fit(train_x, train_y, cat_features=cat_features,
                          eval_set=[(eval_x, eval_y)],
                          early_stopping_rounds=50, verbose_eval=True)
        pred = boost.predict(eval_x)
        score = self.compute_metric(torch.from_numpy(train_y),
                                    torch.from_numpy(pred))
        return score

    def _tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
              num_trials: int):
        if self.task_type == TaskType.REGRESSION:
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, tf_train, tf_val),
                       num_trials)
        self.params.update(study.best_params)
        train_x, train_y, cat_features = self._to_catboost_input(tf_train)
        eval_x, eval_y, _ = self._to_catboost_input(tf_val)
        self.model = catboost.train(self.params, train_x, train_y,
                                    cat_features=cat_features, eval_set=[
                                        (eval_x, eval_y)
                                    ], early_stopping_rounds=50, verbose=False)

    def _predict(self, tf_test: TensorFrame) -> Tensor:
        device = tf_test.device
        test_x, test_y, cat_features = self._to_catboost_input(tf_test)
        pred = self.model.predict(test_x)
        return torch.from_numpy(pred).to(device)
