import numpy as np
import optuna
import torch
import xgboost
from torch import Tensor

from torch_frame import TaskType, TensorFrame
from torch_frame.gbdt import GBDT


class XGBoost(GBDT):
    r"""An XGBoost model implementation with hyper-parameter tuning using
        Optuna.

    This implementation extends GradientBoostingDecisionTrees and aims to find
    optimal hyperparameters by optimizing the given objective function.
    """
    def objective(self, trial: optuna.trial.Trial, dtrain: xgboost.DMatrix,
                  dvalid: xgboost.DMatrix, num_boost_round: int) -> float:
        r""" Objective function to be optimized.

        Args:
            trial (Trial): Optuna trial.
            dtrain (xgboost.DMatrix): Train data.
            dvalid (xgboost.DMatrix): Validation data.
            num_boost_round (int): Number of boosting round.

        Returns:
            score (float): Best objective value. Mean squared error for
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
        tf_train = tf_train.cpu()
        tf_val = tf_val.cpu()
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
        tf_test = tf_test.cpu()
        test_x, test_y, test_ft = self._to_xgboost_input(tf_test)
        dtest = xgboost.DMatrix(test_x, label=test_y, feature_types=test_ft,
                                enable_categorical=True)
        pred = self.model.predict(dtest)
        return torch.from_numpy(pred).to(device)
