import catboost as cb
import numpy as np
import optuna

from torch_frame.gbdt import GBDT


class CatBoost(GBDT):
    def __init__(self):
        pass

    def objective(self, trial):
        self.params = {
            "objective":
            self.obj,
            "eval_metric":
            self.eval_metric,
            "max_depth":
            trial.suggest_int("max_depth", 3, 11),
            "boosting_type":
            trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type":
            trial.suggest_categorical("bootstrap_type",
                                      ["Bayesian", "Bernoulli", "MVS"]),
            "leaf_estimation_iterations":
            trial.suggest_int("leaf_estimation_iterations", 1, 11),
            "l2_leaf_reg":
            trial.suggest_float("l2_leaf_reg", 1, 11, log=True)
        }
        if self.params["bootstrap_type"] == "Bayesian":
            self.params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 1)
        elif self.params["bootstrap_type"] == "Bernoulli":
            self.params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
