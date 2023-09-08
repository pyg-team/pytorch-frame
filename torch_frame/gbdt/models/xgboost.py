import numpy as np
import optuna
import sklearn.metrics
import torch
import xgboost

from torch_frame import TensorFrame, stype
from torch_frame.gbdt import GradientBoostingDecisionTree


class XGBoost(GradientBoostingDecisionTree):
    def __init__(self, task_type):
        self.xgb = xgboost()

    def objective(self, trial):
        params = {
            "booster":
            "gbtree",
            "n_estimators":
            4096,
            "early_stopping_rounds":
            50,
            "max_depth":
            trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":
            trial.suggest_loguniform("min_child_weight", 1e-8, 1e5),
            "subsample":
            trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":
            trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel":
            trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "gamma":
            (0.0 if not trial.suggest_categorical('use_gamma', [True, False])
             else trial.suggest_loguniform('gamma', 1e-8, 1e2)),
            "lambda":
            (0.0 if not trial.suggest_categorical('use_lambda', [True, False])
             else trial.suggest_loguniform('lambda', 1e-8, 1e2)),
            "alpha":
            (0.0 if not trial.suggest_categorical('use_alpha', [True, False])
             else trial.suggest_loguniform('alpha', 1e-8, 1e2)),
            "eta":
            trial.suggest_loguniform('learning_rate', 1e-5, 1.0)
        }
        train_x = torch.cat(self.tf_train.x_dict[stype.numerical],
                            self.tf_train.x_dict[stype.categorical],
                            dim=1).cpu().numpy()
        train_y = self.tf_train.y.cpu().numpy()
        val_x = torch.cat(self.tf_val.x_dict[stype.numerical],
                          self.tf_val.x_dict[stype.categorical],
                          dim=1).cpu().numpy()
        val_y = self.tf_val.y.cpu().numpy()
        bst = xgboost.train(params, train_x, train_y)
        preds = bst.predict(val_x)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(val_y, pred_labels)
        return accuracy

    def fit_tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
                 num_trials: int):
        self.tf_train = tf_train
        self.tf_val = tf_val
        study = optuna.create_study(direction="maximize")
        study.optmize(self.objective, num_trials)
        trial = study.best_trial
        self.best_params = trial.params
        train_x = torch.cat(self.tf_train.x_dict[stype.numerical],
                            self.tf_train.x_dict[stype.categorical],
                            dim=1).cpu().numpy()
        train_y = self.tf_train.y.cpu().numpy()
        self.model = xgboost.train(self.best_params, train_x, train_y)

    def predict(self, tf_test: TensorFrame):
        test_x = torch.cat(tf_test.x_dict[stype.numerical],
                           tf_test.x_dict[stype.categorical],
                           dim=1).cpu().numpy()
        test_y = tf_test.y.cpu().numpy()
        preds = self.model.predict(test_x)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        return accuracy
