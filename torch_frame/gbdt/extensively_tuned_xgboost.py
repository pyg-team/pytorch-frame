import numpy as np
import optuna
import xgboost
from sklearn.metrics import accuracy_score, mean_squared_error

from torch_frame import TaskType, TensorFrame
from torch_frame.gbdt import GradientBoostingDecisionTrees


class ExtensivelyTunedXGBoost(GradientBoostingDecisionTrees):
    def objective(self, trial: optuna.trial.Trial, tf_train: TensorFrame,
                  tf_val: TensorFrame):
        r""" Objective function to be maximized.

        Args:
            trial (Trial): Optuna trial
            tf_train (TensorFrame): Train data
            tf_val (TensorFrame): Validation data

        Returns:
            score (float): Best objective value. Negative root
                mean squared error for regression task and negative
                root mean squared error for classification task.
        """
        self.params = {
            "objective":
            self.obj,
            "eval_metric":
            self.eval_metric,
            "booster":
            "gbtree",
            "max_depth":
            trial.suggest_int("max_depth", 3, 11),
            "min_child_weight":
            trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
            "subsample":
            trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":
            trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel":
            trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "gamma":
            (0.0 if not trial.suggest_categorical('use_gamma', [True, False])
             else trial.suggest_float('gamma', 1e-8, 1e2, log=True)),
            "lambda":
            (0.0 if not trial.suggest_categorical('use_lambda', [True, False])
             else trial.suggest_float('lambda', 1e-8, 1e2, log=True)),
            "alpha":
            (0.0 if not trial.suggest_categorical('use_alpha', [True, False])
             else trial.suggest_float('alpha', 1e-8, 1e2, log=True)),
            "eta":
            trial.suggest_float('learning_rate', 1e-6, 1.0, log=True)
        }
        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, f"validation-{self.eval_metric}")
        train_x = self._tensor_frame_to_numpy(tf_train)
        train_y = tf_train.y.cpu().numpy()
        val_x = self._tensor_frame_to_numpy(tf_val)
        val_y = tf_val.y.cpu().numpy()
        dtrain = xgboost.DMatrix(train_x, label=train_y)
        dvalid = xgboost.DMatrix(val_x, label=val_y)
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["num_class"] = len(np.unique(train_y))
        boost = xgboost.train(self.params, dtrain, num_boost_round=4096,
                              early_stopping_rounds=50, verbose_eval=False,
                              evals=[(dvalid, 'validation')],
                              callbacks=[pruning_callback])
        preds = boost.predict(dvalid)
        if self.task_type == TaskType.REGRESSION:
            score = -mean_squared_error(val_y, preds, squared=False)
        else:
            score = accuracy_score(val_y, preds)
        return score

    def _fit_tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
                  num_trials: int):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, tf_train, tf_val),
                       num_trials)
        self.params.update(study.best_params)

        train_x = self._tensor_frame_to_numpy(tf_train)
        train_y = tf_train.y.cpu().numpy()
        val_x = self._tensor_frame_to_numpy(tf_val)
        val_y = tf_val.y.cpu().numpy()
        dvalid = xgboost.DMatrix(val_x, label=val_y)
        dtrain = xgboost.DMatrix(train_x, label=train_y)
        self.model = xgboost.train(self.params, dtrain, evals=[
            (dvalid, 'validation')
        ], num_boost_round=20, early_stopping_rounds=50)

    def _eval(self, tf_test: TensorFrame):
        test_x = self._tensor_frame_to_numpy(tf_test)
        test_y = tf_test.y.cpu().numpy()
        dtest = xgboost.DMatrix(test_x, label=test_y)
        preds = self.model.predict(dtest)
        if self.task_type == TaskType.REGRESSION:
            metric_score = -mean_squared_error(test_y, preds, squared=False)
        else:
            metric_score = accuracy_score(test_y, preds)
        return metric_score
