import numpy as np
import optuna
import sklearn.metrics
import torch
import xgboost

from torch_frame import TensorFrame, stype
from torch_frame.gbdt import GradientBoostingDecisionTree


def accuracy_score(predicted_labels: np.ndarray, dtrain: xgboost.DMatrix):
    y = dtrain.get_label()
    return "acc", float(np.mean(y == predicted_labels))


class ExtensivelyTunedXGBoost(GradientBoostingDecisionTree):
    def __init__(self, task_type='multiclass_classification'):
        if task_type == 'multiclass_classification':
            self.obj = "multi:softmax"
            self.eval_metric = "mlogloss"
        self.param = {"objective": self.obj, "eval_metric": self.eval_metric}

    def objective(self, trial):
        params = {
            "booster":
            "gbtree",
            "num_class":
            4,
            "objective":
            self.obj,
            "eval_metric":
            self.eval_metric,
            "max_depth":
            trial.suggest_int("max_depth", 3, 10),
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
            trial.suggest_float('learning_rate', 1e-5, 1.0, log=True)
        }
        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, f"validation-{self.eval_metric}")

        train_x = self._tensor_frame_to_numpy(self.tf_train)
        train_y = self.tf_train.y.cpu().numpy()
        val_x = self._tensor_frame_to_numpy(self.tf_val)
        val_y = self.tf_val.y.cpu().numpy()
        dtrain = xgboost.DMatrix(train_x, label=train_y)
        dvalid = xgboost.DMatrix(val_x, label=val_y)
        bst = xgboost.train(params, dtrain, evals=[(dvalid, 'validation')],
                            callbacks=[pruning_callback], num_boost_round=4096,
                            early_stopping_rounds=50)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(val_y, pred_labels)
        return accuracy

    def _tensor_frame_to_numpy(self, tf: TensorFrame):
        if stype.categorical in tf.x_dict and stype.numerical in tf.x_dict:
            return torch.cat(
                (tf.x_dict[stype.numerical], tf.x_dict[stype.categorical]),
                dim=1).cpu().numpy()
        elif stype.categorical in tf.x_dict:
            return tf.x_dict[stype.categorical].cpu().numpy()
        elif stype.numerical in tf.x_dict:
            return tf.x_dict[stype.numerical].cpu().numpy()
        else:
            raise ValueError("The input TensorFrame is empty.")

    def fit_tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
                 num_trials: int):
        self.tf_train = tf_train
        self.tf_val = tf_val
        self.num_class = torch.unique(tf_train.y).size(0)
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, num_trials)
        param = {
            "objective": self.obj,
            "eval_metric": self.eval_metric,
        }
        param.update(study.best_params)
        train_x = self._tensor_frame_to_numpy(self.tf_train)
        train_y = self.tf_train.y.cpu().numpy()
        val_x = self._tensor_frame_to_numpy(self.tf_val)
        val_y = self.tf_val.y.cpu().numpy()
        dvalid = xgboost.DMatrix(val_x, label=val_y)
        dtrain = xgboost.DMatrix(train_x, label=train_y)
        self.model = xgboost.train(param, dtrain, evals=[
            (dvalid, 'validation')
        ], num_boost_round=4096, early_stopping_rounds=50)

    def predict(self, tf_test: TensorFrame):
        test_x = self._tensor_frame_to_numpy(tf_test)
        test_y = tf_test.y.cpu().numpy()
        dtest = xgboost.DMatrix(test_x, label=test_y)
        preds = self.model.predict(dtest)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        return accuracy
