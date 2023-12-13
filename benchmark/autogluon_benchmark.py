# First install package from terminal:
# pip install -U pip
# pip install -U setuptools wheel
# pip install autogluon  # autogluon==1.0.0

import argparse
import os.path as osp

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from torch_frame.datasets.data_frame_benchmark import (
    SPLIT_COL,
    DataFrameBenchmark,
)
from torch_frame.typing import TaskType

DEFAULT_METRIC = {
    'regression': "rmse",
    'binary_classification': "roc_auc",
    'multiclass_classification': "accuracy",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


def compute_metric(pred, target):
    metric = DEFAULT_METRIC[args.task_type]
    if metric == "rmse":
        score = (pred - target).square().mean().sqrt().item()
    elif metric == "MAE":
        score = (pred - target).abs().mean().item()
    elif metric == "roc_auc":
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(target, pred)
    elif metric == "accuracy":
        total_correct = (target == pred).sum().item()
        test_size = len(target)
        score = total_correct / test_size
    else:
        raise ValueError(f'{metric} is not supported.')
    return score


# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                             scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()
train_dataset_df = train_dataset.df.drop(SPLIT_COL, axis=1)
val_dataset_df = val_dataset.df.drop(SPLIT_COL, axis=1)
test_dataset_df = test_dataset.df.drop(SPLIT_COL, axis=1)
train_data = TabularDataset(data=train_dataset_df)
val_data = TabularDataset(data=val_dataset_df)
test_data = TabularDataset(data=test_dataset_df)
predictor = TabularPredictor(label=dataset.target_col,
                             eval_metric='roc_auc').fit(
                                 train_data=train_data, tuning_data=val_data,
                                 presets='high_quality', use_bag_holdout=True,
                                 num_gpus=1)
y_pred = predictor.predict(test_dataset_df.drop(columns=[dataset.target_col]))
print(predictor.leaderboard(test_data))
print(predictor.evaluate(test_data))
