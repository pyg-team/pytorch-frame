"""
Reported (reproduced) results of Tuned XGBoost on TabularBenchmark of
the Trompt paper: https://arxiv.org/abs/2305.18446
electricity (A4): 87.86 (91.09)
eye_movements (A5): 64.77 (64.21)
MagicTelescope (B2): 85.31 (86.50)
bank-marketing (B4): 79.81 (80.41)
california (B5): 90.30 (89.71)
credit (B7): 75.91 (77.4)
pol (B14): 98.35 (98.25)
jannis (mathcal B4): 78.28 (78.27)
"""
import argparse
import os.path as osp
import random

import numpy as np
import torch

from torch_frame import TaskType
from torch_frame.datasets import TabularBenchmark
from torch_frame.gbdt import XGBoost

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='california')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = (torch.device('cuda')
          if torch.cuda.is_available() else torch.device('cpu'))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = TabularBenchmark(root=path, name=args.dataset)
dataset.materialize()
dataset = dataset.shuffle()
# Split ratio following https://arxiv.org/abs/2207.08815
# 70% is used for training. 30% of the remaining is used for validation.
# The final reminder is used for testing.
train_dataset, val_dataset, test_dataset = dataset[:0.7], dataset[
    0.7:0.79], dataset[0.79:]

xgb = XGBoost(task_type=TaskType.MULTICLASS_CLASSIFICATION)
xgb.tune(tf_train=train_dataset.tensor_frame, tf_val=val_dataset.tensor_frame,
         num_trials=10)
pred = xgb.predict(tf_test=test_dataset.tensor_frame)
score = xgb.compute_metric(test_dataset.tensor_frame.y, pred)
print(f'Test acc: {score}')
