"""
Reported (reproduced) results of Tuned XGBoost on TabularBenchmark of
the Trompt paper: https://arxiv.org/abs/2305.18446
electricity (A4): 88.52 (91.09)
eye_movements (A5): 66.57 (64.21)
MagicTelescope (B2): 86.05 (86.50)
bank-marketing (B4): 80.34 (80.41)
california (B5): 90.12 (89.71)
credit (B7): 77.26 (77.4)
pol (B14): 98.09 (97.5)
jannis (mathcal B4): 79.67 (77.81)
"""
import argparse
import os.path as osp
import random

import numpy as np
import torch

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

if dataset.task_type.is_classification:
    num_classes = dataset.num_classes
else:
    num_classes = None
xgb = XGBoost(task_type=dataset.task_type, num_classes=num_classes)
xgb.tune(tf_train=train_dataset.tensor_frame, tf_val=val_dataset.tensor_frame,
         num_trials=10)
pred = xgb.predict(tf_test=test_dataset.tensor_frame)
score = xgb.compute_metric(test_dataset.tensor_frame.y, pred)
print(score)
