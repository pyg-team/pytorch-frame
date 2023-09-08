import argparse
import os.path as osp
import random

import numpy as np
import torch

from torch_frame.datasets import Yandex
from torch_frame.gbdt import ExtensivelyTunedXGBoost

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='jannis')
parser.add_argument('--channels', type=int, default=192)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = (torch.device('cuda')
          if torch.cuda.is_available() else torch.device('cpu'))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Yandex(root=path, name=args.dataset)
dataset.materialize()

train_dataset = dataset.get_split_dataset('train')
val_dataset = dataset.get_split_dataset('val')
test_dataset = dataset.get_split_dataset('test')
XGB = ExtensivelyTunedXGBoost()
XGB.fit_tune(tf_train=train_dataset.tensor_frame,
             tf_val=val_dataset.tensor_frame, num_trials=500)
test_acc = XGB.predict(tf_test=test_dataset.tensor_frame)
print(f'Test acc: {test_acc}')
