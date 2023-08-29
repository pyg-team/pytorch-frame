import argparse
import os.path as osp

import torch

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import TabularBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
    Trompt,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='california')
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Prepare data and loaders
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = TabularBenchmark(root=path, name=args.dataset)
dataset = dataset.materialize()
dataset.shuffle()
# Split ratio following https://arxiv.org/abs/2207.08815
train_loader = DataLoader(dataset[:0.7])
val_loader = DataLoader(dataset[0.7:0.79])
test_loader = DataLoader(dataset[0.79:])
num_classes = int(dataset.tensor_frame.y.max() + 1)

# Initialize encoder and model
encoder = StypeWiseFeatureEncoder(
    out_channels=args.channels,
    col_stats=dataset.col_stats,
    col_names_dict=dataset.tensor_frame.col_names_dict,
    stype_encoder_dict={
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
    },
).to(device)

model = Trompt(
    in_channels=args.channels,
    out_channels=num_classes,
    num_cols=dataset.tensor_frame.num_cols,
)
