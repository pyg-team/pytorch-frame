"""
This script gives an example of how to use torch_frame to design tabular deep
learning models in a modular way.
"""

import argparse
import math
import os.path as osp
import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList
from tqdm import tqdm

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader
from torch_frame.data.stats import StatType
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    Decoder,
    EmbeddingEncoder,
    LinearBucketEncoder,
    StypeWiseFeatureEncoder,
    TableConv,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = (torch.device('cuda')
          if torch.cuda.is_available() else torch.device('cpu'))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Prepare dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Yandex(root=path, name=args.dataset)
# Materialize the dataset, i.e., get tensor frame as its attribute.
dataset.materialize()

# Get pre-defined split
train_dataset = dataset.get_split_dataset('train')
val_dataset = dataset.get_split_dataset('val')
test_dataset = dataset.get_split_dataset('test')

# Set up tensor frames (DataFrame compatible to Pytorch)
# TensorFrame(
#   num_cols=14,
#   num_rows=26048,
#   categorical (8): ['C_feature_0', 'C_feature_1', 'C_feature_2',
#      'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6',
#      'C_feature_7'],
#   numerical (6): ['N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3',
#                   'N_feature_4', 'N_feature_5'],
#   has_target=True,
#   device=cpu,
# )
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)

# Set up data loaders for tensor frames
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)


# Custom table convolution
class SelfAttentionConv(TableConv):
    r"""Simple self-attention-based table covolution to modle interaction
    between different columns.

    Args:
        channels (int): Hidden channel dimensionality
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Linear functions for modeling key/query/value in self-attention.
        self.lin_k = Linear(channels, channels)
        self.lin_q = Linear(channels, channels)
        self.lin_v = Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        r"""Convolves input tensor to model interaction between different cols.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_cols, channels]

        Returns:
            out (Tensor): Output tensor of shape
                [batch_size, num_cols, channels]
        """
        # [batch_size, num_cols, channels]
        x_key = self.lin_k(x)
        x_query = self.lin_q(x)
        x_value = self.lin_v(x)
        # [batch_size, num_cols, num_cols]
        prod = x_query.bmm(x_key.transpose(2, 1)) / math.sqrt(self.channels)
        # Attention weights between all pairs of columns.
        # Shape: [batch_size, num_cols, num_cols]
        attn = F.softmax(prod, dim=-1)
        # Mix x_value based on the attention weights
        out = attn.bmm(x_value)
        # [batch_size, num_cols, num_channels]
        return out


# Custom decoder
class MeanDecoder(Decoder):
    r"""Simple decoder that mean-pools over the embeddings of all columns and
    apply a linear transformation to map the pooled embeddings to desired
    dimensionality.

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channel dimensionality
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Linear function to map pooled embeddings into desired dimensionality
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Mean pooling over the column dimension
        # [batch_size, num_cols, in_channels] -> [batch_size, in_channels]
        out = torch.mean(x, dim=1)
        # [batch_size, out_channels]
        return self.lin(out)


# Custom model
class TabularNN(Module):
    r"""The overall tabular NN model that takes in tensor frame as input and
    outputs row embeddings. It is a combination of
    (1) Feature encoder (self.encoder): Mapping tensor frame into 3-dimensional
        :obj:`x` of shape [batch_size, num_cols, channels]
    (2) Table covolutions (self.convs): Iteratively transforming the
        3-dimensional :obj:`x`
    (3) Decoder (self.decoder): Maps the transformed 3-dimensional x into
        2-dimensional :obj:`out` of shape [batch_size, out_channels].
        Each element of :obj:`out` represents the row embedding of the original
        tensor frame.

    Args:
        channels (int): Input/hidden channel dimensionality.
        out_channels (int): Output channel dimensionality.
        num_layers (int): Number of table covolution layers
        col_stats (Dict[str, Dict[StatType, Any]]): Mapping from column name to
            column statistics. Easily obtained via :obj:`dataset.col_stats`
        col_names_dict (Dict[torch_frame.stype, List[str]]): Mapping from stype
            to a list of column names in the order stored in the tensor frame.
            Easily obtained via :obj:`tensor_frame.col_names_dict`
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        # kwargs for encoder
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        # Set up feature encoder that maps tensor frame into 3-dimensional x
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            # Specify already-imlemented feature encoder for each stype.
            # The custom feature encoder can be implemented by inheriting
            # torch_frame.nn.StypeEncoder
            stype_encoder_dict={
                # Use nn.Embedding-based encoder for categorical features.
                stype.categorical:
                EmbeddingEncoder(),
                # Use bucket-based encoder for numerical features introduced in
                # https://arxiv.org/abs/2203.05556
                # Apply post-hoc layer normalization
                stype.numerical:
                LinearBucketEncoder(post_module=LayerNorm(channels)),
            },
        )
        # Set up table convolutions that iteratively transforms 3-dimensional
        # x into another x
        self.convs = ModuleList()
        for _ in range(num_layers):
            self.convs.append(SelfAttentionConv(channels))

        # Set up decoder that transforms 3-dimensional x into 2-dimensional
        # output tensor
        self.decoder = MeanDecoder(channels, out_channels)

    def forward(self, tf: TensorFrame) -> Tensor:
        # [batch_size, num_cols, channels]
        x, _ = self.encoder(tf)
        for conv in self.convs:
            # [batch_size, num_cols, channels]
            x = conv(x)
        # [batch_size, out_channels]
        out = self.decoder(x)
        return out


# Set up model and optimizer
model = TabularNN(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()
    loss_accum = 0

    for step, tf in enumerate(tqdm(train_loader)):
        pred = model(tf)
        loss = F.cross_entropy(pred, tf.y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss)
        optimizer.step()
    return loss_accum / (step + 1)


@torch.no_grad()
def eval(loader: DataLoader) -> float:
    model.eval()
    is_corret = []

    for tf in loader:
        pred = model(tf)
        pred_class = pred.argmax(dim=-1)
        is_corret.append((tf.y == pred_class).detach().cpu())

    is_correct_cat = torch.cat(is_corret)
    return float(is_correct_cat.sum()) / len(is_correct_cat)


best_val_acc = 0
best_test_acc = 0
for epoch in range(args.epochs):
    print(f"=====epoch {epoch}")
    loss = train()
    print(f'Train loss: {loss}')
    train_acc = eval(train_loader)
    val_acc = eval(val_loader)
    test_acc = eval(test_loader)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    print(f'Train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}')

print(f'Best val acc: {best_val_acc}, best test acc: {best_test_acc}')
