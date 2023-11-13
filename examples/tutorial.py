"""This script gives an example of how to use torch_frame to design tabular
deep learning models in a modular way.
"""

import argparse
import math
import os.path as osp
from typing import Any, Dict, List

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

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Yandex(root=path, name=args.dataset)
# Materialize the dataset, which obtains `TensorFrame` from `DataFrame`.
# `TensorFrame` stores `DataFrame` features as PyTorch tensors organized by
# their stype (semantic type), e.g., categorical, numerical.
dataset.materialize()
# This tutorial only supports training/eval for classification.
assert dataset.task_type.is_classification

# Get pre-defined split
train_dataset, val_dataset, test_dataset = dataset.split()

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame

# Set up data loaders for TensorFrame
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)


# Custom table convolution
class SelfAttentionConv(TableConv):
    r"""Simple self-attention-based table covolution to model interaction
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
        # Mix `x_value` based on the attention weights
        # Shape: [batch_size, num_cols, num_channels]
        out = attn.bmm(x_value)
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
    r"""The overall tabular NN model that takes in `TensorFrame` as input and
    outputs row embeddings. It is a combination of
    (1) Feature encoder (`self.encoder`): Mapping `TensorFrame` into
        3-dimensional :obj:`x` of shape [batch_size, num_cols, channels]
    (2) Table covolutions (`self.convs`): Iteratively transforming the
        3-dimensional :obj:`x`
    (3) Decoder (`self.decoder`): Maps the transformed 3-dimensional :obj:`x`
        into 2-dimensional :obj:`out` of shape [batch_size, out_channels].
        Each element of :obj:`out` represents the row embedding of the original
        `TensorFrame`.

    Args:
        channels (int): Input/hidden channel dimensionality.
        out_channels (int): Output channel dimensionality.
        num_layers (int): Number of table covolution layers
        col_stats (Dict[str, Dict[StatType, Any]]): Mapping from a column name
            to column statistics. Easily obtained via :obj:`dataset.col_stats`
        col_names_dict (Dict[torch_frame.stype, List[str]]): Mapping from stype
            to a list of column names in the order stored in `TensorFrame`.
            Easily obtained via :obj:`tensor_frame.col_names_dict`
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        # kwargs for feature encoder
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        # Specify what feature encoder to use for each stype.
        # The custom feature encoder can be implemented by inheriting
        # torch_frame.nn.StypeEncoder
        stype_encoder_dict = {
            # Use torch.nn.Embedding-based encoder for categorical features.
            stype.categorical:
            EmbeddingEncoder(),
            # Use bucket-based encoder for numerical features introduced in
            # https://arxiv.org/abs/2203.05556
            # Apply post-hoc layer normalization (after linear bucket encoder)
            stype.numerical:
            LinearBucketEncoder(post_module=LayerNorm(channels)),
        }
        # `StypeWiseFeatureEncoder` will take `TensorFrame` as input and apply
        # stype-specific feature encoder (specified via `stype_encoder_dict`)
        # to PyTorch tensor of each stype to get embeddings for each stype. The
        # embeddings of different stypes are then concatenated along the column
        # axis. In all, it transforms `TensorFrame` into 3-dimensional tensor
        # :obj:`x` of shape [batch_size, num_cols, channels].
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        # Set up table convolutions that iteratively transforms 3-dimensional
        # :obj:`x` into another :obj:`x`
        self.convs = ModuleList()
        for _ in range(num_layers):
            self.convs.append(SelfAttentionConv(channels))

        # Set up decoder that transforms 3-dimensional :obj:`x` into
        # 2-dimensional output tensor
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
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        pred = model(tf)
        loss = F.cross_entropy(pred, tf.y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    accum = total_count = 0

    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        pred_class = pred.argmax(dim=-1)
        accum += float((tf.y == pred_class).sum())
        total_count += len(tf.y)

    return accum / total_count


best_val_acc = 0
best_test_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

print(f'Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')
