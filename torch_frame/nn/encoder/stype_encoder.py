from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, ModuleList, Parameter

from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.nn.base import Module


class StypeEncoder(Module, ABC):
    r"""Base class for stype encoder. This module transforms tensor of a
    specific stype, i.e., `Tensorframe.x_dict[stype.xxx]` into 3-dimensional
    column-wise tensor that is input into :class:`TableConv`.

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same stype.
    """
    supported_stypes: Set[stype] = {}
    LAZY_ATTRS = {'out_channels', 'stats_list'}

    @abstractmethod
    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
    ):
        super().__init__(out_channels, stats_list)

    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError


class EmbeddingEncoder(StypeEncoder):
    r"""Embedding look-up based encoder for categorical features. It applies
    :class:`torch.nn.Embedding` for each categorical feature and concatenates
    the output embeddings."""
    supported_stypes = {stype.categorical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
    ):
        super().__init__(out_channels, stats_list)

    def init_modules(self):
        self.embs = ModuleList([])
        for stats in self.stats_list:
            num_categories = len(stats[StatType.COUNT][0])
            self.embs.append(Embedding(num_categories, self.out_channels))

    def forward(self, x: Tensor):
        r"""Maps input :obj:`x` from TensorFrame (shape [batch_size, num_cols])
        into output :obj:`x` of shape [batch_size, num_cols, out_channels].
        """
        # TODO: Make this more efficient.
        # TODO weihua: Handle Nan

        # x: [batch_size, num_cols]
        xs = []
        for i, emb in enumerate(self.embs):
            xs.append(emb(x[:, i]))
        # [batch_size, num_cols, hidden_channels]
        x = torch.stack(xs, dim=1)
        return x

    def reset_parameters(self):
        for emb in self.embs:
            emb.reset_parameters()


class LinearEncoder(StypeEncoder):
    r"""Linear function based encoder for numerical features. It applies linear
    layer :obj:`torch.nn.Linear(1, out_channels)` on each raw numerical feature
    and concatenates the output embeddings. Note that the implementation does
    this for all numerical features in a batched manner."""
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
    ):
        super().__init__(out_channels, stats_list)

    def init_modules(self):
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer('mean', mean)
        std = torch.tensor([stats[StatType.STD]
                            for stats in self.stats_list]) + 1e-6
        self.register_buffer('std', std)
        num_cols = len(self.stats_list)
        self.weight = Parameter(torch.empty(num_cols, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))

    def forward(self, x: Tensor):
        r"""Maps input :obj:`x` from TensorFrame (shape [batch_size, num_cols])
        into output :obj:`x` of shape [batch_size, num_cols, out_channels].
        """
        # TODO weihua: Handle Nan

        # x: [batch_size, num_cols]
        x = (x - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum('ij,jk->ijk', x, self.weight)
        # [batch_size, num_cols, channels] + [num_cols, channels]
        # -> [batch_size, num_cols, channels]
        x = x_lin + self.bias
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=0.1)
        torch.nn.init.zeros_(self.bias)


class LinearBucketEncoder(StypeEncoder):
    r"""A numerical converter that transforms a tensor into a piecewise
    linear representation, followed by a linear transformation. The original encoding is described in https://arxiv.org/abs/2203.05556.

    Args:
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same stype.
            - StatType.QUANTILES: The min, 25th, 50th, 75th quantile, and max of the column.
        out_channels (Optional[int]): The number of output channels for the linear layer.
    """
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
    ):
        super().__init__(out_channels, stats_list)

    def init_modules(self):
        quantiles = [stats[StatType.QUANTILES] for stats in self.stats_list]
        self.boundaries = torch.tensor(quantiles)
        self.interval = self.boundaries[:, 1:] - self.boundaries[:, :-1] + 1e-9
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, self.interval.shape[-1], self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))

    def forward(self, x: Tensor):
        encoded_values = []
        for i in range(x.size(1)):
            # Utilize torch.bucketize to find the corresponding bucket indices
            bucket_indices = torch.bucketize(x[:, i], self.boundaries[i, 1:-1])

            # Create a mask for the one-hot encoding based on bucket indices
            one_hot_mask = torch.nn.functional.one_hot(
                bucket_indices,
                len(self.boundaries[i]) - 1).float()

            # Create a mask for values that are greater than upper bounds
            greater_mask = (x[:, i:i + 1] > self.boundaries[i, :-1]).float()

            # Combine the masks to create encoded_values
            encoded_value = (one_hot_mask * x[:, i:i + 1] - one_hot_mask *
                             self.boundaries[i, :-1].unsqueeze(0)
                             ) / self.interval[i].unsqueeze(
                                 0) + greater_mask * (1 - one_hot_mask)
            encoded_values.append(encoded_value)

        # Apply column-wise linear transformation
        encoded_values = torch.stack(encoded_values, dim=1).squeeze()
        x_lin = torch.einsum('ijk,jkl->ijl', encoded_values, self.weight)
        x = x_lin + self.bias
        return x

    def reset_parameters(self):
        # Reset learnable parameters of the linear transformation
        torch.nn.init.normal_(self.weight, std=0.1)
        torch.nn.init.zeros_(self.bias)