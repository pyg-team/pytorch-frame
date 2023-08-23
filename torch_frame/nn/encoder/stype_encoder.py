from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
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
            # 0-th category is for NaN.
            self.embs.append(
                Embedding(
                    num_categories + 1,
                    self.out_channels,
                    padding_idx=0,
                ))
        self.reset_parameters()

    def forward(self, x: Tensor):
        r"""Maps input :obj:`x` from TensorFrame (shape [batch_size, num_cols])
        into output :obj:`x` of shape [batch_size, num_cols, out_channels]. It
        outputs non-learnable all-zero embedding for :obj:`NaN` category
        (specified as -1).
        """
        # TODO: Make this more efficient.
        # Increment the index by one so that NaN index (-1) becomes 0
        # (padding_idx)
        x = x + 1

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
        self.reset_parameters()

    def forward(self, x: Tensor):
        r"""Maps input :obj:`x` from TensorFrame (shape [batch_size, num_cols])
        into output :obj:`x` of shape [batch_size, num_cols, out_channels].  It
        outputs non-learnable all-zero embedding for :obj:`NaN` entries.
        """
        # x: [batch_size, num_cols]
        x = (x - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum('ij,jk->ijk', x, self.weight)
        # [batch_size, num_cols, channels] + [num_cols, channels]
        # -> [batch_size, num_cols, channels]
        x = x_lin + self.bias
        return torch.nan_to_num(x, nan=0)

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=0.1)
        torch.nn.init.zeros_(self.bias)
