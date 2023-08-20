from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList

from torch_frame import Stype
from torch_frame.data.stats import StatType
from torch_frame.nn.base import Module


class StypeEncoder(Module, ABC):
    r"""Base class for stype encoder. This module transforms tensor of a
    specific stype, i.e., `Tensorframe.x_dict[Stype.xxx]` into 3-dimensional
    column-wise tensor that is input into :class:`TableConv`.

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same Stype.
    """
    supported_stypes: Set[Stype] = {}
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
    supported_stypes = {Stype.categorical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
    ):
        super().__init__(out_channels, stats_list)

    def init_modules(self):
        self.embs = ModuleList([])
        for stats in self.stats_list:
            num_categories = len(stats[StatType.CATEGORY_COUNTS][0])
            self.embs.append(Embedding(num_categories, self.out_channels))

    def forward(self, x: Tensor):
        r"""Maps input :obj:`x` from TensorFrame (shape [batch_size, num_cols])
        into output :obj:`x` of shape [batch_size, num_cols, out_channels].
        """
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
    supported_stypes = {Stype.numerical}

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
        self.lin = Linear(num_cols, self.out_channels)

    def forward(self, x: Tensor):
        r"""Maps input :obj:`x` from TensorFrame (shape [batch_size, num_cols])
        into output :obj:`x` of shape [batch_size, num_cols, out_channels].
        """
        # TODO weihua: Handle Nan

        # x: [batch_size, num_cols]
        x = (x - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum('ij,kj->ijk', x, self.lin.weight)
        # [batch_size, num_cols, channels] + [channels]
        # -> [batch_size, num_cols, channels]
        x = x_lin + self.lin.bias
        return x

    def reset_parameters(self):
        self.lin.reset_parameters()
