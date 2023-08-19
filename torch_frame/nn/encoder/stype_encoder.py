from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList

from torch_frame import Stype
from torch_frame.data.stats import StatType
from torch_frame.nn.base import Module


class StypeEncoder(Module, ABC):
    r"""Base class for stype encoder. This module transforms
    `Tensorframe.x_dict[Stype.xxx]` into 3-dimensional column-wise tensor that
    is input into :class:`TableConv`.

    Args:
        out_channels (int): The output dimensionality
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same Stype.
    """
    stype_supported: Set[Stype] = {}
    LAZY_ATTRS = {'out_channels', 'stats_list'}

    @abstractmethod
    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.stats_list = stats_list

    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError


class EmbeddingEncoder(StypeEncoder):
    stype_supported = {Stype.categorical}

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
        # x: [batch_size, num_cols]
        xs = []
        for i, emb in enumerate(self.embs):
            xs.append(emb(x[:, i]))
        # [batch_size, num_cols, hidden_channels]
        x = torch.stack(xs, dim=1)
        return x

    def reset_parameters(self):
        for emb in self.embs:
            torch.nn.init.normal_(emb.weight)


class LinearEncoder(StypeEncoder):
    stype_supported = {Stype.numerical}

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
        std = torch.tensor(
            [stats[StatType.STD] + 1e-6 for stats in self.stats_list]) + 1e-6
        self.register_buffer('std', std)
        self.lin = Linear(len(self.stats_list),
                          len(self.stats_list) * self.out_channels)

    def forward(self, x: Tensor):
        # x: [batch_size, num_cols]
        batch_size, num_cols = x.shape
        x = (x - self.mean) / self.std
        # x: [batch_size, num_cols * out_channels]
        x = self.lin(x)
        # x: [batch_size, num_cols, out_channels]
        x = x.view(batch_size, num_cols, self.out_channels)
        return x

    def reset_parameters(self):
        self.lin.reset_parameters()
