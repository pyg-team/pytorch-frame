from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleList, Parameter, Sequential

from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.nn.base import Module


def reset_parameters_soft(module: Module):
    r"""Call reset_parameters() only when it exists. Skip activation module."""
    if (hasattr(module, 'reset_parameters')
            and callable(module.reset_parameters)):
        module.reset_parameters()


class StypeEncoder(Module, ABC):
    r"""Base class for stype encoder. This module transforms tensor of a
    specific stype, i.e., `Tensorframe.x_dict[stype.xxx]` into 3-dimensional
    column-wise tensor that is input into :class:`TableConv`.

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same stype.
        post_module (Module, optional): The posthoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will be
            applied to the output. (default: :obj:`None`)
    """
    supported_stypes: Set[stype] = {}
    LAZY_ATTRS = {'out_channels', 'stats_list'}

    @abstractmethod
    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[Module] = None,
    ):
        super().__init__(out_channels, stats_list, post_module)

    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        # Initialize the parameters of `post_module`
        if self.post_module is not None:
            if isinstance(self.post_module, Sequential):
                for module in self.post_module:
                    reset_parameters_soft(module)
            else:
                reset_parameters_soft(self.post_module)

    def post_forward(self, out: Tensor) -> Tensor:
        r"""Post-forward function applied to :obj:`out` of shape
        [batch_size, num_cols, channels]. It also returns :obj:`out` of the
        same shape."""
        if self.post_module is not None:
            shape_before = out.shape
            out = self.post_module(out)
            if out.shape != shape_before:
                raise RuntimeError(
                    f"post_module must not alter the shape of the tensor, but "
                    f"it changed the shape from {shape_before} to "
                    f"{out.shape}.")
        return out


class EmbeddingEncoder(StypeEncoder):
    r"""Embedding look-up based encoder for categorical features. It applies
    :class:`torch.nn.Embedding` for each categorical feature and concatenates
    the output embeddings."""
    supported_stypes = {stype.categorical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[Module] = None,
    ):
        super().__init__(out_channels, stats_list, post_module)

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

    def reset_parameters(self):
        super().reset_parameters()
        for emb in self.embs:
            emb.reset_parameters()

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
        out = torch.stack(xs, dim=1)
        return self.post_forward(out)


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
        post_module: Optional[Module] = None,
    ):
        super().__init__(out_channels, stats_list, post_module)

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

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.1)
        torch.nn.init.zeros_(self.bias)

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
        out = torch.nan_to_num(x, nan=0)
        return self.post_forward(out)


class LinearBucketEncoder(StypeEncoder):
    r"""A numerical converter that transforms a tensor into a piecewise
    linear representation, followed by a linear transformation. The original
    encoding is described in https://arxiv.org/abs/2203.05556"""
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[Module] = None,
    ):
        super().__init__(out_channels, stats_list, post_module)

    def init_modules(self):
        # The min, 25th, 50th, 75th quantile, and max of the column.
        quantiles = [stats[StatType.QUANTILES] for stats in self.stats_list]
        self.boundaries = torch.tensor(quantiles)
        self.interval = self.boundaries[:, 1:] - self.boundaries[:, :-1] + 1e-9
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, self.interval.shape[-1], self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # Reset learnable parameters of the linear transformation
        torch.nn.init.normal_(self.weight, std=0.1)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor):
        encoded_values = []
        for i in range(x.size(1)):
            # Utilize torch.bucketize to find the corresponding bucket indices
            xi = x[:, i].contiguous()
            bucket_indices = torch.bucketize(xi, self.boundaries[i, 1:-1])

            # Create a mask for the one-hot encoding based on bucket indices
            one_hot_mask = torch.nn.functional.one_hot(
                bucket_indices,
                len(self.boundaries[i]) - 1).float()

            # Create a mask for values that are greater than upper bounds
            greater_mask = (x[:, i:i + 1] > self.boundaries[i, :-1]).float()

            # Combine the masks to create encoded_values
            # [batch_size, num_buckets]
            encoded_value = (one_hot_mask * x[:, i:i + 1] - one_hot_mask *
                             self.boundaries[i, :-1].unsqueeze(0)) / (
                                 self.interval[i].unsqueeze(0) + greater_mask *
                                 (1 - one_hot_mask))
            encoded_values.append(encoded_value)
        # Apply column-wise linear transformation
        out = torch.stack(encoded_values, dim=1)
        # [batch_size, num_cols, num_buckets],[num_cols, num_buckets, channels]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum('ijk,jkl->ijl', out, self.weight)
        x = x_lin + self.bias
        out = torch.nan_to_num(x, nan=0)
        return self.post_forward(out)


class LinearPeriodicEncoder(StypeEncoder):
    r"""A periodic encoder that utilizes sinusoidal functions to transform the
    input tensor into a 3-dimensional tensor. The encoding is defined using
    trainable parameters and includes the application of sine and cosine
    functions. The original encoding is described
    in https://arxiv.org/abs/2203.05556.

    Args:
        n_bins (int): Number of bins for periodic encoding
    """
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[Module] = None,
        n_bins: Optional[int] = 16,
    ):
        self.n_bins = n_bins
        super().__init__(out_channels, stats_list, post_module)

    def init_modules(self):
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer('mean', mean)
        std = torch.tensor([stats[StatType.STD]
                            for stats in self.stats_list]) + 1e-6
        self.register_buffer('std', std)
        num_cols = len(self.stats_list)
        self.linear_in = Parameter(torch.empty((num_cols, self.n_bins)))
        self.linear_out = Parameter(
            torch.empty((num_cols, self.n_bins * 2, self.out_channels)))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.normal_(self.linear_in, std=0.1)
        torch.nn.init.normal_(self.linear_out, std=0.1)

    def forward(self, x: Tensor):
        x = (x - self.mean) / self.std
        # Compute the value 'v' by scaling the input 'x' with
        # 'self.linear_in', and applying a 2π periodic
        # transformation.
        v = 2 * torch.pi * self.linear_in[None] * x[..., None]

        # Compute the sine and cosine values and concatenate them
        x = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)

        # [batch_size, num_cols, num_buckets],[num_cols, num_buckets, channels]
        # -> [batch_size, num_cols, channels]
        x = torch.einsum('ijk,jkl->ijl', x, self.linear_out)
        out = torch.nan_to_num(x, nan=0)
        return self.post_forward(out)
