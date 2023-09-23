import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleList, Parameter, Sequential
from torch.nn.init import kaiming_uniform_

from torch_frame import NAStrategy, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.base import Module

from ..utils.init import attenuated_kaiming_uniform_


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
        stype (stype): The stype of the encoder input.
        post_module (Module, optional): The posthoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will be
            applied to the output. (default: :obj:`None`)
        na_strategy (NAStrategy, optional): The strategy for imputing NaN
            values. If na_strategy is None, then it outputs non-learnable
            all-zero embedding for :obj:`NaN` category. (default: :obj:`None`)
    """
    supported_stypes: Set[stype] = {}
    LAZY_ATTRS = {'out_channels', 'stats_list', 'stype'}

    def __init__(self, out_channels: Optional[int] = None,
                 stats_list: Optional[List[Dict[StatType, Any]]] = None,
                 stype: Optional[stype] = None,
                 post_module: Optional[Module] = None,
                 na_strategy: Optional[NAStrategy] = None):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        if self.na_strategy is not None:
            if (self.stype == stype.numerical
                    and not self.na_strategy.is_numerical_strategy):
                raise ValueError(
                    f"{self.na_strategy} cannot be used on numerical columns.")
            if (self.stype == stype.categorical
                    and not self.na_strategy.is_categorical_strategy):
                raise ValueError(
                    f"{self.na_strategy} cannot be used on categorical"
                    " columns.")

    @abstractmethod
    def reset_parameters(self):
        r"""Initialize the parameters of `post_module`"""
        if self.post_module is not None:
            if isinstance(self.post_module, Sequential):
                for m in self.post_module:
                    reset_parameters_soft(m)
            else:
                reset_parameters_soft(self.post_module)

    def forward(self, x: Tensor):
        # NaN handling of the input Tensor
        x = self.na_forward(x)
        # Main encoding into column embeddings
        x = self.encode_forward(x)
        # Handle NaN in case na_strategy is None
        x = torch.nan_to_num(x, nan=0)
        # Post-forward (e.g., normalization, activation)
        return self.post_forward(x)

    @abstractmethod
    def encode_forward(self, x: Tensor) -> Tensor:
        r"""The main forward function. Maps input :obj:`x` from TensorFrame
        (shape [batch_size, num_cols]) into output :obj:`x` of shape
        [batch_size, num_cols, out_channels]."""
        raise NotImplementedError

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

    def na_forward(self, x: Tensor) -> Tensor:
        r"""Replace NaN values in a x :obj:`Tensor` given na_strategy.

        Args:
            x (Tensor): Input :obj:`Tensor`.

        Returns:
            x (Tensor): Output :obj:`Tensor` with NaNs replaced given
                na_strategy.
        """
        if self.na_strategy is None:
            return x

        x = x.clone()
        for col in range(x.size(1)):
            column_data = x[:, col]
            if self.stype == stype.numerical:
                nan_mask = torch.isnan(column_data)
            else:
                nan_mask = (column_data == -1)
            if not nan_mask.any():
                continue
            if self.na_strategy == NAStrategy.MOST_FREQUENT:
                # Categorical index is sorted based on count,
                # so 0-th index is always the most frequent.
                fill_value = 0
            elif self.na_strategy == NAStrategy.MEAN:
                fill_value = self.stats_list[col][StatType.MEAN]
            elif self.na_strategy == NAStrategy.ZEROS:
                fill_value = 0
            column_data[nan_mask] = fill_value
        return x


class EmbeddingEncoder(StypeEncoder):
    r"""Embedding look-up based encoder for categorical features. It applies
    :class:`torch.nn.Embedding` for each categorical feature and concatenates
    the output embeddings."""
    supported_stypes = {stype.categorical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
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

    def encode_forward(self, x: Tensor) -> Tensor:
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
        return out


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
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
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
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, num_cols]
        x = (x - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum('ij,jk->ijk', x, self.weight)
        # [batch_size, num_cols, channels] + [num_cols, channels]
        # -> [batch_size, num_cols, channels]
        x = x_lin + self.bias
        return x


class StackEncoder(StypeEncoder):
    r"""Simply stack input numerical features of shape [batch_size, num_cols]
    into [batch_size, num_cols, out_channels]."""
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer('mean', mean)
        std = torch.tensor([stats[StatType.STD]
                            for stats in self.stats_list]) + 1e-6
        self.register_buffer('std', std)

    def reset_parameters(self):
        super().reset_parameters()

    def encode_forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, num_cols]
        x = (x - self.mean) / self.std
        # x: [batch_size, num_cols, out_channels]
        x = x.unsqueeze(2).repeat(1, 1, self.out_channels)
        return x


class LinearBucketEncoder(StypeEncoder):
    r"""A numerical converter that transforms a tensor into a piecewise
    linear representation, followed by a linear transformation. The original
    encoding is described in https://arxiv.org/abs/2203.05556"""
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
        # The min, 25th, 50th, 75th quantile, and max of the column.
        quantiles = [stats[StatType.QUANTILES] for stats in self.stats_list]
        boundaries = torch.tensor(quantiles)
        self.register_buffer('boundaries', boundaries)
        self.register_buffer('interval',
                             boundaries[:, 1:] - boundaries[:, :-1] + 1e-8)
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, self.interval.shape[-1], self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # Reset learnable parameters of the linear transformation
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(self, x: Tensor) -> Tensor:
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
        return x


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
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
        n_bins: Optional[int] = 16,
    ):
        self.n_bins = n_bins
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
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
        torch.nn.init.normal_(self.linear_in, std=0.01)
        torch.nn.init.normal_(self.linear_out, std=0.01)

    def encode_forward(self, x: Tensor) -> Tensor:
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
        return x


class ExcelFormerEncoder(StypeEncoder):
    r""" An attention based encoder that transforms input numerical features
    to a 3-dimentional tensor.
    Before being fed to the embedding layer, numerical features are normalized
    and categorical features are transformed into numerical features by the
    CatBoost Encoder implemented with the Sklearn Python package. The features
    are then ranked based on mutural information.
    The original encoding is described in https://arxiv.org/pdf/2301.02819

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same stype.
    """
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: int,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer('mean', mean)
        std = torch.tensor([stats[StatType.STD]
                            for stats in self.stats_list]) + 1e-6
        self.register_buffer('std', std)
        num_cols = len(self.stats_list)
        self.W_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.W_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.reset_parameters()

    def encode_forward(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        x1 = self.W_1[None] * x[:, :, None] + self.b_1[None]
        x2 = self.W_2[None] * x[:, :, None] + self.b_2[None]
        x = torch.tanh(x1) * x2
        return x

    def reset_parameters(self):
        super().reset_parameters()
        attenuated_kaiming_uniform_(self.W_1)
        attenuated_kaiming_uniform_(self.W_2)
        kaiming_uniform_(self.b_1, a=math.sqrt(5))
        kaiming_uniform_(self.b_2, a=math.sqrt(5))
