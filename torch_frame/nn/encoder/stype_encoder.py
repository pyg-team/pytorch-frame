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
    specific stype, i.e., `TensorFrame.feat_dict[stype.xxx]` into 3-dimensional
    column-wise tensor that is input into :class:`TableConv`.

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (List[Dict[torch_frame.data.stats.StatType, Any]]): The list
            of stats for each column within the same stype.
        stype (stype): The stype of the encoder input.
        post_module (Module, optional): The post-hoc module applied to the
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
            if self.stype == stype.text_embedded:
                raise ValueError(f"Only the default `na_strategy` (None) "
                                 f"can be used on embedded text columns, but "
                                 f"{self.na_strategy} is given.")

    @abstractmethod
    def reset_parameters(self):
        r"""Initialize the parameters of `post_module`"""
        if self.post_module is not None:
            if isinstance(self.post_module, Sequential):
                for m in self.post_module:
                    reset_parameters_soft(m)
            else:
                reset_parameters_soft(self.post_module)

    def forward(self, feat: Tensor) -> Tensor:
        # Clone the tensor to avoid in-place modification
        feat = feat.clone()
        # NaN handling of the input Tensor
        feat = self.na_forward(feat)
        # Main encoding into column embeddings
        x = self.encode_forward(feat)
        # Handle NaN in case na_strategy is None
        x = torch.nan_to_num(x, nan=0)
        # Post-forward (e.g., normalization, activation)
        return self.post_forward(x)

    @abstractmethod
    def encode_forward(self, feat: Tensor) -> Tensor:
        r"""The main forward function. Maps input :obj:`feat` from TensorFrame
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

    def na_forward(self, feat: Tensor) -> Tensor:
        r"""Replace NaN values in input :obj:`Tensor` given :obj:`na_strategy`.

        Args:
            feat (Tensor): Input :obj:`Tensor`.

        Returns:
            torch.Tensor: Output :obj:`Tensor` with NaNs replaced given
                :obj:`na_strategy`.
        """
        if self.na_strategy is None:
            return feat

        for col in range(feat.size(1)):
            column_data = feat[:, col]
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
            else:
                raise ValueError(f"Unsupported NA strategy {self.na_strategy}")
            column_data[nan_mask] = fill_value
        return feat


class EmbeddingEncoder(StypeEncoder):
    r"""An embedding look-up based encoder for categorical features. It
    applies :class:`torch.nn.Embedding` for each categorical feature and
    concatenates the output embeddings."""
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

    def encode_forward(self, feat: Tensor) -> Tensor:
        # TODO: Make this more efficient.
        # Increment the index by one so that NaN index (-1) becomes 0
        # (padding_idx)
        # feat: [batch_size, num_cols]
        feat = feat + 1
        xs = []
        for i, emb in enumerate(self.embs):
            xs.append(emb(feat[:, i]))
        # [batch_size, num_cols, hidden_channels]
        x = torch.stack(xs, dim=1)
        return x


class LinearEncoder(StypeEncoder):
    r"""A linear function based encoder for numerical features. It applies
    linear layer :obj:`torch.nn.Linear(1, out_channels)` on each raw numerical
    feature and concatenates the output embeddings. Note that the
    implementation does this for all numerical features in a batched manner."""
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

    def encode_forward(self, feat: Tensor) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum('ij,jk->ijk', feat, self.weight)
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

    def encode_forward(self, feat: Tensor) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # x: [batch_size, num_cols, out_channels]
        x = feat.unsqueeze(2).repeat(1, 1, self.out_channels)
        return x


class LinearBucketEncoder(StypeEncoder):
    r"""A numerical converter that transforms a tensor into a piecewise
    linear representation, followed by a linear transformation. The original
    encoding is described in
    `"On Embeddings for Numerical Features in Tabular Deep Learning"
    <https://arxiv.org/abs/2203.05556>`_.
    """
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
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, boundaries.shape[1] - 1, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # Reset learnable parameters of the linear transformation
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(self, feat: Tensor) -> Tensor:
        encoded_values = []
        for i in range(feat.size(1)):
            # Utilize torch.bucketize to find the corresponding bucket indices
            feat_i = feat[:, i].contiguous()
            bucket_indices = torch.bucketize(feat_i, self.boundaries[i, 1:-1])

            # Combine the masks to create encoded_values
            # [batch_size, num_buckets]
            boundary_start = self.boundaries[i, bucket_indices]
            boundary_end = self.boundaries[i, bucket_indices + 1]
            frac = (feat_i - boundary_start) / (boundary_end - boundary_start +
                                                1e-8)
            # Create a mask for values that are greater than upper bounds
            greater_mask = (feat_i.view(-1, 1)
                            > self.boundaries[i, :-1]).float()
            greater_mask[
                torch.arange(len(bucket_indices), device=greater_mask.device),
                bucket_indices] = frac
            encoded_values.append(greater_mask)
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
    functions. The original encoding is described in
    `"On Embeddings for Numerical Features in Tabular Deep Learning"
    <https://arxiv.org/abs/2203.05556>`_.

    Args:
        n_bins (int): Number of bins for periodic encoding.
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

    def encode_forward(self, feat: Tensor) -> Tensor:
        feat = (feat - self.mean) / self.std
        # Compute the value 'v' by scaling the input 'x' with
        # 'self.linear_in', and applying a 2π periodic
        # transformation.
        v = 2 * torch.pi * self.linear_in[None] * feat[..., None]

        # Compute the sine and cosine values and concatenate them
        feat_sincos = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)

        # [batch_size, num_cols, num_buckets],[num_cols, num_buckets, channels]
        # -> [batch_size, num_cols, channels]
        x = torch.einsum('ijk,jkl->ijl', feat_sincos, self.linear_out)
        return x


class ExcelFormerEncoder(StypeEncoder):
    r""" An attention based encoder that transforms input numerical features
    to a 3-dimensional tensor.

    Before being fed to the embedding layer, numerical features are normalized
    and categorical features are transformed into numerical features by the
    CatBoost Encoder implemented with the Sklearn Python package. The features
    are then ranked based on mutual information.
    The original encoding is described in
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.


    Args:
        out_channels (int): The output channel dimensionality.
        stats_list (List[Dict[StatType, Any]]): The list of stats for each
            column within the same stype.
    """
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
        self.W_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.W_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.reset_parameters()

    def encode_forward(self, feat: Tensor) -> Tensor:
        feat = (feat - self.mean) / self.std
        x1 = self.W_1[None] * feat[:, :, None] + self.b_1[None]
        x2 = self.W_2[None] * feat[:, :, None] + self.b_2[None]
        x = torch.tanh(x1) * x2
        return x

    def reset_parameters(self):
        super().reset_parameters()
        attenuated_kaiming_uniform_(self.W_1)
        attenuated_kaiming_uniform_(self.W_2)
        kaiming_uniform_(self.b_1, a=math.sqrt(5))
        kaiming_uniform_(self.b_2, a=math.sqrt(5))


class LinearEmbeddingEncoder(StypeEncoder):
    r"""Linear function based encoder for pre-computed embedding features.
    It applies a linear layer :obj:`torch.nn.Linear(in_channels, out_channels)`
    on each embedding feature (:obj:`in_channels` is the dimensionality of the
    embedding) and concatenates the output embeddings. Note that the
    implementation does this for all numerical features in a batched manner.

    Args:
        in_channels (int): The dimensionality of the embedding feature. Needs
            to be specified manually.
    """
    # NOTE: We currently support text embeddings but in principle, this encoder
    # can support any pre-encoded embeddings, including image/audio/graph
    # embeddings.
    supported_stypes = {stype.text_embedded}

    def __init__(
        self,
        out_channels: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        stype: Optional[stype] = None,
        post_module: Optional[Module] = None,
        na_strategy: Optional[NAStrategy] = None,
        in_channels: Optional[int] = None,
    ):
        if in_channels is None:
            raise ValueError("Please manuallly specify the `in_channels`, "
                             "which is the text embedding dimensionality.")
        self.in_channels = in_channels
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, self.in_channels, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(self, feat: Tensor) -> Tensor:
        # [batch_size, num_cols, in_channels] *
        # [num_cols, in_channels, out_channels]
        # -> [batch_size, num_cols, out_channels]
        x_lin = torch.einsum('ijk,jkl->ijl', feat, self.weight)
        # [batch_size, num_cols, out_channels] + [num_cols, out_channels]
        # -> [batch_size, num_cols, out_channels]
        x = x_lin + self.bias
        return x
