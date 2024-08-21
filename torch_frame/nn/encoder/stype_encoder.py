from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    Embedding,
    EmbeddingBag,
    ModuleList,
    Parameter,
    ParameterList,
    Sequential,
)
from torch.nn.init import kaiming_uniform_

from torch_frame import NAStrategy, stype
from torch_frame.config import ModelConfig
from torch_frame.data.mapper import TimestampTensorMapper
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.data.multi_tensor import _MultiTensor
from torch_frame.data.stats import StatType
from torch_frame.nn.base import Module
from torch_frame.nn.encoding import CyclicEncoding, PositionalEncoding
from torch_frame.typing import TensorData

from ..utils.init import attenuated_kaiming_uniform_


def reset_parameters_soft(module: Module):
    r"""Call reset_parameters() only when it exists. Skip activation module."""
    if hasattr(module, "reset_parameters") and callable(
            module.reset_parameters):
        module.reset_parameters()


def get_na_mask(tensor: Tensor) -> Tensor:
    r"""Obtains the Na maks of the input :obj:`Tensor`.

    Args:
        tensor (Tensor): Input :obj:`Tensor`.
    """
    if tensor.is_floating_point():
        na_mask = torch.isnan(tensor)
    else:
        na_mask = tensor == -1
    return na_mask


class StypeEncoder(Module, ABC):
    r"""Base class for stype encoder. This module transforms tensor of a
    specific stype, i.e., `TensorFrame.feat_dict[stype.xxx]` into 3-dimensional
    column-wise tensor that is input into :class:`TableConv`.

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (list[dict[torch_frame.data.stats.StatType, Any]]): The list
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

    supported_stypes: set[stype] = {}
    LAZY_ATTRS = {"out_channels", "stats_list", "stype"}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ):
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
            if (self.stype == stype.multicategorical
                    and not self.na_strategy.is_multicategorical_strategy):
                raise ValueError(
                    f"{self.na_strategy} cannot be used on multicategorical"
                    " columns.")
            if (self.stype == stype.timestamp
                    and not self.na_strategy.is_timestamp_strategy):
                raise ValueError(
                    f"{self.na_strategy} cannot be used on timestamp"
                    " columns.")
            elif self.stype == stype.embedding:
                raise ValueError(f"Only the default `na_strategy` (None) "
                                 f"can be used on {self.stype} columns, but "
                                 f"{self.na_strategy} is given.")

            fill_values = []
            for col in range(len(self.stats_list)):
                if self.na_strategy == NAStrategy.MOST_FREQUENT:
                    # Categorical index is sorted based on count,
                    # so 0-th index is always the most frequent.
                    fill_value = 0
                elif self.na_strategy == NAStrategy.MEAN:
                    fill_value = self.stats_list[col][StatType.MEAN]
                elif self.na_strategy == NAStrategy.ZEROS:
                    fill_value = 0
                elif self.na_strategy == NAStrategy.NEWEST_TIMESTAMP:
                    fill_value = self.stats_list[col][StatType.NEWEST_TIME]
                elif self.na_strategy == NAStrategy.OLDEST_TIMESTAMP:
                    fill_value = self.stats_list[col][StatType.OLDEST_TIME]
                elif self.na_strategy == NAStrategy.MEDIAN_TIMESTAMP:
                    fill_value = self.stats_list[col][StatType.MEDIAN_TIME]
                else:
                    raise ValueError(
                        f"Unsupported NA strategy {self.na_strategy}")
                fill_values.append(fill_value)

            if (isinstance(fill_values[0], Tensor)
                    and fill_values[0].size(0) > 1):
                fill_values = torch.stack(fill_values)
            else:
                fill_values = torch.tensor(fill_values)

            self.register_buffer("fill_values", fill_values)

    @abstractmethod
    def reset_parameters(self):
        r"""Initialize the parameters of `post_module`."""
        if self.post_module is not None:
            if isinstance(self.post_module, Sequential):
                for m in self.post_module:
                    reset_parameters_soft(m)
            else:
                reset_parameters_soft(self.post_module)

    def forward(
        self,
        feat: TensorData,
        col_names: list[str] | None = None,
    ) -> Tensor:
        if col_names is not None:
            if isinstance(feat, dict):
                num_cols = next(iter(feat.values())).shape[1]
            else:
                num_cols = feat.shape[1]
            if num_cols != len(col_names):
                raise ValueError(
                    f"The number of columns in feat and the length of "
                    f"col_names must match (got {num_cols} and "
                    f"{len(col_names)}, respectively.)")
        # NaN handling of the input Tensor
        feat = self.na_forward(feat)
        # Main encoding into column embeddings
        x = self.encode_forward(feat, col_names)
        # Handle NaN in case na_strategy is None
        x = torch.nan_to_num(x, nan=0)
        # Post-forward (e.g., normalization, activation)
        return self.post_forward(x)

    @abstractmethod
    def encode_forward(
        self,
        feat: TensorData,
        col_names: list[str] | None = None,
    ) -> Tensor:
        r"""The main forward function. Maps input :obj:`feat` from TensorFrame
        (shape [batch_size, num_cols]) into output :obj:`x` of shape
        :obj:`[batch_size, num_cols, out_channels]`.
        """
        raise NotImplementedError

    def post_forward(self, out: Tensor) -> Tensor:
        r"""Post-forward function applied to :obj:`out` of shape
        [batch_size, num_cols, channels]. It also returns :obj:`out` of the
        same shape.
        """
        if self.post_module is not None:
            shape_before = out.shape
            out = self.post_module(out)
            if out.shape != shape_before:
                raise RuntimeError(
                    f"post_module must not alter the shape of the tensor, but "
                    f"it changed the shape from {shape_before} to "
                    f"{out.shape}.")
        return out

    def na_forward(self, feat: TensorData) -> TensorData:
        r"""Replace NaN values in input :obj:`TensorData` given
        :obj:`na_strategy`.

        Args:
            feat (TensorData): Input :obj:`TensorData`.

        Returns:
            TensorData: Output :obj:`TensorData` with NaNs replaced given
                :obj:`na_strategy`.
        """
        if self.na_strategy is None:
            return feat

        # Since we are not changing the number of items in each column, it's
        # faster to just clone the values, while reusing the same offset
        # object.
        if isinstance(feat, Tensor):
            # cache for future use
            na_mask = get_na_mask(feat)
            feat = feat.clone()
        elif isinstance(feat, MultiEmbeddingTensor):
            feat = MultiEmbeddingTensor(num_rows=feat.num_rows,
                                        num_cols=feat.num_cols,
                                        values=feat.values.clone(),
                                        offset=feat.offset)
        elif isinstance(feat, MultiNestedTensor):
            feat = MultiNestedTensor(num_rows=feat.num_rows,
                                     num_cols=feat.num_cols,
                                     values=feat.values.clone(),
                                     offset=feat.offset)
        else:
            raise ValueError(f"Unrecognized type {type(feat)} in na_forward.")

        if isinstance(feat, _MultiTensor):
            for col, fill_value in enumerate(self.fill_values):
                feat.fillna_col(col, fill_value)
        else:
            if na_mask.ndim == 3:
                # when feat is 3D, it is faster to iterate over columns
                for col, fill_value in enumerate(self.fill_values):
                    col_data = feat[:, col]
                    col_na_mask = na_mask[:, col].any(dim=-1)
                    col_data[col_na_mask] = fill_value
            else:  # na_mask.ndim == 2
                assert feat.size(-1) == self.fill_values.size(-1)
                feat = torch.where(na_mask, self.fill_values, feat)

        return feat


class EmbeddingEncoder(StypeEncoder):
    r"""An embedding look-up based encoder for categorical features. It
    applies :class:`torch.nn.Embedding` for each categorical feature and
    concatenates the output embeddings.
    """

    supported_stypes = {stype.categorical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ) -> None:
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self):
        super().init_modules()
        num_categories_list = [0]
        for stats in self.stats_list:
            num_categories = len(stats[StatType.COUNT][0])
            num_categories_list.append(num_categories)
        # Single embedding module that stores embeddings of all categories
        # across all categorical columns.
        # 0-th category is for NaN.
        self.emb = Embedding(
            sum(num_categories_list) + 1,
            self.out_channels,
            padding_idx=0,
        )
        # [num_cols, ]
        self.register_buffer(
            "offset",
            torch.cumsum(
                torch.tensor(num_categories_list[:-1], dtype=torch.long),
                dim=0))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.emb.reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        # Get NaN mask
        na_mask = feat < 0
        # Increment the index by one not to conflict with the padding idx
        # Also add offset for each column to avoid embedding conflict
        feat = feat + self.offset + 1
        # Use 0th index for NaN
        feat[na_mask] = 0
        # [batch_size, num_cols, channels]
        return self.emb(feat)


class MultiCategoricalEmbeddingEncoder(StypeEncoder):
    r"""An embedding look-up based encoder for multi_categorical features. It
    applies :class:`torch.nn.EmbeddingBag` for each categorical feature and
    concatenates the output embeddings.

    Args:
        mode (str): "sum", "mean" or "max".
            Specifies the way to reduce the bag. (default: :obj:`mean`)
    """

    supported_stypes = {stype.multicategorical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
        mode: str = "mean",
    ) -> None:
        self.mode = mode
        if mode not in ["mean", "sum", "max"]:
            raise ValueError(
                f"Unknown mode {mode} for MultiCategoricalEmbeddingEncoder.",
                "Please use 'mean', 'sum' or 'max'.",
            )
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        self.embs = ModuleList([])
        for stats in self.stats_list:
            num_categories = len(stats[StatType.MULTI_COUNT][0])
            # 0-th category is for NaN.
            self.embs.append(
                EmbeddingBag(
                    num_categories + 1,
                    self.out_channels,
                    padding_idx=0,
                    mode=self.mode,
                ))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for emb in self.embs:
            emb.reset_parameters()

    def encode_forward(
        self,
        feat: MultiNestedTensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # TODO: Make this more efficient.
        # Increment the index by one so that NaN index (-1) becomes 0
        # (padding_idx)
        # feat: [batch_size, num_cols]
        xs = []
        for i, emb in enumerate(self.embs):
            col_feat = feat[:, i]
            xs.append(emb(col_feat.values + 1, col_feat.offset[:-1]))
        # [batch_size, num_cols, hidden_channels]
        x = torch.stack(xs, dim=1)
        return x


class LinearEncoder(StypeEncoder):
    r"""A linear function based encoder for numerical features. It applies
    linear layer :obj:`torch.nn.Linear(1, out_channels)` on each raw numerical
    feature and concatenates the output embeddings. Note that the
    implementation does this for all numerical features in a batched manner.
    """

    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = (torch.tensor([stats[StatType.STD]
                             for stats in self.stats_list]) + 1e-6)
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        self.weight = Parameter(torch.empty(num_cols, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum("ij,jk->ijk", feat, self.weight)
        # [batch_size, num_cols, channels] + [num_cols, channels]
        # -> [batch_size, num_cols, channels]
        x = x_lin + self.bias
        return x


class StackEncoder(StypeEncoder):
    r"""Simply stack input numerical features of shape
    :obj:`[batch_size, num_cols]` into
    :obj:`[batch_size, num_cols, out_channels]`.
    """

    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ) -> None:
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = (torch.tensor([stats[StatType.STD]
                             for stats in self.stats_list]) + 1e-6)
        self.register_buffer("std", std)

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
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
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ) -> None:
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        # The min, 25th, 50th, 75th quantile, and max of the column.
        quantiles = [stats[StatType.QUANTILES] for stats in self.stats_list]
        boundaries = torch.tensor(quantiles)
        self.register_buffer("boundaries", boundaries)
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, boundaries.shape[1] - 1, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # Reset learnable parameters of the linear transformation
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
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
                bucket_indices,
            ] = frac
            encoded_values.append(greater_mask)
        # Apply column-wise linear transformation
        out = torch.stack(encoded_values, dim=1)
        # [batch_size, num_cols, num_buckets],[num_cols, num_buckets, channels]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum("ijk,jkl->ijl", out, self.weight)
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
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
        n_bins: int | None = 16,
    ) -> None:
        self.n_bins = n_bins
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = (torch.tensor([stats[StatType.STD]
                             for stats in self.stats_list]) + 1e-6)
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        self.linear_in = Parameter(torch.empty((num_cols, self.n_bins)))
        self.linear_out = Parameter(
            torch.empty((num_cols, self.n_bins * 2, self.out_channels)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.linear_in, std=0.01)
        torch.nn.init.normal_(self.linear_out, std=0.01)

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        feat = (feat - self.mean) / self.std
        # Compute the value 'v' by scaling the input 'x' with
        # 'self.linear_in', and applying a 2π periodic
        # transformation.
        v = 2 * torch.pi * self.linear_in[None] * feat[..., None]

        # Compute the sine and cosine values and concatenate them
        feat_sincos = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)

        # [batch_size, num_cols, num_buckets],[num_cols, num_buckets, channels]
        # -> [batch_size, num_cols, channels]
        x = torch.einsum("ijk,jkl->ijl", feat_sincos, self.linear_out)
        return x


class ExcelFormerEncoder(StypeEncoder):
    r"""An attention based encoder that transforms input numerical features
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
        stats_list (list[dict[StatType, Any]]): The list of stats for each
            column within the same stype.
    """

    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = (torch.tensor([stats[StatType.STD]
                             for stats in self.stats_list]) + 1e-6)
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        self.W_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.W_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        feat = (feat - self.mean) / self.std
        x1 = self.W_1[None] * feat[:, :, None] + self.b_1[None]
        x2 = self.W_2[None] * feat[:, :, None] + self.b_2[None]
        x = torch.tanh(x1) * x2
        return x

    def reset_parameters(self) -> None:
        super().reset_parameters()
        attenuated_kaiming_uniform_(self.W_1)
        attenuated_kaiming_uniform_(self.W_2)
        kaiming_uniform_(self.b_1, a=math.sqrt(5))
        kaiming_uniform_(self.b_2, a=math.sqrt(5))


class LinearEmbeddingEncoder(StypeEncoder):
    r"""Linear function based encoder for pre-computed embedding features.
    It applies a linear layer :obj:`torch.nn.Linear(emb_dim, out_channels)`
    on each embedding feature and concatenates the output embeddings.
    """
    supported_stypes = {stype.embedding}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ) -> None:
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        num_cols = len(self.stats_list)
        self.emb_dim_list = [
            stats[StatType.EMB_DIM] for stats in self.stats_list
        ]
        self.weight_list = ParameterList([
            Parameter(torch.empty(emb_dim, self.out_channels))
            for emb_dim in self.emb_dim_list
        ])
        self.biases = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for weight in self.weight_list:
            torch.nn.init.normal_(weight, std=0.01)
        torch.nn.init.zeros_(self.biases)

    def encode_forward(
        self,
        feat: MultiEmbeddingTensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        x_lins: list[Tensor] = []
        start_idx = 0
        for idx, col_dim in enumerate(self.emb_dim_list):
            end_idx = start_idx + col_dim
            # [batch_size, emb_dim] * [emb_dim, out_channels]
            # -> [batch_size, out_channels]
            x_lin = feat.values[:, start_idx:end_idx] @ self.weight_list[idx]
            x_lins.append(x_lin)
        # [batch_size, num_cols, out_channels]
        x = torch.stack(x_lins, dim=1)
        # [batch_size, num_cols, out_channels] + [num_cols, out_channels]
        # -> [batch_size, num_cols, out_channels]
        x = x + self.biases
        return x


class LinearModelEncoder(StypeEncoder):
    r"""Linear function based encoder with a specified model outputs embedding
    features. It applies a linear layer :obj:`torch.nn.Linear(in_channels,
    out_channels)` on each embedding feature (:obj:`in_channels` is the
    dimensionality of the embedding) and concatenates the output embeddings.
    The :obj:`model` will also be trained together with the linear layer.
    Note that the implementation does this for all columns in a batched manner.

    Args:
        col_to_model_cfg (dict): A dictionary mapping column names to
            :class:`~torch_frame.config.ModelConfig`, which specifies a model
            to map a single-column :class:`TensorData` object of shape
            :obj:`[batch_size, 1, *]` into row embeddings of shape
            :obj:`[batch_size, 1, model_out_channels]`.
    """

    supported_stypes = {
        stype.text_embedded,
        stype.text_tokenized,
        stype.numerical,
        stype.embedding,
        stype.timestamp,
        stype.categorical,
        stype.multicategorical,
    }

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
        col_to_model_cfg: dict[str, ModelConfig] | None = None,
    ) -> None:
        if col_to_model_cfg is None:
            raise ValueError("Please manually specify `col_to_model_cfg`, "
                             "which outputs embeddings that will be fed into "
                             "linear layer.")
        # TODO: Support non-dictionary col_to_model_cfg
        assert isinstance(col_to_model_cfg, dict)

        self.in_channels_dict = {
            col_name: model_cfg.out_channels
            for col_name, model_cfg in col_to_model_cfg.items()
        }

        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

        self.model_dict = torch.nn.ModuleDict({
            col_name: model_cfg.model
            for col_name, model_cfg in col_to_model_cfg.items()
        })

    def init_modules(self) -> None:
        super().init_modules()
        self.weight_dict = torch.nn.ParameterDict()
        self.bias_dict = torch.nn.ParameterDict()
        for col_name, in_channels in self.in_channels_dict.items():
            self.weight_dict[col_name] = Parameter(
                torch.empty(in_channels, self.out_channels))
            self.bias_dict[col_name] = Parameter(torch.empty(
                self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for col_name in self.weight_dict:
            torch.nn.init.normal_(self.weight_dict[col_name], std=0.01)
            torch.nn.init.zeros_(self.bias_dict[col_name])

    def encode_forward(
        self,
        feat: TensorData,
        col_names: list[str] | None = None,
    ) -> Tensor:
        xs = []
        for i, col_name in enumerate(col_names):
            if self.stype.use_dict_multi_nested_tensor:
                # [batch_size, 1, in_channels]
                x = self.model_dict[col_name]({
                    key: feat[key][:, i]
                    for key in feat
                })
            else:
                input_feat = feat[:, i]

                if input_feat.ndim == 1:
                    # Numerical and categorical cases:
                    # [batch_size] -> [batch_size, 1, 1]
                    input_feat = input_feat.view(-1, 1, 1)
                elif input_feat.ndim == 2:
                    # Timestamp case:
                    # [batch_size, time_size] -> [batch_size, 1, time_size]
                    input_feat = input_feat.unsqueeze(dim=1)

                assert input_feat.ndim == 3
                assert input_feat.shape[:2] == (len(input_feat), 1)

                x = self.model_dict[col_name](input_feat)
            # [batch_size, 1, out_channels]
            x_lin = x @ self.weight_dict[col_name] + self.bias_dict[col_name]
            xs.append(x_lin)
        # [batch_size, num_cols, out_channels]
        x = torch.cat(xs, dim=1)
        return x


class TimestampEncoder(StypeEncoder):
    r"""TimestampEncoder for timestamp stype. Year is encoded with
    :class:`torch_frame.nn.encoding.PositionalEncoding`. The other
    features, including month, day, dayofweek, hour, minute and second,
    are encoded using :class:`torch_frame.nn.encoding.CyclicEncoding`.
    It applies linear layer for each column in a batched manner. The
    TimestampEncoder does not support NaN timestamps, because
    :class:`torch_frame.nn.encoding.PositionalEncoding` does not support
    negative tensor values. So :class:`torch_frame.NAStrategy.MEDIAN_TIMESTAMP`
    is applied as the default :class:`~torch_frame.NAStrategy`.

    Args:
        out_size (int): Output dimension of the positional and cyclic
            encodings.
    """
    supported_stypes = {stype.timestamp}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = NAStrategy.MEDIAN_TIMESTAMP,
        out_size: int = 8,
    ) -> None:
        self.out_size = out_size
        super().__init__(out_channels, stats_list, stype, post_module,
                         na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        # Ensure that the first element is year.
        assert TimestampTensorMapper.TIME_TO_INDEX['YEAR'] == 0

        # Init normalization constant
        min_year = torch.tensor([
            self.stats_list[i][StatType.YEAR_RANGE][0]
            for i in range(len(self.stats_list))
        ])
        self.register_buffer("min_year", min_year)
        max_values = TimestampTensorMapper.CYCLIC_VALUES_NORMALIZATION_CONSTANT
        self.register_buffer("max_values", max_values)

        # Init positional/cyclic encoding
        self.positional_encoding = PositionalEncoding(self.out_size)
        self.cyclic_encoding = CyclicEncoding(self.out_size)

        # Init linear function
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(num_cols, len(TimestampTensorMapper.TIME_TO_INDEX),
                        self.out_size, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        feat = feat.to(torch.float32)
        # [batch_size, num_cols, 1] - [1, num_cols, 1]
        feat_year = feat[..., :1] - self.min_year.view(1, -1, 1)
        # [batch_size, num_cols, num_rest] / [1, 1, num_rest]
        feat_rest = feat[..., 1:] / self.max_values.view(1, 1, -1)
        # [batch_size, num_cols, num_time_feats, out_size]
        x = torch.cat([
            self.positional_encoding(feat_year),
            self.cyclic_encoding(feat_rest)
        ], dim=2)
        # [batch_size, num_cols, num_time_feats, out_size] *
        # [num_cols, num_time_feats, out_size, out_channels]
        # -> [batch_size, num_cols, out_channels]
        x_lin = torch.einsum('ijkl,jklm->ijm', x, self.weight)
        # [batch_size, num_cols, out_channels] + [num_cols, out_channels]
        x = x_lin + self.bias
        return x
