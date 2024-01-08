from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Identity, Linear, Module, ModuleList

import torch_frame
from torch_frame import stype
from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    StackEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.typing import NAStrategy


class TabNet(Module):
    r"""The TabNet model introduced in the
    `"TabNet: Attentive Interpretable Tabular Learning"
    <https://arxiv.org/abs/1908.07442>`_ paper.

    .. note::

        For an example of using TabNet, see `examples/tabnet.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        tabnet.py>`_.

    Args:
        out_channels (int): Output dimensionality
        num_layers (int): Number of TabNet layers.
        split_feat_channels (int): Dimensionality of feature channels.
        split_attn_channels (int): Dimensionality of attention channels.
        gamma (float): The gamma value for updating the prior for the attention
            mask.
        col_stats (Dict[str,Dict[torch_frame.data.stats.StatType,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[torch_frame.stype, List[str]]): A
            dictionary that maps :class:`~torch_frame.stype` to a list of
            column names. The column names are sorted based on the ordering
            that appear in :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`StackEncoder()` for
            numerical feature)
        num_shared_glu_layers (int): Number of GLU layers shared across the
            :obj:`num_layers` :class:`FeatureTransformer`s. (default: :obj:`2`)
        num_dependent_glu_layers (int, optional): Number of GLU layers to use
            in each of :obj:`num_layers` :class:`FeatureTransformer`s.
            (default: :obj:`2`)
        cat_emb_channels (int, optional): The categorical embedding
            dimensionality.
    """
    def __init__(
        self,
        out_channels: int,
        num_layers: int,
        split_feat_channels: int,
        split_attn_channels: int,
        gamma: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        num_shared_glu_layers: int = 2,
        num_dependent_glu_layers: int = 2,
        cat_emb_channels: int = 2,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.split_feat_channels = split_feat_channels
        self.split_attn_channels = split_attn_channels
        self.num_layers = num_layers
        self.gamma = gamma

        num_cols = sum([len(v) for v in col_names_dict.values()])
        # if there is no categorical feature, we just set cat_emb_channels to 1
        cat_emb_channels = (cat_emb_channels if torch_frame.categorical
                            in col_names_dict else 1)
        in_channels = cat_emb_channels * num_cols

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical:
                EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
                stype.numerical:
                StackEncoder(na_strategy=NAStrategy.MEAN),
            }

        # Map input tensor frame into (batch_size, num_cols, cat_emb_channels),
        # which is flattened into (batch_size, in_channels)
        self.feature_encoder = StypeWiseFeatureEncoder(
            out_channels=cat_emb_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        # Batch norm applied to input feature.
        self.bn = BatchNorm1d(in_channels)

        shared_glu_block: Module
        if num_shared_glu_layers > 0:
            shared_glu_block = GLUBlock(
                in_channels=in_channels,
                out_channels=split_feat_channels + split_attn_channels,
                no_first_residual=True,
            )
        else:
            shared_glu_block = Identity()

        self.feat_transformers = ModuleList()
        for _ in range(self.num_layers + 1):
            self.feat_transformers.append(
                FeatureTransformer(
                    in_channels,
                    split_feat_channels + split_attn_channels,
                    num_dependent_glu_layers=num_dependent_glu_layers,
                    shared_glu_block=shared_glu_block,
                ))

        self.attn_transformers = ModuleList()
        for _ in range(self.num_layers):
            self.attn_transformers.append(
                AttentiveTransformer(
                    in_channels=split_attn_channels,
                    out_channels=in_channels,
                ))

        self.lin = Linear(self.split_feat_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feature_encoder.reset_parameters()
        self.bn.reset_parameters()
        for feat_transformer in self.feat_transformers:
            feat_transformer.reset_parameters()
        for attn_transformer in self.attn_transformers:
            attn_transformer.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        tf: TensorFrame,
        return_reg: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.
            return_reg (bool): Whether to return the entropy regularization.

        Returns:
            Union[torch.Tensor, (torch.Tensor, torch.Tensor)]: The output
                embeddings of size :obj:`[batch_size, out_channels]`.
                If :obj:`return_reg` is :obj:`True`, return the entropy
                regularization as well.
        """
        # [batch_size, num_cols, cat_emb_channels]
        x, _ = self.feature_encoder(tf)
        batch_size = x.shape[0]
        # [batch_size, num_cols * cat_emb_channels]
        x = x.view(batch_size, math.prod(x.shape[1:]))
        x = self.bn(x)

        # [batch_size, num_cols * cat_emb_channels]
        prior = torch.ones_like(x)
        reg = torch.tensor(0., device=x.device)

        # [batch_size, split_attn_channels]
        attention_x = self.feat_transformers[0](x)
        attention_x = attention_x[:, self.split_feat_channels:]

        outs = []
        for i in range(self.num_layers):
            # [batch_size, num_cols * cat_emb_channels]
            attention_mask = self.attn_transformers[i](attention_x, prior)

            # [batch_size, num_cols * cat_emb_channels]
            masked_x = attention_mask * x
            # [batch_size, split_feat_channels + split_attn_channel]
            out = self.feat_transformers[i + 1](masked_x)

            # Get the split feature
            # [batch_size, split_feat_channels]
            feature_x = F.relu(out[:, :self.split_feat_channels])
            outs.append(feature_x)
            # Get the split attention
            # [batch_size, split_attn_channels]
            attention_x = out[:, self.split_feat_channels:]

            # Update prior
            prior = (self.gamma - attention_mask) * prior

            # Compute entropy regularization
            if return_reg and batch_size > 0:
                entropy = -torch.sum(
                    attention_mask * torch.log(attention_mask + 1e-15),
                    dim=1).mean()
                reg += entropy

        out = sum(outs)
        out = self.lin(out)

        if return_reg:
            return out, reg / self.num_layers
        else:
            return out


class FeatureTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dependent_glu_layers: int,
        shared_glu_block: Module,
    ) -> None:
        super().__init__()

        self.shared_glu_block = shared_glu_block

        self.dependent: Module
        if num_dependent_glu_layers == 0:
            self.dependent = Identity()
        else:
            if not isinstance(self.shared_glu_block, Identity):
                in_channels = out_channels
                no_first_residual = False
            else:
                no_first_residual = True
            self.dependent = GLUBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                no_first_residual=no_first_residual,
                num_glu_layers=num_dependent_glu_layers,
            )
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.shared_glu_block(x)
        x = self.dependent(x)
        return x

    def reset_parameters(self) -> None:
        if not isinstance(self.shared_glu_block, Identity):
            self.shared_glu_block.reset_parameters()
        if not isinstance(self.dependent, Identity):
            self.dependent.reset_parameters()


class GLUBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_glu_layers: int = 2,
        no_first_residual: bool = False,
    ) -> None:
        super().__init__()
        self.no_first_residual = no_first_residual
        self.glu_layers = ModuleList()

        for i in range(num_glu_layers):
            if i == 0:
                glu_layer = GLULayer(in_channels, out_channels)
            else:
                glu_layer = GLULayer(out_channels, out_channels)
            self.glu_layers.append(glu_layer)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            if self.no_first_residual and i == 0:
                x = glu_layer(x)
            else:
                x = x * math.sqrt(0.5) + glu_layer(x)
        return x

    def reset_parameters(self) -> None:
        for glu_layer in self.glu_layers:
            glu_layer.reset_parameters()


class GLULayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = Linear(in_channels, out_channels * 2, bias=False)
        self.glu = GLU()
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        return self.glu(x)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()


class AttentiveTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bn = GhostBatchNorm1d(out_channels)
        self.reset_parameters()

    def forward(self, x: Tensor, prior: Tensor) -> Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = prior * x
        # Using softmax instead of sparsemax since softmax performs better.
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.bn.reset_parameters()


class GhostBatchNorm1d(torch.nn.Module):
    r"""Ghost Batch Normalization https://arxiv.org/abs/1705.08741."""
    def __init__(
        self,
        input_dim: int,
        virtual_batch_size: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        if len(x) > 0:
            num_chunks = math.ceil(len(x) / self.virtual_batch_size)
            chunks = torch.chunk(x, num_chunks, dim=0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

    def reset_parameters(self) -> None:
        self.bn.reset_parameters()
