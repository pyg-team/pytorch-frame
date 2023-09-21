import math
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Identity, Linear, Module, ModuleList

import torch_frame
from torch_frame import stype
from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    StackEncoder,
    StypeWiseFeatureEncoder,
)


class TabNet(Module):
    r"""TODO add doctring"""
    def __init__(
            self,
            out_channels: int,
            # kwargs for encoder
            col_stats: Dict[str, Dict[StatType, Any]],
            col_names_dict: Dict[torch_frame.stype, List[str]],
            # kwargs for TabNet
            split_feature_channels: int = 8,
            split_attention_channels: int = 8,
            num_layers: int = 3,
            gamma: float = 1.3,
            num_shared_glu_layers: int = 2,
            num_dependent_glu_layers: int = 2,
            num_multi: int = 2):
        super().__init__()

        self.split_feature_channels = split_feature_channels
        self.split_attention_channels = split_attention_channels
        self.num_layers = num_layers
        self.gamma = gamma

        num_cols = sum([len(v) for v in col_names_dict.values()])
        in_channels = num_multi * num_cols

        # Maps input tensor frame into (batch_size, num_cols, num_multi),
        # which is then flattened into (batch_size, num_cols * num_multi)
        self.feature_encoder = StypeWiseFeatureEncoder(
            out_channels=num_multi,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: StackEncoder(),
            },
        )

        self.bn = BatchNorm1d(in_channels)

        shared_glu_block = Identity()
        if num_shared_glu_layers > 0:
            shared_glu_block = GLUBlock(
                in_channels=in_channels,
                out_channels=split_feature_channels + split_attention_channels,
                no_first_residual=True,
            )

        self.feat_transformers = ModuleList()
        self.attn_transformers = ModuleList()

        self.feat_transformers.append(
            FeatureTransformer(
                in_channels,
                split_feature_channels + split_attention_channels,
                num_dependent_glu_layers=num_dependent_glu_layers,
                shared_glu_block=shared_glu_block,
            ))

        for _ in range(self.num_layers):
            self.feat_transformers.append(
                FeatureTransformer(
                    in_channels,
                    split_feature_channels + split_attention_channels,
                    num_dependent_glu_layers=num_dependent_glu_layers,
                    shared_glu_block=shared_glu_block,
                ))

            self.attn_transformers.append(
                AttentiveTransformer(
                    in_channels=split_attention_channels,
                    out_channels=in_channels,
                ))

        self.lin = Linear(self.split_feature_channels, out_channels)

    def forward(
            self, tf: TensorFrame,
            return_reg: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""TODO add doctring"""
        # [batch_size, num_cols, num_multi]
        x, _ = self.feature_encoder(tf)
        batch_size = x.shape[0]
        # [batch_size, num_cols * num_multi]
        x = x.view(batch_size, -1)
        x = self.bn(x)

        # [batch_size, num_cols * num_multi]
        prior = torch.ones_like(x)

        if return_reg:
            reg = 0.

        # [batch_size, split_attention_channels]
        attention_x = self.feat_transformers[0](
            x)[:, self.split_feature_channels:]

        outs = []
        for i in range(self.num_layers):
            # [batch_size, num_cols * num_multi]
            attention_mask = self.attn_transformers[i](attention_x, prior)

            # [batch_size, num_cols * num_multi]
            masked_x = attention_mask * x
            # [batch_size, split_feature_channels + split_attention_channel]
            out = self.feat_transformers[i + 1](masked_x)

            # Get the split feature
            # [batch_size, split_feature_channels]
            feature_x = F.relu(out[:, :self.split_feature_channels])
            outs.append(feature_x)
            # Get the split attention
            # [batch_size, split_attention_channels]
            attention_x = out[:, self.split_feature_channels:]

            # Update prior
            prior = (self.gamma - attention_mask) * prior

            # Compute step regularization
            if return_reg:
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
    r"""TODO add doctring"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dependent_glu_layers: int,
        shared_glu_block: Module,
    ):
        super().__init__()

        self.shared_glu_block = shared_glu_block

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

    def forward(self, x: Tensor) -> Tensor:
        x = self.shared_glu_block(x)
        x = self.dependent(x)
        return x


class GLUBlock(Module):
    r"""TODO add doctring"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_glu_layers: int = 2,
        no_first_residual: bool = False,
    ):
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
                x = torch.add(x, glu_layer(x))
                x = x * math.sqrt(0.5)
        return x

    def reset_parameters(self):
        for glu_layer in self.glu_layers:
            glu_layer.reset_parameters()


class GLULayer(Module):
    r"""TODO add doctring"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.lin = Linear(in_channels, out_channels * 2, bias=False)
        # TODO Use GBN instead of BN
        self.bn = BatchNorm1d(out_channels * 2)
        self.glu = GLU()
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = self.bn(x)
        return self.glu(x)

    def reset_parameters(self):
        # TODO Check how linear is reset
        self.lin.reset_parameters()
        self.bn.reset_parameters()


class AttentiveTransformer(Module):
    r"""TODO add doctring"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)
        # TODO Use GBN instead of BN
        self.bn = BatchNorm1d(out_channels)
        self.reset_parameters()

    def forward(self, x: Tensor, prior: Tensor) -> Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = prior * x
        # TODO Use Sparsemax instead of softmax
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        # TODO Check how linear is reset
        self.lin.reset_parameters()
        self.bn.reset_parameters()
