from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear


class TabNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        split_feature_channels: int = 8,
        split_attention_channels: int = 8,
        num_steps: int = 3,
        gamma: float = 1.3,
        num_shared_glu_layers: int = 2,
        num_dependent_glu_layers: int = 2,
        epsilon: float = 1e-15,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.split_feature_channels = split_feature_channels
        self.split_attention_channels = split_attention_channels
        self.num_steps = num_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_dependent_glu_layers = num_dependent_glu_layers
        self.num_shared_glu_layers = num_shared_glu_layers
        self.attention_channels = self.in_channels

        self.bn = BatchNorm1d(self.in_channels)

        shared_glu_block = torch.nn.Identity()
        if self.num_shared_glu_layers > 0:
            shared_glu_block = GLUBlock(
                in_channels=self.in_channels,
                out_channels=split_feature_channels + split_attention_channels,
                no_first_residual=True,
            )

        self.feat_transformers = torch.nn.ModuleList()
        self.attn_transformers = torch.nn.ModuleList()

        self.feat_transformers.append(
            FeatureTransformer(
                self.in_channels,
                split_feature_channels + split_attention_channels,
                num_dependent_glu_layers=self.num_dependent_glu_layers,
                shared_glu_block=shared_glu_block,
            ))

        for _ in range(self.num_steps):
            self.feat_transformers.append(
                FeatureTransformer(
                    self.in_channels,
                    split_feature_channels + split_attention_channels,
                    num_dependent_glu_layers=self.num_dependent_glu_layers,
                    shared_glu_block=shared_glu_block,
                ))

            self.attn_transformers.append(
                AttentiveTransformer(
                    in_channels=split_attention_channels,
                    out_channels=self.attention_channels,
                ))

    def update_prior(self, prior: Tensor, mask: Tensor) -> Tensor:
        return torch.mul(self.gamma - mask, prior)

    def compute_step_regularization(self, mask: Tensor) -> Tensor:
        return torch.mean(
            torch.sum(torch.mul(mask, torch.log(mask + self.epsilon)), dim=1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.bn(x)

        B = x.shape[0]
        prior = torch.ones((B, self.attention_channels)).to(x.device)

        regularization = 0.0
        attention_x = self.feat_transformers[0](
            x)[:, self.split_feature_channels:]

        outs = []
        for i in range(1, self.num_steps + 1):
            attention_mask = self.attn_transformers[i - 1](prior, attention_x)

            masked_x = torch.mul(attention_mask, x)
            out = self.feat_transformers[i](masked_x)

            # Get the split feature
            feature_x = F.relu(out[:, :self.split_feature_channels])
            outs.append(feature_x)
            # Get the split attention
            attention_x = out[:, self.split_feature_channels:]

            # Update prior
            prior = self.update_prior(prior, attention_mask)

            # Compute step regularization
            regularization += self.compute_step_regularization(attention_mask)

        regularization /= self.num_steps
        out = sum(outs)
        return out, regularization


class FeatureTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dependent_glu_layers: int,
        shared_glu_block: torch.nn.Module,
    ):
        super(FeatureTransformer, self).__init__()

        self.shared_glu_block = shared_glu_block

        if num_dependent_glu_layers == 0:
            self.dependent = torch.nn.Identity()
        else:
            if not isinstance(self.shared_glu_block, torch.nn.Identity):
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

    def forward(self, x):
        x = self.shared_glu_block(x)
        x = self.dependent(x)
        return x


class GLUBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_glu_layers: int = 2,
        no_first_residual: bool = False,
    ):
        super(GLUBlock, self).__init__()
        self.no_first_residual = no_first_residual
        self.glu_layers = torch.nn.ModuleList()
        normalizer = torch.sqrt(torch.tensor([0.5], dtype=torch.float))
        self.register_buffer('normalizer', normalizer)

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
                x = x * self.normalizer
        return x

    def reset_parameters(self):
        for glu_layer in self.glu_layers:
            glu_layer.reset_parameters()


class GLULayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(GLULayer, self).__init__()
        self.fc = Linear(in_channels, out_channels * 2, bias=False)
        # TODO (zecheng): Use GBN instead of BN
        self.bn = BatchNorm1d(out_channels * 2)
        self.glu = torch.nn.GLU()
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.bn(x)
        return self.glu(x)

    def reset_parameters(self):
        # TODO (zecheng): Check how linear is reset
        self.fc.reset_parameters()
        self.bn.reset_parameters()


class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(in_channels, out_channels, bias=False)
        # TODO (zecheng): Use GBN instead of BN
        self.bn = BatchNorm1d(out_channels)
        self.reset_parameters()

    def forward(self, x: Tensor, prior: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = torch.mul(prior, x)
        # TODO (zecheng): Use Sparsemax instead of softmax
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        # TODO (zecheng): Check how linear is reset
        self.fc.reset_parameters()
        self.bn.reset_parameters()
