from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import (
    LayerNorm,
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from torch_frame.nn.conv import TableConv


class FTTransformerConvs(TableConv):
    r"""The FT-Transformer backbone in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    This module concatenates a learnable CLS token embedding :obj:`x_cls` to
    the input tensor :obj:`x` and applies a multi-layer Transformer on the
    concatenated tensor. After the Transformer layer, the output tensor is
    divided into two parts: (1) :obj:`x`, corresponding to the original input
    tensor, and (2) :obj:`x_cls`, corresponding to the CLS token tensor.

    Args:
        channels (int): Input/output channel dimensionality
        feedforward_channels (int, optional): Hidden channels used by
            feedforward network of the Transformer model. If :obj:`None`, it
            will be set to :obj:`channels` (default: :obj:`None`)
        num_layers (int): Number of transformer encoder layers. (default: 3)
        nhead (int): Number of heads in multi-head attention (default: 8)
        dropout (int): The dropout value (default: 0.1)
        activation (str): The activation function (default: :obj:`relu`)
    """
    def __init__(
        self,
        channels: int,
        feedforward_channels: int | None = None,
        # Arguments for Transformer
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.2,
        activation: str = 'relu',
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=feedforward_channels or channels,
            dropout=dropout,
            activation=activation,
            # Input and output tensors are provided as
            # [batch_size, seq_len, channels]
            batch_first=True,
        )
        encoder_norm = LayerNorm(channels)
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer,
                                              num_layers=num_layers,
                                              norm=encoder_norm)
        self.cls_embedding = Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""CLS-token augmented Transformer convolution.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_cols, channels]

        Returns:
            (torch.Tensor, torch.Tensor): (Output tensor of shape
            [batch_size, num_cols, channels] corresponding to the input
            columns, Output tensor of shape [batch_size, channels],
            corresponding to the added CLS token column.)
        """
        B, _, _ = x.shape
        # [batch_size, num_cols, channels]
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        # [batch_size, num_cols + 1, channels]
        x_concat = torch.cat([x_cls, x], dim=1)
        # [batch_size, num_cols + 1, channels]
        x_concat = self.transformer(x_concat)
        x_cls, x = x_concat[:, 0, :], x_concat[:, 1:, :]
        return x, x_cls
