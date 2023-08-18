import torch
from torch import Tensor
from torch.nn import (
    LayerNorm,
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from torch_frame.nn.conv import TableConv


class FTTransformerConv(TableConv):
    """FT-Transformer model introduced in https://arxiv.org/abs/2106.11959
    This module adds the CLS token column before the first column of input x
    and applies a multi-layer Transformer on the resulting tensor. The first
    CLS column can be used for read-out (see :obj:`torch_frame.nn.CLSDecoder`).

    Args:
        channels (int): Input/output channel dimensionality
        num_layers (int): Number of transformer layers. (default: 4)
        nhead (int): Number of heads in multi-head attention (default: 4)
        dropout (int): the dropout value (default: 0.1)
        activation (str): The activation function (default: :obj:`relu`)
    """
    def __init__(
        self,
        channels: int,
        # Arguments for Transformer
        num_layers: int = 4,
        nhead: int = 4,
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=channels,
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
        torch.nn.init.normal_(self.cls_embedding, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        B, _, _ = x.shape
        # [B, 1, C]
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        # [B, N + 1, C]
        x = torch.cat([x_cls, x], dim=1)
        # [B, N + 1, C]
        x = self.transformer(x)
        return x
