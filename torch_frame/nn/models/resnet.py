from typing import Any, Dict, List, Optional

from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
    StypeWiseFeatureEncoder,
)


class FCResidualBlock(Module):
    r"""A module that implements a fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str): The type of normalization to use
            ('batchnorm', 'layernorm', or None).
        dropout_prob (float): The dropout probability (default:
            0.0, i.e., no dropout).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 normalization: Optional[str] = 'layernorm',
                 dropout_prob: float = 0.0):
        super(FCResidualBlock, self).__init__()
        self.linear1 = Linear(in_channels, out_channels)
        self.linear2 = Linear(out_channels, out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_prob)

        if normalization == 'batchnorm':
            self.norm1 = BatchNorm1d(out_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == 'layernorm':
            self.norm1 = LayerNorm(out_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        if in_channels != out_channels:
            self.shortcut = Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.linear1(x)
        if self.norm1:
            out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        if self.norm2:
            out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict: Optional[Dict[torch_frame.stype,
                                          StypeEncoder]] = None,
        normalization: str = 'layernorm',
        dropout_prob: float = 0.2,
    ):
        r"""The ResNet model introduced in https://arxiv.org/abs/2106.11959

        Args:
            channels (int): The number of channels in the backbone layers.
            out_channels (int): The number of output channels in the decoder.
            num_layers (int): The number of layers in the backbone.
            col_stats (Dict[str, Dict[StatType, Any]]): Dictionary containing
                column statistics
            col_names_dict (Dict[torch_frame.stype, List[str]]): Dictionary
                containing column names categorized by statistical type
            stype_encoder_dict (Optional[Dict[torch_frame.stype,StypeEncoder]):
                Dictionary containing encoder type per column statistics
                (default: :obj:None, will call EmbeddingEncoder() for
                categorial feature and LinearEncoder() for numerical feature)
            normalization (str): The type of normalization to use
                ('batchnorm', 'layernorm', or None).
            dropout_prob (float): The dropout probability (default:
                0.2).
        """
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        in_channels = channels * (len(col_stats) - 1)
        self.backbone = Sequential(*[
            FCResidualBlock(in_channels if i == 0 else channels, channels,
                            normalization=normalization,
                            dropout_prob=dropout_prob)
            for i in range(num_layers)
        ])

        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)

        # Flattening the encoder output
        x = x.view(x.size(0), -1)

        x = self.backbone(x)
        out = self.decoder(x)
        return out
