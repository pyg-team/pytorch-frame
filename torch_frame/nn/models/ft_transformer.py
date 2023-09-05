from typing import Any, Dict, List

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.nn.conv import FTTransformerConvs


class FTTransformer(Module):
    r"""The FT-Transformer model introduced in https://arxiv.org/abs/2106.11959

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Numner of layers.  (default: 3)
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        # kwargs for encoder
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            },
        )
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.backbone.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :obj:`TensorFrame` object into output prediction.

        Args:
            x (Tensor): Input :obj:`TensorFrame` object.

        Returns:
            out (Tensor): Output. The shape is [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        x, x_cls = self.backbone(x)
        out = self.decoder(x_cls)
        return out
