from typing import Any, Dict, List, Optional

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential
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
from torch_frame.nn.conv import FTTransformerConvs


class FTTransformer(Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Numner of layers.  (default: :obj:`3`)
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:obj:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (Optional[Dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            Dictionary containing encoder type per column statistics (default:
            :obj:`None`, will call
            :class:`torch_frame.nn.encoder.EmbeddingEncoder()` for categorial
            feature and :class:`torch_frame.nn.encoder.LinearEncoder()`
            for numerical feature)
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict: Optional[Dict[torch_frame.stype,
                                          StypeEncoder]] = None,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

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
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.backbone.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :obj:`TensorFrame` object into output prediction.

        Args:
            x (:class:`torch_frame.TensorFrame`):
                Input :obj:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        x, x_cls = self.backbone(x)
        out = self.decoder(x_cls)
        return out
