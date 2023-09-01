from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder
from torch_frame.nn.encoder.stype_encoder import ExcelFormerEncoder
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_layers (int): Number of :class:`ExcelFormerConv` layers.
        num_heads (int): Number of attention heads used in :class:`DiaM`
        col_stats (Dict[str, Dict[StatType, Any]]): A dictionary that maps
            column name into stats. Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[torch_frame.stype, List[str]]): A dictionary that
            maps stype to a list of column names. The column names are sorted
            based on the ordering that appear in :obj:`tensor_frame.x_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        diam_dropout (float, optional): diam_dropout (default: :obj:`0.0`)
        aium_dropout (float, optional): aium_dropout (default: :obj:`0.0`)
        residual_dropout (float, optional): residual dropout (default: `0.0`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if len(col_names_dict[stype.categorical]) != 0:
            raise ValueError("ExcelFormer only accepts numerical"
                             " features.")
        assert len(col_names_dict[stype.categorical]) == 0
        self.excelformer_encoder = StypeWiseFeatureEncoder(
            out_channels=self.in_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: ExcelFormerEncoder(out_channels)
            },
        )
        self.excelformer_convs = ModuleList([
            ExcelFormerConv(in_channels, num_cols, num_heads, diam_dropout,
                            aium_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.excelformer_decoder = ExcelFormerDecoder(in_channels,
                                                      out_channels, num_cols)
        self.reset_parameters()

    def reset_parameters(self):
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.excelformer_decoder.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :obj:`TensorFrame` object into
        output predictions.

        Args:
            tf (TensorFrame): Input :obj:TensorFrame column-wise
            tensor-frame object.

        Returns:
            x (Tensor): [batch_size, num_cols, out_channels].
        """
        x, _ = self.excelformer_encoder(tf)
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        return x
