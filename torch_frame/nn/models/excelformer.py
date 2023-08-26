from typing import Optional

from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.modules.module import Module

from torch_frame.nn.conv import ExcelFormerConv


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_layers (int): Number of :class:`ExcelFormerConv` layers.
        num_heads (int): Number of attention heads used in :class:`DiaM`
        diam_dropout (Optional[float]): diam_dropout (default: None)
        aium_dropout (Optional[float]): aium_dropout (default: None)
        residual_dropout (Optional[float]): residual dropout (default: None)

    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 num_heads: int, diam_dropout: Optional[float] = None,
                 aium_dropout: Optional[float] = None,
                 residual_dropout: Optional[float] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.excelformer_convs = ModuleList([
            ExcelFormerConv(in_channels, num_heads, diam_dropout, aium_dropout,
                            residual_dropout) for _ in range(num_layers)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        return x
