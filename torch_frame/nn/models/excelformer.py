from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.modules.module import Module

from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_layers (int): Number of :class:`ExcelFormerConv` layers.
        num_heads (int): Number of attention heads used in :class:`DiaM`
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
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into output predictions.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            x (Tensor): [batch_size, num_cols, out_channels].
        """
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        return x
