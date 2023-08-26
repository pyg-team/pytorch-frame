import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.nn.modules.module import Module

from torch_frame.nn.conv import ExcelFormerConv, ExcelFormerPredictionHead


class ExcelFormer(Module):
    def __init__(self, in_channels, out_channels, num_layers, num_heads, num_cols, residual_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.excelformer_convs = ModuleList([
            ExcelFormerConv(in_channels, out_channels, num_heads,
                 num_cols,
                 diam_dropout = 0.1,
                 aium_dropout = 0.1,
                 residual_dropout = residual_dropout)
            for _ in range(num_layers)
        ])
        self.prediction_head = ExcelFormerPredictionHead(in_channels, num_cols, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.prediction_head.reset_parameters()

    def forward(self, x):
        batch_size = len(x)
        outs = []
        # [batch_size, num_features, in_channels]
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        return self.prediction_head(x)
