import torch
from torch import Tensor
from torch.nn import Linear, PReLU

from torch_frame.nn.decoder import Decoder


class ExcelFormerDecoder(Decoder):
    r"""The ExcelFormer decoder introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channel dimensionality
        num_cols (int): Number of columns.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_f = Linear(num_cols, self.out_channels)
        self.activation = PReLU()
        self.lin_d = Linear(self.in_channels, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_f.reset_parameters()
        self.lin_d.reset_parameters()
        with torch.no_grad():
            self.activation.weight.fill_(0.25)

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into output predictions.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            Tensor: [batch_size, out_channels].
        """
        x = x.transpose(1, 2)
        x = self.lin_f(x)
        x = self.activation(x)
        x = self.lin_d(x.transpose(1, 2)).squeeze(2)
        return x
