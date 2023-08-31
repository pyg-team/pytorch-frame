from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential
from torch.nn.modules.module import Module

from torch_frame.nn.conv import FTTransformerConvs


class FTTransformer(Module):
    r"""The FT-Transformer model introduced in https://arxiv.org/abs/2106.11959

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_layers (int): Numner of :class:`TromptConv` layers.  (default: 3)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
        num_layers: int = 3,
    ):
        super().__init__()

        self.backbone = FTTransformerConvs(channels=in_channels,
                                           num_layers=num_layers)
        self.decoder = Sequential(
            LayerNorm(in_channels),
            ReLU(),
            Linear(in_channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.backbone.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into a series of output predictions.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            out (Tensor): Output. The shape is [batch_size, out_channels].
        """

        x, x_cls = self.backbone(x)
        out = self.decoder(x_cls)
        return out
