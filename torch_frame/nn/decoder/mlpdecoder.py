from torch import Tensor
from torch.nn import Linear, ReLU

from torch_frame.nn.decoder import Decoder


class MLPDecoder(Decoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_cols: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = ReLU()
        self.lin_1 = Linear(num_cols, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)
        self.lin_3 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()
        self.lin_3.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into output predictions.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            Tensor: [batch_size, out_channels].
        """
        x = self.lin_1(x)
        x = self.activation(x)
        x = self.lin_2(x)
        x = self.activation(x)
        x = self.lin_3(x)
        return x
