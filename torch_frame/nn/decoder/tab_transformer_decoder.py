from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential

from torch_frame.nn.decoder import Decoder


class MLP(Module):
    def __init__(self, dims):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = ReLU()
            layers.append(act)

        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TabTransformerDecoder(Decoder):
    r"""The ExcelFormer Prediction Head introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channel dimensionality
        num_cols (int): Number of columns.
    """
    def __init__(self, in_channels: int, out_channels: int, num_cols: int):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into output predictions.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            x (Tensor): [batch_size, out_channels].
        """
        x = x.transpose(1, 2)
        x = self.lin_f(x)
        x = self.activation(x)
        x = self.lin_d(x.transpose(1, 2)).squeeze(2)
        return x
