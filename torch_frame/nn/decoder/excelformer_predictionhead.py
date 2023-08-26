import torch.nn.functional as F
from torch.nn import Linear, PReLU
from torch.nn.init import xavier_uniform_, zeros_

from torch_frame.nn.decoder import Decoder


class ExcelFormerPredictionHead(Decoder):
    r"""The ExcelFormer Prediction Head introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channel dimensionality
        num_cols (int): Number of columns.
    """
    def __init__(self, in_channels: int, out_channels: int, num_cols: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W_f = Linear(num_cols, self.out_channels)
        self.activation = PReLU()
        self.W_d = Linear(self.in_channels, 1)

    def reset_parameters(self):
        xavier_uniform_(self.W_f.weight)
        zeros_(self.W_f.bias)
        xavier_uniform_(self.W_d.weight)
        zeros_(self.W_d.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.W_f(x)
        x = self.activation(x)
        x = self.W_d(x.transpose(1, 2)).squeeze(2)
        if self.out_channels == 1:
            x = F.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
        return x
