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
        self.channels = in_channels
        self.C = out_channels
        self.W = Linear(num_cols, self.C)
        self.W_d = Linear(self.channels, 1)

    def reset_parameters(self):
        xavier_uniform_(self.W.weight)
        zeros_(self.W.bias)
        xavier_uniform_(self.W_d.weight)
        zeros_(self.W_d.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.W(x)[:, :, 0]
        activition = PReLU()
        x = activition(x)
        x = self.W_d(x)
        return x
