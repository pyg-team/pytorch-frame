from torch import Tensor

from torch_frame.nn.decoder import Decoder


class CLSDecoder(Decoder):
    """CLS-token decoder used by FT-Transformer model in
    https://arxiv.org/abs/2106.11959

    This module simply extracts the first column embeddings (corresponding to
    the CLS token) of input x output by `torch_frame.nn.conv.FTTransformerConv`
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x[:, 0, :]
