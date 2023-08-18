from torch import Tensor

from torch_frame.nn.decoder import Decoder


class CLSDecoder(Decoder):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x[:, 0, :]
