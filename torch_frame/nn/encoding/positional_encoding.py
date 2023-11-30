import numpy as np
import torch
from torch import Tensor

from torch_frame.nn.encoding import Encoding


class PositionalEncoding(Encoding):
    r"""Positional encoding introduced in `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ paper.

    Args:
        num_freqs (int): Number of frequencies to be applied to the
        input :obj:`Tensor`, which is also the dimension of the output
        :obj:`Tensor`.
    """
    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        div_term = torch.exp(
            torch.arange(0, self.num_freqs, 2).float() *
            (-np.log(10000.0) / self.num_freqs))
        self.register_buffer("div_term", div_term)
        position = torch.arange(self.num_freqs)
        self.register_buffer("position", position)

    def _forward(self, tensor: Tensor) -> Tensor:
        positional_encoding = tensor.unsqueeze(-1) * self.position.reshape(
            (1, ) * tensor.ndim + (-1, ))
        positional_encoding[..., :self.num_freqs // 2] = torch.sin(
            positional_encoding[..., :self.num_freqs // 2] *
            self.div_term[:self.num_freqs // 2])
        positional_encoding[..., self.num_freqs // 2:] = torch.cos(
            positional_encoding[..., self.num_freqs // 2:] * self.div_term)
        return positional_encoding
