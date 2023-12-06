import math

import torch
from torch import Tensor
from torch.nn import Module


class CyclicEncoding(Module):
    r"""Cyclic encoding for input data containing values between 0 and 1.
    This function maps each value in the input using sine and cosine
    functions of different wavelengths to preserve the cyclical nature. This
    is particularly useful for encoding cyclical features like hours of a
    day, days of the week, etc. Given an input tensor of shape
    :obj:`(*, )`, this encoding expands it into an output tensor of shape
    :obj:`(*, out_size)`.

    Args:
        out_size (int): The output dimension size.
    """
    def __init__(self, out_size: int) -> None:
        super().__init__()
        if out_size % 2 != 0:
            raise ValueError(
                f"out_size should be divisible by 2 (got {out_size}).")
        self.out_size = out_size
        mult_term = torch.arange(1, self.out_size // 2 + 1)
        self.register_buffer("mult_term", mult_term)

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert torch.all((input_tensor >= 0) & (input_tensor <= 1))
        mult_tensor = input_tensor.unsqueeze(-1) * self.mult_term.reshape(
            (1, ) * input_tensor.ndim + (-1, ))
        return torch.cat([
            torch.sin(mult_tensor * math.pi),
            torch.cos(mult_tensor * 2 * math.pi)
        ], dim=-1)
