import math

import torch
from torch import Tensor
from torch.nn import Module


class CyclicEncoding(Module):
    def __init__(self, out_size: int):
        super().__init__()
        if out_size % 2 != 0:
            raise ValueError(
                f"out_size should be divisible by 2 (got {out_size}).")
        self.out_size = out_size
        mult_term = (torch.arange(0, self.out_size, 2) + 2) * math.pi
        self.register_buffer("mult_term", mult_term)

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert torch.all((input_tensor >= 0) & (input_tensor <= 1))
        mult_tensor = input_tensor.unsqueeze(-1) * self.mult_term.reshape(
            (1, ) * input_tensor.ndim + (-1, ))
        return torch.cat([torch.sin(mult_tensor),
                          torch.cos(mult_tensor)], dim=-1)
