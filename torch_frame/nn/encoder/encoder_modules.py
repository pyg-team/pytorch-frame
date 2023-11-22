import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from torch_frame import stype
class PositionalEncoder(Module):
    def __init__(self, positions: int, num_freqs: int):
        self.num_freqs = num_freqs
        div_term = torch.exp(
            torch.arange(0, self.num_freqs, 2).float() *
            (-np.log(10000.0) / self.num_freqs))
        self.register_buffer("div_term", div_term)
        position = torch.arange(0, stype.timestamp.timestamp_num_cols - 1,
                                dtype=torch.float)
        self.register_buffer("position", position)
        super().__init__()
    
    def forward(self, tensor: Tensor) -> Tensor:
        input.unsqueeze(-1) * torch.arange(num_freqs).reshape((1,) * tensor.ndim + (-1,))
