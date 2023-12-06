import torch
from torch import Tensor
from torch.nn import Module


class PositionalEncoding(Module):
    r"""Positional encoding introduced in `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ paper. Given an input tensor of shape
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
        mult_term = torch.pow(1 / 10000.0,
                              torch.arange(0, self.out_size, 2) / out_size)
        self.register_buffer("mult_term", mult_term)

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert torch.all(input_tensor >= 0)
        # (*, 1) * (1, ..., 1, out_size // 2) -> (*, out_size // 2)
        mult_tensor = input_tensor.unsqueeze(-1) * self.mult_term.reshape(
            (1, ) * input_tensor.ndim + (-1, ))
        # cat([(*, out_size // 2), (*, out_size // 2)]) -> (*, out_size)
        return torch.cat([torch.sin(mult_tensor),
                          torch.cos(mult_tensor)], dim=-1)
