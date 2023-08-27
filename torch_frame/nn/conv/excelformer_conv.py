import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.init import zeros_

from torch_frame.nn.conv import TableConv
from torch_frame.utils.initialization_methods import (
    attenuated_kaiming_uniform_,
)


def attenuated_initialization(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        attenuated_kaiming_uniform_(m.weight)
        zeros_(m.bias)


class AiuM(Module):
    r"""Attentive Intra-feature Update Module

    Args:
        channels (int): Input channel dimensionality
        dropout (float): Percentage of random deactivation in the AiuM module
    """
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.lin_1 = Linear(channels, channels)
        self.lin_2 = Linear(channels, channels)
        self.dropout = Dropout(dropout)

    def reset_parameters(self):
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, x):
        x = self.dropout(F.tanh(self.lin_1(x)) * (self.lin_2(x)))
        return x


class DiaM(Module):
    r"""Directed Inter-feature Attention Module

    Args:
        channels (int): Input channel dimensionality
        num_heads (int): Number of heads in the attention module
        dropout (float): Percentage of random deactivation in the DiaM module
    """
    def __init__(self, channels: int, num_heads: int, dropout: float):
        if num_heads > 1:
            assert channels % num_heads == 0
        super().__init__()
        self.lin_q = Linear(channels, channels)
        self.lin_k = Linear(channels, channels)
        self.lin_v = Linear(channels, channels)
        self.lin_out = Linear(channels, channels) if num_heads > 1 else None
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()
        if self.lin_out:
            self.lin_out.reset_parameters()

    def _reshape(self, x: Tensor) -> Tensor:
        B, num_cols, channels = x.shape
        d_head = channels // self.num_heads
        return (x.reshape(B, num_cols, self.num_heads,
                          d_head).transpose(1,
                                            2).reshape(B * self.num_heads,
                                                       num_cols, d_head))

    def get_attention_mask(self, input_shape, device):
        B, _, channels = input_shape
        seq_ids = torch.arange(channels, device=device)
        attention_mask = seq_ids[None, None, :].repeat(B, channels,
                                                       1) <= seq_ids[None, :,
                                                                     None]
        attention_mask = (1.0 - attention_mask.float()) * -1e5
        return attention_mask

    def forward(self, x: Tensor) -> Tensor:
        B, num_cols, _ = x.shape
        Q, K, V = self.lin_q(x), self.lin_k(x), self.lin_v(x)
        for tensor in [Q, K, V]:
            assert tensor.shape[-1] % self.num_heads == 0
        d_heads = V.shape[-1] // self.num_heads
        Q = self._reshape(Q)
        K = self._reshape(K)
        attention_score = torch.einsum('ijk, ilk->ijl', Q, K)
        masks = self.get_attention_mask(attention_score.shape,
                                        attention_score.device)
        attention = self.dropout(
            F.softmax((attention_score + masks) / math.sqrt(d_heads), dim=-1))
        x = torch.einsum('ijk, ikl->ijl', attention, self._reshape(V))
        x = x.reshape(B, self.num_heads, num_cols,
                      d_heads).transpose(1,
                                         2).reshape(B, num_cols,
                                                    self.num_heads * d_heads)
        if self.lin_out is not None:
            x = self.lin_out(x)
        return x


class ExcelFormerConv(TableConv):
    r"""The ExcelFormer Layer introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        channels (int): Input/output channel dimensionality
        num_heads (int): Number of attention heads
        diam_dropout (float): diam_dropout (default: 0)
        aium_dropout (float): aium_dropout (default: 0)
        residual_dropout (float): residual dropout (default: 0)
        initialization (str): 'kaiming' or 'xavier' for attenuated
        kaiming or attenuated xavier initialization
    """
    def __init__(self, channels: int, num_heads: int, diam_dropout: float = 0,
                 aium_dropout: float = 0, residual_dropout: float = 0,
                 initialization: str = 'kaiming') -> None:

        super().__init__()
        self.norm_1 = LayerNorm(channels)
        self.DiaM = DiaM(channels, num_heads, diam_dropout)
        self.norm_2 = LayerNorm(channels)
        self.AiuM = AiuM(channels, aium_dropout)
        self.residual_dropout = residual_dropout

        self.DiaM.apply(attenuated_initialization)
        self.AiuM.apply(attenuated_initialization)

    def reset_parameters(self):
        self.DiaM.reset_parameters()
        self.AiuM.reset_parameters()

    def _start_residual(self, x):
        x_residual = x
        return x_residual

    def _end_residual(self, x, x_residual):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout,
                                   self.training)
        x = x + x_residual
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_1(x)
        x = self._start_residual(x)
        x_residual = self.DiaM(x)
        x = self._end_residual(x, x_residual)
        x = self._start_residual(x)
        x_residual = self.norm_2(x)
        x_residual = self.AiuM(x)
        x = self._end_residual(x, x_residual)
        return x
