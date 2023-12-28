import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.init import zeros_

from torch_frame.nn.conv import TableConv
from torch_frame.nn.utils.init import attenuated_kaiming_uniform_


def init_attenuated(linear: Linear) -> None:
    attenuated_kaiming_uniform_(linear.weight)
    zeros_(linear.bias)


class AiuM(Module):
    r"""Attentive Intra-feature Update Module.

    Args:
        channels (int): Input channel dimensionality
        dropout (float): Percentage of random deactivation in the AiuM module
    """
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.lin_1 = Linear(channels, channels)
        self.lin_2 = Linear(channels, channels)
        self.dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_attenuated(self.lin_1)
        init_attenuated(self.lin_2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(torch.tanh(self.lin_1(x)) * (self.lin_2(x)))
        return x


class DiaM(Module):
    r"""Directed Inter-feature Attention Module.

    Args:
        channels (int): Input channel dimensionality
        num_cols (int): Number of columns
        num_heads (int): Number of heads in the attention module
        dropout (float): Percentage of random deactivation in the DiaM module
    """
    def __init__(self, channels: int, num_cols: int, num_heads: int,
                 dropout: float) -> None:
        if num_heads > 1:
            assert channels % num_heads == 0
        super().__init__()
        self.lin_q = Linear(channels, channels)
        self.lin_k = Linear(channels, channels)
        self.lin_v = Linear(channels, channels)
        self.lin_out = Linear(channels, channels) if num_heads > 1 else None
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.register_buffer('seq_ids', torch.arange(num_cols))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in [self.lin_q, self.lin_k, self.lin_v]:
            init_attenuated(lin)
        if self.lin_out is not None:
            init_attenuated(self.lin_out)

    def _reshape(self, x: Tensor) -> Tensor:
        B, num_cols, channels = x.shape
        d_head = channels // self.num_heads
        x = x.reshape(B, num_cols, self.num_heads, d_head)
        x = x.transpose(1, 2)
        x = x.reshape(B * self.num_heads, num_cols, d_head)
        return x

    def get_attention_mask(self, input_shape: torch.Size):
        r"""Generate an attention mask for a given input shape.

        The function constructs an attention mask using the sequence ids
        of the input columns. The mask is created such that the elements
        in the upper triangle portion (except for the diagonal elements)
        of the attention map are all zeros. The reset of elements' values
        are 1e-5.

        Args:
            input_shape (torch.Size): Shape of the input tensor. Expected
                to be (Batch size, _, Channels).

        Returns:
            torch.Tensor: The generated attention mask with values 0 or -1e5.
        """
        B, _, num_cols = input_shape
        attention_mask = (self.seq_ids[None, None, :].repeat(B, num_cols, 1)
                          <= self.seq_ids[None, :, None])
        attention_mask = (1.0 - attention_mask.float()) * -1e5
        return attention_mask

    def forward(self, x: Tensor) -> Tensor:
        B, num_cols, _ = x.shape
        Q, K, V = self.lin_q(x), self.lin_k(x), self.lin_v(x)
        d_heads = V.shape[-1] // self.num_heads
        Q = self._reshape(Q)
        K = self._reshape(K)
        attention_score = torch.einsum('ijk, ilk->ijl', Q, K)
        masks = self.get_attention_mask(attention_score.shape)
        scaled_attention_score = (attention_score + masks) / math.sqrt(d_heads)
        attention_probs = F.softmax(scaled_attention_score, dim=-1)
        attention = self.dropout(attention_probs)
        x = torch.einsum('ijk, ikl->ijl', attention, self._reshape(V))
        x = x.reshape(B, self.num_heads, num_cols, d_heads)
        x = x.transpose(1, 2)
        x = x.reshape(B, num_cols, self.num_heads * d_heads)
        if self.lin_out is not None:
            x = self.lin_out(x)
        return x


class ExcelFormerConv(TableConv):
    r"""The ExcelFormer Layer introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    Args:
        channels (int): Input/output channel dimensionality.
        num_cols (int): Number of columns.
        num_heads (int): Number of attention heads.
        diam_dropout (float): diam_dropout. (default: 0)
        aium_dropout (float): aium_dropout. (default: 0)
        residual_dropout (float): residual dropout. (default: 0)
    """
    def __init__(
        self,
        channels: int,
        num_cols: int,
        num_heads: int,
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ) -> None:

        super().__init__()
        self.norm_1 = LayerNorm(channels)
        self.DiaM = DiaM(channels, num_cols, num_heads, diam_dropout)
        self.norm_2 = LayerNorm(channels)
        self.AiuM = AiuM(channels, aium_dropout)
        self.residual_dropout = residual_dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.DiaM.reset_parameters()
        self.AiuM.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_1(x)
        x_residual = self.DiaM(x)
        x = F.dropout(x_residual, self.residual_dropout, self.training) + x
        x_residual = self.norm_2(x)
        x_residual = self.AiuM(x_residual)
        x = F.dropout(x_residual, self.residual_dropout, self.training) + x
        return x
