import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module

from torch_frame.nn.conv import TableConv


class GEGLU(Module):
    r"""GEGLU activation proposed in the :ref:`GLU Variants Improve Transformer
    <https://arxiv.org/abs/2002.05202>`_ paper.
    """
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FFN(Module):
    r"""Feedforward network.

    Args:
        channels (int): Input channel dimensionality
        mult (int): Expansion factor of the first layer (default: :obj:`4`)
        dropout (float): Percentage of random deactivation (default: :obj:`0.`)
    """
    def __init__(
        self,
        channels: int,
        mult: int = 4,
        dropout: float = 0.,
    ) -> None:
        super().__init__()
        self.lin_1 = Linear(channels, mult * channels * 2)
        self.geglu = GEGLU()
        self.dropout = Dropout(dropout)
        self.lin_2 = Linear(channels * mult, channels)

    def reset_parameters(self) -> None:
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin_1(x)
        x = self.geglu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        return x


class SelfAttention(Module):
    r"""Self-attention module.

    Args:
        channels (int): Input channel dimensionality
        num_heads (int): Number of heads in Attention module
        dropout (float): Percentage of random deactivation (default: :obj:`0.`)
    """
    def __init__(self, channels: int, num_heads: int, dropout=0.):
        super().__init__()
        self.lin_q = Linear(channels, channels)
        self.lin_k = Linear(channels, channels)
        self.lin_v = Linear(channels, channels)
        self.lin_out = Linear(channels, channels)
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        d_head = channels // num_heads
        self.scale = d_head**-0.5

    def _reshape(self, x: Tensor) -> Tensor:
        B, num_cols, channels = x.shape
        d_head = channels // self.num_heads
        x = x.reshape(B, num_cols, self.num_heads, d_head)
        x = x.transpose(1, 2)
        x = x.reshape(B * self.num_heads, num_cols, d_head)
        return x

    def forward(self, x: Tensor) -> Tensor:
        B, num_cols, _ = x.shape
        Q, K, V = self.lin_q(x), self.lin_k(x), self.lin_v(x)
        Q = self._reshape(Q)
        K = self._reshape(K)  # b * num_heads, num_cols, d_head
        d_heads = V.shape[-1] // self.num_heads
        attention_score = torch.einsum('ijk, ilk->ijl', Q,
                                       K)  # b * num_heads, num_col, num_col
        scaled_attention_score = attention_score * self.scale
        attention_probs = F.softmax(scaled_attention_score, dim=-1)
        attention = self.dropout(attention_probs)
        x = torch.einsum('ijk, ikl->ijl', attention,
                         self._reshape(V))  # b *num_heads, num_cols, d_heads
        x = x.reshape(B, self.num_heads, num_cols, d_heads)
        x = x.transpose(1, 2)
        x = x.reshape(B, num_cols, self.num_heads * d_heads)
        return self.lin_out(x)

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()
        self.lin_out.reset_parameters()


class TabTransformerConv(TableConv):
    r"""The TabTransformer Layer introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    Args:
        channels (int): Input/output channel dimensionality
        num_heads (int): Number of attention heads
        attn_dropout (float): attention module dropout (default: :obj:`0.`)
        ffn_dropout (float): attention module dropout (default: :obj:`0.`)
    """
    def __init__(self, channels: int, num_heads: int, attn_dropout: float = 0.,
                 ffn_dropout: float = 0.):
        super().__init__()
        self.norm_1 = LayerNorm(channels)
        self.attn = SelfAttention(channels, num_heads, attn_dropout)
        self.norm_2 = LayerNorm(channels)
        self.ffn = FFN(channels, dropout=ffn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_1(x)
        out = self.attn(x)
        x = x + out
        x = self.ffn(x)
        return x

    def reset_parameters(self):
        self.norm_1.reset_parameters()
        self.attn.reset_parameters()
        self.norm_2.reset_parameters()
        self.ffn.reset_parameters()
