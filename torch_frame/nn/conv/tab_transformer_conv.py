import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module

from torch_frame.nn.conv import TableConv


class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FFN(Module):
    def __init__(self, channels, mult=4, dropout=0.):
        self.lin_1 = Linear(channels, mult * channels * 2)
        self.geglu = GEGLU()
        self.dropout = Dropout(dropout)
        self.lin_2 = Linear(channels * mult, channels)

    def forward(self, x):
        x = self.lin_1(x)
        x = self.geglu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        return x


class Attention(Module):
    def __init__(self, channels: int, num_cols: int, num_heads: int,
                 dropout=0.):
        self.lin_q = Linear(channels, channels)
        self.lin_k = Linear(channels, channels)
        self.lin_v = Linear(channels, channels)
        self.lin_out = Linear(channels, num_cols)
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)

    def _reshape(self, x: Tensor) -> Tensor:
        B, num_cols, channels = x.shape
        d_head = channels // self.num_heads
        x = x.reshape(B, num_cols, self.num_heads, d_head)
        x = x.transpose(1, 2)
        x = x.reshape(B * self.num_heads, num_cols, d_head)
        return x

    def forward(self, x):
        B, num_cols, _ = x
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
        return self.lin_out(x), attention


class TabTransformerConv(TableConv):
    def __init__(self, channels: int, num_categorical_cols: int,
                 num_heads: int, embedding_pad_dim: int, num_layers: int,
                 attn_dropout: float):
        super().__init__()
        # self.padded_embedding = Embedding(embedding_pad_dim, channels)
        self.norm_1 = LayerNorm(channels)
        self.attn = Attention(channels, num_categorical_cols, num_heads,
                              attn_dropout)
        self.norm_2 = LayerNorm(channels)
        self.ffn = FFN(channels)

    def forward(self, x, return_attn=False):
        x = self.norm_1(x)
        out, attention = self.attn(x)
        x += out
        x = self.ffn(x)
        if not return_attn:
            return x
        else:
            return x, attention
