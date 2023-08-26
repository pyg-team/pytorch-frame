from typing import Optional, Tuple, cast, Dict
import torch
from torch import Tensor
from torch.nn import (
    Dropout,
    LayerNorm,
    Module,
    Linear,
    PReLU
)
from torch.nn.init import xavier_uniform_, zeros_
import math
import torch.nn.functional as F

from torch_frame.nn.conv import TableConv

class AiuM(Module):
    '''
    Attentive Intra-feature Update Module
    '''
    def __init__(self, d, dropout):
        super().__init__()
        self.W_1 = Linear(d, d)
        self.W_2 = Linear(d, d)
        self.dropout = Dropout(dropout)

    def reset_parameters(self):
        for W in [self.W_1, self.W_2]:
            xavier_uniform_(W.weight)
            zeros_(W.bias)
    
    def forward(self, x):
        return self.dropout(F.tanh(self.W_1(x)) * (self.W_2(x)))


class DiaM(Module):
    '''
    Directed Inter-feature Attention Module
    '''
    def __init__(self, d, num_heads, dropout):
        if num_heads > 1:
            assert d % num_heads == 0
        super().__init__()
        self.W_q = Linear(d, d)
        self.W_k = Linear(d, d)
        self.W_v = Linear(d, d)
        self.W_out = Linear(d, d) if num_heads > 1 else None
        self.num_heads = num_heads
        self.dropout = Dropout(dropout) if dropout else None

    def reset_parameters(self):
        for W in [self.W_q, self.W_k, self.W_v]:
            xavier_uniform_(W.weight)
            zeros_(W.bias)
        if self.W_out:
            xavier_uniform_(self.W_out.weight)
            zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        B, channels, d = x.shape
        d_head = d // self.num_heads
        return (
            x.reshape(B, channels, self.num_heads, d_head)
            .transpose(1, 2)
            .reshape(B * self.num_heads, channels, d_head)
        )
    
    def get_attention_mask(self, input_shape, device):
        B, _, seq_len = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = seq_ids[None, None, :].repeat(B, seq_len, 1) <= seq_ids[None, :, None]
        attention_mask = (1.0 - attention_mask.float()) * -1e5
        return attention_mask
    
    def forward(self, z: Tensor) -> Tensor:
        Q, K, V = self.W_q(z), self.W_k(z), self.W_v(z)
        for tensor in [Q, K, V]:
            assert tensor.shape[-1] % self.num_heads == 0
        B = len(Q)
        d = V.shape[-1] // self.num_heads
        Q = self._reshape(Q)
        K = self._reshape(K)
        attention_score = Q @ K.transpose(1, 2) 
        masks = self.get_attention_mask(attention_score.shape, attention_score.device)
        attention = F.softmax((attention_score + masks)/math.sqrt(d), dim=-1)
        x = attention @ self._reshape(V)
        x = x.reshape(B, self.num_heads, Q.shape[-1], d)
        if self.W_out is not None:
            x = self.W_out(x)
        return x

class ExcelFormerConv(TableConv):
    """
    One layer of DiaM and AiuM
    """
    def __init__(self,
                 channels,
                 num_heads,
                 diam_dropout = 0.1,
                 aium_dropout = 0.1,
                 residual_dropout = 0.1,
                 ) -> None:

        super.__init__()
        self.norm_1 = LayerNorm(channels)
        self.DiaM = DiaM(channels, num_heads, diam_dropout)
        d_head = channels // num_heads
        self.norm_2 = LayerNorm(d_head)
        self.AiuM = AiuM(channels, aium_dropout)
        self.residual_dropout = residual_dropout

    def reset_parameters(self):
        self.DiaM.reset_parameters()
        self.AiuM.reset_parameters()

    def _start_residual(self, x):
        x_residual = x
        return x_residual

    def _end_residual(self, x, x_residual):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        return x
    
    def forward(self, x: Tensor):
        x = self.norm_1(x)
        x = self._start_residual(x)
        x_residual = self.DiaM(x)
        x = self._end_residual(x, x_residual)
        x = self._start_residual(x)
        x_residual = self.norm_2(x)
        x_residual = self.AiuM(x)
        x = self._end_residual(x, x_residual)
        return x