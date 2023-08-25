from typing import Optional, Tuple, cast, Dict
import torch
from torch import Tensor
from torch.nn import (
    Dropout,
    LayerNorm,
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
    ModuleList,
    Module,
    ModuleDict,
    Linear,
    PReLU
)
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_, zeros_, _calculate_correct_fan, calculate_gain
import math
import torch.nn.functional as F

from torch_frame.nn.conv import TableConv

def attenuated_kaiming_uniform_(tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def tanglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * torch.tanh(b)

class AiuM(Module):
    '''
    Attentive Intra-feature Update Module
    '''
    def __init__(self, d, dropout):
        super().__init__()
        self.W_1 = Linear(d, d)
        self.W_2 = Linear(d, d)
        for W in [self.W_1, self.W_2]:
            xavier_normal_(W.weight)
            zeros_(W.bias)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(tanglu(self.W_1(x)) * (self.W_2(x)))


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
        for W in [self.W_q, self.W_k, self.W_v]:
            xavier_normal_(W.weight)
            zeros_(W.bias)
        if self.W_out:
            xavier_normal_(self.W_out.weight)
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

class ExcelFormerPredictionHead(Module):
    def __init__(self, channels, num_features, target_category_count):
        super().__init__()
        self.channels = channels
        self.C = target_category_count
        self.W = Linear(num_features, self.C)
        self.W_d = Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x @ self.W.weight + self.W.bias
        x = PReLU(x)
        x = self.W_d(x)
        return x.squeeze(1)

class ExcelFormerConv(TableConv):
    """
    One layer of DiaM and AiuM
    """
    def __init__(self,
                 channels,
                 num_heads,
                 num_features,
                 diam_dropout,
                 aium_dropout,
                 residual_dropout,
                 target_category_count,
                 ) -> None:

        super.__init__()
        self.norm_1 = LayerNorm(channels)
        self.DiaM = DiaM(channels, num_heads, diam_dropout)
        d_head = channels // num_heads
        self.norm_2 = LayerNorm(d_head)
        self.AiuM = AiuM(channels, aium_dropout)
        self.residual_dropout = residual_dropout
        self.prediction_head = ExcelFormerPredictionHead(channels, num_features, target_category_count)

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
        x_residual = self._start_residual(x)
        x = self.DiaM.forward(x)
        x = self._end_residual(x, x_residual)
        x_residual = self._start_residual(x)
        x = self.norm_2(x)
        x = self.AiuM.forward(x)
        x = self._end_residual(x, x_residual)
        if not self.training:
            x = self.prediction_head.forward(x)
        return x