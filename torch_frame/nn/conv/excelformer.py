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


class ExcelFormerConv(TableConv):
    """
    One layer of DiaM and AiuM
    """
    def __init__(self,
                 channels,
                 num_heads,
                 attention_dropout,
                 ffn_dropout,
                 residual_dropout,
                 prenormalization,
                 first_layer, 
                 kv_compression: Optional[float],
                 kv_compression_sharing: Optional[str],
                 init_scale: float = 0.1) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super.__init__()
        self.layer = ModuleDict({
            # Attenuated Initialization
            'attention': MultiHeadAttention(
                channels, num_heads, attention_dropout,
            ),
            'linear0': Linear(channels, channels * 2),
            'norm1': LayerNorm(channels),                
        })
        xavier_uniform_(self.layer['linear0'].weight)
        zeros_(self.layer['linear0'].bias)
        if not prenormalization or not first_layer:
            self.layer['norm0'] = LayerNorm(channels)
        self.activation = tanglu
        self.prenormalization = prenormalization
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        self.head = Linear(channels, channels)
        xavier_normal_(self.head.weight)
        self.last_fc = Linear(channels, 1)
        xavier_normal_(self.last_fc.weight)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
    
    def forward(self, x: Tensor):
        for layer_idx, layer in enumerate(self.layers):
            layer = cast(Dict[str, Module], layer)
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = self._end_residual(x, x_residual, layer, 0)

            # reglu
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        x = self.last_fc(x.transpose(1,2))[:,:,0] # b f d -> b d
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x) # TODO: before last_fc？
        x = self.head(x)
        x = x.squeeze(-1)
        return x