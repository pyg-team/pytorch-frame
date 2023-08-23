from typing import Optional, Tuple, cast, Dict
import torch
from torch import Tensor
from torch.nn import (
    LayerNorm,
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
    ModuleList,
    Module,
    ModuleDict,
    Linear,
    MultiheadAttention,
    PReLU
)
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_, zeros_

from torch_frame.nn.conv import TableConv

def tanglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * torch.tanh(b)

class ExcelFormer(TableConv):
    def __init__(self, channels, num_layers, num_heads, attention_dropout, prenormalization, ffn_dropout, residual_dropout):
        super.__init__()
        self.layers = ModuleList([])
        for layer_idx in range(num_layers):
            layer = ModuleDict({
                # Attenuated Initialization
                'attention': MultiheadAttention(
                    channels, num_heads, attention_dropout,
                ),
                'linear0': Linear(channels, channels * 2),
                'norm1': LayerNorm(channels),                
            })
            xavier_uniform_(layer['linear0'].weight)
            zeros_(layer['linear0'].bias)
            if not prenormalization or layer_idx:
                layer['norm0'] = LayerNorm(channels)
            self.layers.append(layer)
        self.activation = tanglu
        self.last_activation = PReLU()
        self.prenormalization = prenormalization
        self.last_normalization = LayerNorm(channels) if prenormalization else None
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