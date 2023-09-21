import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, Embedding, ModuleList
from torch.nn.init import zeros_

from torch_frame.nn.conv import TableConv

from ..utils.init import attenuated_kaiming_uniform_

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP(Module):
    def __init__(self, channles: int):
        pass
class TabTransformerConv(TableConv):
    def __init__(self, channels: int, num_categorical_cols: int, embedding_pad_dim: int, num_layers: int):
        super().__init__()
        # self.padded_embedding = Embedding(embedding_pad_dim, channels)
        self.layers = ModuleList([])
        self.pre_norm_1 = PreNorm(channels, Attention())
