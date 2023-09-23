from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Module, ModuleList, ReLU
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import EmbeddingEncoder
from torch_frame.nn.conv import TabTransformerConv


class MLP(Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.layer1 = Linear(in_channels, hidden_size)
        self.act1 = ReLU()
        self.layer2 = Linear(hidden_size, out_channels)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x


class TabTransformer(Module):
    r"""The FT-Transformer model introduced in https://arxiv.org/abs/2106.11959

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Numner of layers.  (default: 3)
        col_stats (Dict[str, Dict[StatType, Any]]): Dictionary containing
            column statistics
        col_names_dict (Dict[torch_frame.stype, List[str]]): Dictionary
            containing column names categorized by statistical type
        stype_encoder_dict (Optional[Dict[torch_frame.stype,StypeEncoder]) :
            Dictionary containing encoder type per column statistics (default:
            :obj:None, will call EmbeddingEncoder() for categorial feature and
            LinearEncoder() for numerical feature)
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        embedding_pad_dim: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        if stype.categorical in col_names_dict:
            stats_list = [
                col_stats[col_name]
                for col_name in col_names_dict[stype.categorical]
            ]
        else:
            stats_list = []
        self.cat_encoder = EmbeddingEncoder(
            out_channels=channels,
            stats_list=stats_list,
            stype=stype.categorical,
        )
        self.num_encoder = LayerNorm(len(col_names_dict[stype.numerical]))
        self.padded_embedding = Embedding(embedding_pad_dim, channels)
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv(
                channels=channels,
                num_categorical_cols=len(col_names_dict[stype.categorical]),
                num_heads=num_heads) for _ in range(num_layers)
        ])
        self.decoder = MLP(channels, out_channels, channels)
        self.reset_parameters()

    def reset_parameters(self):
        for tab_transformer_conv in self.tab_transformer_convs:
            tab_transformer_conv.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :obj:`TensorFrame` object into output prediction.

        Args:
            x (Tensor): Input :obj:`TensorFrame` object.

        Returns:
            out (Tensor): Output. The shape is [batch_size, out_channels].
        """
        B, _ = tf.x_dict[stype.categorical].shape
        x_cat = self.cat_encoder(tf.x_dict[stype.categorical])
        x_cat = torch.cat((x_cat, self.padded_embedding), dim=1)
        print("shape of x_cat", x_cat.shape)
        x_num = self.num_encoder(tf.x_dict[stype.numerical])
        for tab_transformer_conv in self.tab_transformer_convs:
            x_cat = tab_transformer_conv(x_cat)
        x_cat = x_cat.reshape(B, -1)
        print("dim cate", x_cat.shape)
        print("dim num ", x_num.shape)
        x = torch.cat((x_cat.reshape(B, -1), x_num), dim=1)
        print("dim x ", x.shape)
        out = self.decoder(x)
        return out
