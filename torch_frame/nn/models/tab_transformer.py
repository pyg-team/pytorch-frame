import logging
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList, ReLU
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import ContextualEmbeddingEncoder
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
    r"""The Tab-Transformer model introduced in
        https://arxiv.org/abs/2012.06678

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Numner of layers.
        num_heads (int): Number of heads in the self-attention layer.
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
        self.stypes = []
        if not (stype.categorical in col_names_dict
                and len(col_names_dict[stype.categorical]) != 0):
            logging.info(
                "The data does not contain any categorical columns. "
                "TabTransformer will simply be a multi layer perceptron ")
        else:
            self.stypes.append(stype.categorical)

        if stype.numerical in col_names_dict and len(
                col_names_dict[stype.numerical]) != 0:
            self.stypes.append(stype.numerical)
        stats_list = [
            col_stats[col_name]
            for col_name in col_names_dict[stype.categorical]
        ]
        self.cat_encoder = ContextualEmbeddingEncoder(
            out_channels=channels,
            stats_list=stats_list,
            stype=stype.categorical,
            contextual_column_pad=embedding_pad_dim,
        )
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv(channels=channels, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.num_encoder = LayerNorm(len(col_names_dict[stype.numerical]))
        self.decoder = MLP(
            len(col_names_dict[stype.categorical]) * channels +
            len(col_names_dict[stype.numerical]), out_channels, channels)
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
        xs = []
        if stype.categorical in self.stypes:
            B, _ = tf.x_dict[stype.categorical].shape
            x_cat = self.cat_encoder(tf.x_dict[stype.categorical])
            for tab_transformer_conv in self.tab_transformer_convs:
                x_cat = tab_transformer_conv(x_cat)
            x_cat = x_cat.reshape(B, -1)
            xs.append(x_cat)
        if stype.numerical in self.stypes:
            x_num = self.num_encoder(tf.x_dict[stype.numerical])
            xs.append(x_num)
        x = torch.cat(xs, dim=1)
        out = self.decoder(x)
        return out
