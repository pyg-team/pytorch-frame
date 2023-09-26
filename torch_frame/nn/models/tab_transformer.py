import logging
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
    r""" MLP decoder for TabTransformer.

    Args:
        in_channels (int): Input dimensionality.
        out_channles (int): Output dimensionality.
        hidden_size (int): Size of hidden layer.
    """
    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        print("hidden size ", hidden_size)
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
        encoder_pad_size (int): Size of contextual padding to the encoder.
        decoder_hidden_layer_size (int): Size of the hidden layer of MLP
            decoder.
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
        encoder_pad_size: int,
        decoder_hidden_layer_size: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
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
        self.cat_encoder = EmbeddingEncoder(out_channels=channels,
                                            stats_list=stats_list,
                                            stype=stype.categorical)
        self.pad_embedding = Embedding(len(col_names_dict[stype.categorical]),
                                       encoder_pad_size)
        in_channels = channels + encoder_pad_size
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv(channels=in_channels, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.num_encoder = LayerNorm(len(col_names_dict[stype.numerical]))
        self.decoder = MLP(
            len(col_names_dict[stype.categorical]) * in_channels +
            len(col_names_dict[stype.numerical]), out_channels,
            decoder_hidden_layer_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.cat_encoder.reset_parameters()
        self.num_encoder.reset_parameters()
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
            x_cat = self.cat_encoder(
                tf.x_dict[stype.categorical])  # B, col, channel
            x_pad = self.pad_embedding.weight.unsqueeze(0).repeat(
                B, 1, 1)  # B, col, 2
            x_cat = torch.cat((x_cat, x_pad), dim=-1)
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
