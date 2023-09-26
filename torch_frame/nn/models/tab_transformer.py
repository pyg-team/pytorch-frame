from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import EmbeddingEncoder
from torch_frame.nn.conv import TabTransformerConv


class TabTransformer(Module):
    r"""The Tab-Transformer model introduced in
        https://arxiv.org/abs/2012.06678. The model employs a contextual
        embedding encoder on categorical features and executes multi-layer
        column-interaction modeling exclusively on the categorical features.
        If the input :obj:`TensorFrame` lacks categorical features, the model
        simply applies layer normalization on input features and utilizes an
        MLP(Multilayer Perceptron) for decoding.

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
        self.stypes = list(col_names_dict.keys())
        categorical_stats_list = []
        categorical_col_len = 0
        numerical_col_len = 0
        if stype.categorical in col_names_dict:
            categorical_stats_list = [
                col_stats[col_name]
                for col_name in col_names_dict[stype.categorical]
            ]
            categorical_col_len = len(col_names_dict[stype.categorical])
        if stype.numerical in col_names_dict:
            numerical_col_len = len(col_names_dict[stype.numerical])
        self.cat_encoder = EmbeddingEncoder(out_channels=channels,
                                            stats_list=categorical_stats_list,
                                            stype=stype.categorical)
        # We use the categorical embedding with EmbeddingEncoder and
        # added contextual padding to the end of each feature.
        self.pad_embedding = Embedding(categorical_col_len, encoder_pad_size)
        in_channels = channels + encoder_pad_size
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv(channels=in_channels, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.num_norm = LayerNorm(numerical_col_len)
        self.decoder = Sequential(
            Linear(categorical_col_len * in_channels + numerical_col_len,
                   decoder_hidden_layer_size), ReLU(),
            Linear(decoder_hidden_layer_size, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.cat_encoder.reset_parameters()
        self.num_norm.reset_parameters()
        for tab_transformer_conv in self.tab_transformer_convs:
            tab_transformer_conv.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :obj:`TensorFrame` object into output prediction.

        Args:
            x (Tensor): Input :obj:`TensorFrame` object.

        Returns:
            out (Tensor): Output. The shape is [batch_size, out_channels].
        """
        xs = []
        if stype.categorical in tf.x_dict:
            B, _ = tf.x_dict[stype.categorical].shape
            x_cat = self.cat_encoder(tf.x_dict[stype.categorical])
            # A contextual padding [B, num_cols, encoder_pad_size]is added to
            # the categorical embedding [B, num_cols, channels].
            x_pad = self.pad_embedding.weight.unsqueeze(0).repeat(B, 1, 1)
            # The final categorical embedding is of size [B, num_cols,
            # channels + encoder_pad_size]
            x_cat = torch.cat((x_cat, x_pad), dim=-1)
            for tab_transformer_conv in self.tab_transformer_convs:
                x_cat = tab_transformer_conv(x_cat)
            x_cat = x_cat.reshape(B, -1)
            xs.append(x_cat)
        if stype.numerical in tf.x_dict:
            x_num = self.num_norm(tf.x_dict[stype.numerical])
            xs.append(x_num)
        x = torch.cat(xs, dim=1)
        out = self.decoder(x)
        return out
