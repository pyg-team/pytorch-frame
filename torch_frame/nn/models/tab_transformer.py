from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    SELU,
    BatchNorm1d,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv
from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder, StackEncoder
from torch_frame.typing import NAStrategy


class TabTransformer(Module):
    r"""The Tab-Transformer model introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    The model pads a column positional embedding in categorical feature
    embeddings and executes multi-layer column-interaction modeling exclusively
    on the categorical features. For numerical features, the model simply
    applies layer normalization on input features. The model utilizes an
    MLP(Multilayer Perceptron) for decoding.

    .. note::

        For an example of using TabTransformer, see `examples/tabtransformer.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        tabtransformer.py>`_.

    Args:
        channels (int): Input channel dimensionality.
        out_channels (int): Output channels dimensionality.
        num_layers (int): Number of convolution layers.
        num_heads (int): Number of heads in the self-attention layer.
        encoder_pad_size (int): Size of positional encoding padding to the
            categorical embeddings.
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        encoder_pad_size: int,
        attn_dropout: float,
        ffn_dropout: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        self.col_names_dict = col_names_dict
        categorical_col_len = 0
        numerical_col_len = 0
        if stype.categorical in self.col_names_dict:
            categorical_stats_list = [
                col_stats[col_name]
                for col_name in self.col_names_dict[stype.categorical]
            ]
            categorical_col_len = len(self.col_names_dict[stype.categorical])
            self.cat_encoder = EmbeddingEncoder(
                out_channels=channels - encoder_pad_size,
                stats_list=categorical_stats_list,
                stype=stype.categorical,
                na_strategy=NAStrategy.MOST_FREQUENT,
            )
            # Use the categorical embedding with EmbeddingEncoder and
            # added contextual padding to the end of each feature.
            self.pad_embedding = Embedding(categorical_col_len,
                                           encoder_pad_size)
            # Apply transformer convolution only over categorical columns
            self.tab_transformer_convs = ModuleList([
                TabTransformerConv(channels=channels, num_heads=num_heads,
                                   attn_dropout=attn_dropout,
                                   ffn_dropout=ffn_dropout)
                for _ in range(num_layers)
            ])
        if stype.numerical in self.col_names_dict:
            numerical_stats_list = [
                col_stats[col_name]
                for col_name in self.col_names_dict[stype.numerical]
            ]
            numerical_col_len = len(self.col_names_dict[stype.numerical])
            # Use stack encoder to normalize the numerical columns.
            self.num_encoder = StackEncoder(
                out_channels=1,
                stats_list=numerical_stats_list,
                stype=stype.numerical,
            )
            self.num_norm = LayerNorm(numerical_col_len)
        mlp_input_len = categorical_col_len * channels + numerical_col_len
        mlp_first_hidden_layer_size = 2 * mlp_input_len
        mlp_second_hidden_layer_size = 4 * mlp_input_len
        self.decoder = Sequential(
            Linear(mlp_input_len, mlp_first_hidden_layer_size),
            BatchNorm1d(mlp_first_hidden_layer_size), SELU(),
            Linear(2 * mlp_input_len, mlp_second_hidden_layer_size),
            BatchNorm1d(mlp_second_hidden_layer_size), SELU(),
            Linear(mlp_second_hidden_layer_size, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if stype.categorical in self.col_names_dict:
            self.cat_encoder.reset_parameters()
            torch.nn.init.normal_(self.pad_embedding.weight, std=0.01)
            for tab_transformer_conv in self.tab_transformer_convs:
                tab_transformer_conv.reset_parameters()
        if stype.numerical in self.col_names_dict:
            self.num_encoder.reset_parameters()
            self.num_norm.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, SELU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        xs = []
        batch_size = len(tf)
        if stype.categorical in self.col_names_dict:
            x_cat = self.cat_encoder(tf.feat_dict[stype.categorical])
            # A positional embedding [batch_size, num_cols, encoder_pad_size]
            # is padded to the categorical embedding
            # [batch_size, num_cols, channels].
            pos_enc_pad = self.pad_embedding.weight.unsqueeze(0).repeat(
                batch_size, 1, 1)
            # The final categorical embedding is of size [B, num_cols,
            # channels + encoder_pad_size]
            x_cat = torch.cat((x_cat, pos_enc_pad), dim=-1)
            for tab_transformer_conv in self.tab_transformer_convs:
                x_cat = tab_transformer_conv(x_cat)
            x_cat = x_cat.reshape(batch_size, math.prod(x_cat.shape[1:]))
            xs.append(x_cat)
        if stype.numerical in self.col_names_dict:
            x_num = self.num_encoder(tf.feat_dict[stype.numerical])
            x_num = x_num.view(batch_size, math.prod(x_num.shape[1:]))
            x_num = self.num_norm(x_num)
            xs.append(x_num)
        x = torch.cat(xs, dim=1)
        out = self.decoder(x)
        return out
