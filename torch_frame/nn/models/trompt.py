from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList, Parameter, ReLU, Sequential
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.nn.conv import TromptConv
from torch_frame.nn.decoder import TromptDecoder


class Trompt(Module):
    r"""The Trompt model introduced in the
    `"Trompt: Towards a Better Deep Neural Network for Tabular Data"
    <https://arxiv.org/abs/2305.18446>`_ paper.

    .. note::

        For an example of using Trompt, see `examples/trompt.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        trompt.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_prompts (int): Number of prompt columns.
        num_layers (int, optional): Number of :class:`TromptConv` layers.
            (default: :obj:`6`)
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:obj:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_prompts: int,
        num_layers: int,
        # kwargs for encoder
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])

        self.x_prompt = Parameter(torch.empty(num_prompts, channels))
        self.encoders = ModuleList()
        self.trompt_convs = ModuleList()
        for _ in range(num_layers):
            self.encoders.append(
                StypeWiseFeatureEncoder(
                    out_channels=channels,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict={
                        stype.categorical:
                        EmbeddingEncoder(post_module=LayerNorm(channels)),
                        stype.numerical:
                        LinearEncoder(post_module=Sequential(
                            ReLU(), LayerNorm(channels))),
                    },
                ))
            self.trompt_convs.append(
                TromptConv(channels, num_cols, num_prompts))
        # Decoder is shared across layers.
        self.trompt_decoder = TromptDecoder(channels, out_channels,
                                            num_prompts)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.x_prompt, std=0.01)
        for encoder in self.encoders:
            encoder.reset_parameters()
        for trompt_conv in self.trompt_convs:
            trompt_conv.reset_parameters()
        self.trompt_decoder.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :obj:`TensorFrame` object into a series of output
        predictions at each layer.

        Args:
            tf (:class:`torch_frame.TensorFrame`):
                Input :obj:`TensorFrame` object.

        Returns:
            torch.Tensor: Output predictions stacked across layers. The
            shape is [batch_size, num_layers, out_channels].
        """
        batch_size = len(tf)
        outs = []
        # [batch_size, num_prompts, channels]
        x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
        for i in range(self.num_layers):
            # [batch_size, num_cols, channels]
            x, _ = self.encoders[i](tf)
            # [batch_size, num_prompts, channels]
            x_prompt = self.trompt_convs[i](x, x_prompt)
            # [batch_size, out_channels]
            out = self.trompt_decoder(x_prompt)
            # [batch_size, 1, out_channels]
            out = out.view(batch_size, 1, self.out_channels)
            outs.append(out)
        # [batch_size, num_layers, out_channels]
        stacked_out = torch.cat(outs, dim=1)
        return stacked_out
