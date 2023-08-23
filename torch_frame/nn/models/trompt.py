import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.nn.modules.module import Module

from torch_frame.nn.conv import TromptConv
from torch_frame.nn.decoder import TromptDownstream


class Trompt(Module):
    r"""The Trompt model introduced in https://arxiv.org/abs/2305.18446

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_prompts (int): Number of prompt columns.
        num_layers (int): Numner of :class:`TromptConv` layers.  (default: 6)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
        num_prompts: int,
        num_layers: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_cols = num_cols

        self.x_prompt = Parameter(torch.empty(num_prompts, in_channels))
        self.trompt_convs = ModuleList([
            TromptConv(in_channels, num_cols, num_prompts)
            for _ in range(num_layers)
        ])
        self.trompt_downstream = TromptDownstream(in_channels, out_channels,
                                                  num_prompts)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.x_prompt, std=0.1)
        for trompt_conv in self.trompt_convs:
            trompt_conv.reset_parameters()
        self.trompt_downstream.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into a series of output predictions at each
        layer.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            stacked_out (Tensor): Output predictions stacked across layers. The
                shape is [batch_size, num_layers, out_channels].
        """

        batch_size = len(x)
        outs = []
        # [batch_size, num_prompts, in_channels]
        x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
        for trompt_conv in self.trompt_convs:
            # [batch_size, num_prompts, in_channels]
            x_prompt = trompt_conv(x, x_prompt)
            # [batch_size, out_channels]
            out = self.trompt_downstream(x_prompt)
            # [batch_size, 1, out_channels]
            out = out.view(batch_size, 1, self.out_channels)
            outs.append(out)
        # [batch_size, num_layers, out_channels]
        stacked_out = torch.cat(outs, dim=1)
        return stacked_out
