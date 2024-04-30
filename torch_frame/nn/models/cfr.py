from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential

import torch_frame
from torch_frame import TensorFrame, categorical, numerical
from torch_frame.data.stats import StatType


class MLPBlock(Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.model = Sequential(*[
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        ])

    def reset_parameters(self) -> None:
        for block in self.model:
            if isinstance(block, Linear):
                block.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforming :obj:`x` into output predictions.

        Args:
            x (Tensor): Input column-wise tensor of shape
                [batch_size, num_cols, in_channels]

        Returns:
            Tensor: [batch_size, out_channels].
        """
        return self.model(x)


class CFR(Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        decoder_hidden_channels: int,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]] | None,
        col_names_dict: dict[torch_frame.stype, list[str]] | None,
        epsilon: float = 0.3,
    ):

        super().__init__()
        if col_stats is not None and col_names_dict is not None:
            numerical_stats_list = [
                col_stats[col_name]
                for col_name in col_names_dict[torch_frame.numerical]
            ]
            mean = torch.tensor(
                [stats[StatType.MEAN] for stats in numerical_stats_list])
            self.register_buffer("mean", mean)
            std = (torch.tensor(
                [stats[StatType.STD]
                 for stats in numerical_stats_list]) + 1e-6)
            self.register_buffer("std", std)
        self.representation_learner = MLPBlock(channels, hidden_channels,
                                               hidden_channels)
        # decoder for treatment group
        self.treatment_decoder = MLPBlock(hidden_channels,
                                          decoder_hidden_channels,
                                          out_channels)
        # decoder for control group
        self.control_decoder = MLPBlock(hidden_channels,
                                        decoder_hidden_channels, out_channels)
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_learner.reset_parameters()
        self.treatment_decoder.reset_parameters()
        self.control_decoder.reset_parameters()

    def forward(self, tf: TensorFrame,
                treatment_index: int) -> Tuple[Tensor, Tensor]:
        r"""T stands for treatment and y stands for output."""
        feat_cat = tf.feat_dict[categorical]
        feat_num = tf.feat_dict[numerical]
        if hasattr(self, 'mean'):
            feat_num = (feat_num - self.mean) / self.std
        assert isinstance(feat_cat, Tensor)
        assert isinstance(feat_num, Tensor)
        x = torch.cat([feat_cat, feat_num], dim=1)
        t = x[:, treatment_index].clone()
        # Swap the treatment col with the last column of x
        x[:, treatment_index] = x[:, -1]
        x[:, -1] = t
        # Remove the treatment column
        x = x[:, :-1]

        out = self.representation_learner(x)  # batch_size, hidden_channels
        treated_mask = t == 1
        treated = out[treated_mask, :]
        control = out[~treated_mask, :]
        pred = torch.zeros((len(x), 1), dtype=x.dtype, device=x.device)
        pred[~treated_mask, :] = self.control_decoder(control)
        pred[treated_mask, :] = self.treatment_decoder(treated)
        treated_mean = torch.mean(treated, dim=0)
        control_mean = torch.mean(control, dim=0)
        ipm = 2 * torch.norm(treated_mean - control_mean, p=2)
        return pred, self.epsilon * ipm
