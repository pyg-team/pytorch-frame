from typing import Any, Tuple

import torch
import torch.nn.functional as F
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


class BalanceScoreEstimator(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, x: Tensor):
        return F.sigmoid(self.model(x))


class BCAUSS(Module):
    r"""The BCAUSS model introduced in the
    `"Learning end-to-end patient representations through self-supervised
    covariate balancing for causal treatment effect estimation"
    <https://www.sciencedirect.com/science/article/pii/S1532046423000606>`_
    paper.


    .. note::

        For an example of using ExcelFormer, see `examples/bcauss.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        bcauss.py>`_.

    Args:
        in_channels (int): Input channel dimensionality
        hidden_channels (int): Hidden channel dimensionality
        decoder_hidden_channels (int): Hidden channel dimensionality
            in decoder.
        out_channels (int): Output channels dimensionality
        col_stats(dict[str,dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:obj:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        epsilon (float): Constant weighting factor that controls the relative
            importance of balance score w.r.t. squared factual loss
            (default: :obj:`0.5`).
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        decoder_hidden_channels: int,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]] | None,
        col_names_dict: dict[torch_frame.stype, list[str]] | None,
        epsilon: float = 1.0,
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
        self.balance_score_learner = BalanceScoreEstimator(
            hidden_channels, out_channels)
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
        self.balance_score_learner.reset_parameters()
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
        penalty = self.balance_score_learner(out)
        treated_weight = treated_mask.unsqueeze(-1) / (penalty + 0.01)
        control_weight = ~treated_mask.unsqueeze(-1) / (penalty + 0.01)
        balance_score = torch.mean(
            torch.square(
                torch.sum(treated_weight * x, dim=0) /
                torch.sum(treated_weight + 0.01) -
                torch.sum(control_weight * x, dim=0) /
                (torch.sum(control_weight + 0.01))))
        return pred, self.epsilon * balance_score
