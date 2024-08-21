from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder
from torch_frame.nn.encoder.stype_encoder import ExcelFormerEncoder
from torch_frame.nn.encoder.stypewise_encoder import (
    StypeEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.typing import NAStrategy


def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: float | Tensor = 0.5,
    mixup_type: str | None = None,
    mi_scores: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Mixup input numerical feature tensor :obj:`x` by swapping some
    feature elements of two shuffled sample samples. The shuffle rates for
    each row is sampled from the Beta distribution. The target `y` is also
    linearly mixed up.

    Args:
        x (Tensor): The input numerical feature.
        y (Tensor): The target.
        num_classes (int): Number of classes.
        beta (float): The concentration parameter of the Beta distribution.
            (default: :obj:`0.5`)
        mixup_type (str, optional): The mixup methods. No mixup if set to
            :obj:`None`, options `feature` and `hidden` are `FEAT-MIX`
            (mixup at feature dimension) and `HIDDEN-MIX` (mixup at
            hidden dimension) proposed in ExcelFormer paper.
            (default: :obj:`None`)
        mi_scores (Tensor, optional): Mutual information scores only used in
            the mixup weight calculation for `FEAT-MIX`.
            (default: :obj:`None`)

    Returns:
        x_mixedup (Tensor): The mixedup numerical feature.
        y_mixedup (Tensor): Transformed target of size
            :obj:`[batch_size, num_classes]`
    """
    assert num_classes > 0
    assert mixup_type in [None, 'feature', 'hidden']

    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample(torch.Size((len(x), 1)))
    shuffled_idx = torch.randperm(len(x), device=x.device)
    assert x.ndim == 3, """
    FEAT-MIX or HIDDEN-MIX is for encoded numerical features
    of size [batch_size, num_cols, in_channels]."""
    b, f, d = x.shape
    if mixup_type == 'feature':
        assert mi_scores is not None
        mi_scores = mi_scores.to(x.device)
        # Hard mask (feature dimension)
        mixup_mask = torch.rand(torch.Size((b, f)),
                                device=x.device) < shuffle_rates
        # L1 normalized mutual information scores
        norm_mi_scores = mi_scores / mi_scores.sum()
        # Mixup weights
        lam = torch.sum(
            norm_mi_scores.unsqueeze(0) * mixup_mask, dim=1, keepdim=True)
        mixup_mask = mixup_mask.unsqueeze(2)
    elif mixup_type == 'hidden':
        # Hard mask (hidden dimension)
        mixup_mask = torch.rand(torch.Size((b, d)),
                                device=x.device) < shuffle_rates
        mixup_mask = mixup_mask.unsqueeze(1)
        # Mixup weights
        lam = shuffle_rates
    else:
        # No mixup
        mixup_mask = torch.ones_like(x, dtype=torch.bool)
        # Fake mixup weights
        lam = torch.ones_like(shuffle_rates)
    x_mixedup = mixup_mask * x + ~mixup_mask * x[shuffled_idx]

    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        lam = lam.squeeze(1)
        y_mixedup = lam * y + (1 - lam) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (lam * one_hot_y + (1 - lam) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    ExcelFormer first converts the categorical features with a target
    statistics encoder (i.e., :class:`CatBoostEncoder` in the paper)
    into numerical features. Then it sorts the numerical features
    with mutual information sort. So the model itself limits to
    numerical features.

    .. note::

        For an example of using ExcelFormer, see `examples/excelformer.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        excelformer.py>`_.

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_layers (int): Number of
            :class:`torch_frame.nn.conv.ExcelFormerConv` layers.
        num_heads (int): Number of attention heads used in :class:`DiaM`
        col_stats(dict[str,dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:obj:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`ExcelFormerEncoder()`
            for numerical feature)
        diam_dropout (float, optional): diam_dropout. (default: :obj:`0.0`)
        aium_dropout (float, optional): aium_dropout. (default: :obj:`0.0`)
        residual_dropout (float, optional): residual dropout.
            (default: :obj:`0.0`)
        mixup (str, optional): mixup type.
            :obj:`None`, :obj:`feature`, or :obj:`hidden`.
            (default: :obj:`None`)
        beta (float, optional): Shape parameter for beta distribution to
                calculate shuffle rate in mixup. Only useful when `mixup` is
                not :obj:`None`. (default: :obj:`0.5`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
        num_layers: int,
        num_heads: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        mixup: str | None = None,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        assert mixup in [None, 'feature', 'hidden']

        self.in_channels = in_channels
        self.out_channels = out_channels
        if col_names_dict.keys() != {stype.numerical}:
            raise ValueError("ExcelFormer only accepts numerical "
                             "features.")

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.numerical:
                ExcelFormerEncoder(out_channels, na_strategy=NAStrategy.MEAN)
            }

        self.excelformer_encoder = StypeWiseFeatureEncoder(
            out_channels=self.in_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.excelformer_convs = ModuleList([
            ExcelFormerConv(in_channels, num_cols, num_heads, diam_dropout,
                            aium_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.excelformer_decoder = ExcelFormerDecoder(in_channels,
                                                      out_channels, num_cols)
        self.reset_parameters()
        self.mixup = mixup
        self.beta = beta

    def reset_parameters(self) -> None:
        self.excelformer_encoder.reset_parameters()
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.excelformer_decoder.reset_parameters()

    def forward(
        self,
        tf: TensorFrame,
        mixup_encoded: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings. If
        :obj:`mixup_encoded` is :obj:`True`, it produces the output embeddings
        together with the mixed-up targets in :obj:`self.mixup` manner.

        Args:
            tf (:class:`torch_frame.TensorFrame`): Input :class:`TensorFrame`
                object.
            mixup_encoded (bool): Whether to mixup on encoded numerical
                features, i.e., `FEAT-MIX` and `HIDDEN-MIX`.
                (default: :obj:`False`)

        Returns:
            torch.Tensor | tuple[Tensor, Tensor]: The output embeddings of size
                [batch_size, out_channels]. If :obj:`mixup_encoded` is
                :obj:`True`, return the mixed-up targets of size
                [batch_size, num_classes] as well.
        """
        x, _ = self.excelformer_encoder(tf)
        # FEAT-MIX or HIDDEN-MIX is compatible with `torch.compile`
        if mixup_encoded:
            assert tf.y is not None
            x, y_mixedup = feature_mixup(
                x,
                tf.y,
                num_classes=self.out_channels,
                beta=self.beta,
                mixup_type=self.mixup,
                mi_scores=getattr(tf, 'mi_scores', None),
            )
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        out = self.excelformer_decoder(x)

        if mixup_encoded:
            return out, y_mixedup
        return out
