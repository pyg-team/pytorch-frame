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
from torch_frame.typing import NAStrategy, TensorData


def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: float = 0.5,
    mixup_type: str = 'ordinary',
    mi_scores: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Mixup :obj: input numerical feature tensor :obj:`x` by swapping some
    feature elements of two shuffled sample samples. The shuffle rates for
    each row is sampled from the Beta distribution. The target `y` is also
    linearly mixed up.

    Args:
        x (Tensor): The input numerical feature.
        y (Tensor): The target.
        num_classes (int): Number of classes.
        beta (float): The concentration parameter of the Beta distribution.
        mixup_type (str): The mixup methods.
        mi_scores (Tensor): Mutual information scores for FEAT-MIX.

    Returns:
        x_mixedup (Tensor): The mixedup numerical feature.
        y_mixedup (Tensor): Transformed target of size
            :obj:`[batch_size, num_classes]`
    """
    assert num_classes > 0
    assert mixup_type in ['ordinary', 'feature', 'hidden']
    beta = torch.tensor(beta, dtype=torch.float32, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample(torch.Size((len(x), 1)))
    shuffled_idx = torch.randperm(len(x), device=x.device)
    if mixup_type == 'ordinary':
        # Soft masks
        feat_masks = shuffle_rates
        x_mixedup = feat_masks * x + (1 - feat_masks) * x[shuffled_idx]
        lmbd = feat_masks
    else:
        assert x.ndim == 3, 'FEAT-MIX or HIDDEN-MIX requires encoded features'
        if mixup_type == 'feature':
            assert mi_scores is not None
            mi_scores = mi_scores.to(x.device)
            # Hard masks (feature dimension)
            feat_masks = torch.rand(torch.Size((x.shape[0], x.shape[1])),
                                    device=x.device) > shuffle_rates
            l1_norm_mi = mi_scores / mi_scores.sum()
            lmbd = torch.sum(
                l1_norm_mi.unsqueeze(0) * feat_masks, dim=1, keepdim=True)
            feat_masks = feat_masks.unsqueeze(2)
        else:
            # Hard masks (hidden dimension)
            feat_masks = torch.rand(torch.Size((x.shape[0], x.shape[2])),
                                    device=x.device) < shuffle_rates
            feat_masks = feat_masks.unsqueeze(1)
            lmbd = shuffle_rates
        x_mixedup = feat_masks * x + ~feat_masks * x[shuffled_idx]

    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        lmbd = lmbd.squeeze(1)
        y_mixedup = lmbd * y + (1 - lmbd) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (lmbd * one_hot_y + (1 - lmbd) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    ExcelFormer first converts the categorical features with a target
    statistics encoder into numerical features. Then it sorts the
    numerical features with mutual information sort. For categorical
    features, they are converted to numerical ones with CatBoostEncoder.

    .. note::

        For an example of using ExcelFormer, see `examples/excelformer.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        excelfromer.py>`_.

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
            (default: :obj:`'none'`, i.e., no mixup)
        beta (float, optional): Shape parameter for beta distribution to
                calculate shuffle rate in mixup. Only useful when `mixup` is
                not `'none'` or during training. (default: :obj:`0.5`)
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
        mixup: str = 'none',
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        assert mixup in ['none', 'ordinary', 'feature', 'hidden']

        self.in_channels = in_channels
        self.out_channels = out_channels
        if col_names_dict.keys() != {stype.numerical}:
            raise ValueError("ExcelFormer only accepts numerical"
                             " features.")

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
        mixup_encoded: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings.

        Args:
            tf (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.
            mixup_encoded (bool):
                Whether to mixup on encoded numerical features, i.e.,
                FEAT-MIX and HIDDEN-MIX.

        Returns:
            torch.Tensor: The output embeddings of size
                [batch_size, out_channels].
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

    def forward_mixup(self, tf: TensorFrame) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings. If
        `self.mixup` is not `'none'`, it produces the output embeddings
        together with the mixed-up targets.

        Args:
            tf (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: The output embeddings of size
                [batch_size, out_channels].
        """
        # Includes non-matrix operations required in ordinary mixup
        # that are incompatible with `torch.compile`
        if stype.numerical not in tf.feat_dict:
            raise ValueError(
                "Excelformer only takes in numerical features, but the input "
                "TensorFrame object does not have numerical features.")
        assert tf.y is not None
        # Ordinary mixup on non-encoded numerical features
        if self.mixup == 'ordinary':
            numerical_feat = tf.feat_dict[stype.numerical]
            assert isinstance(numerical_feat, Tensor)
            # Mixup numerical features brefore embedding (naive)
            x_mixedup, y_mixedup = feature_mixup(
                numerical_feat,
                tf.y,
                num_classes=self.out_channels,
                beta=self.beta,
                mixup_type=self.mixup,
            )
            # Create a new `feat_dict`, where stype.numerical is swapped with
            # mixed up feature.
            feat_dict: dict[stype, TensorData] = {}
            for stype_name, x in tf.feat_dict.items():
                if stype_name == stype.numerical:
                    feat_dict[stype_name] = x_mixedup
                else:
                    feat_dict[stype_name] = x
            tf_mixedup = TensorFrame(feat_dict, tf.col_names_dict, tf.y)

            # Call Excelformer forward function
            out_mixedup = self(tf_mixedup)
            return out_mixedup, y_mixedup
        # FEAT-MIX or HIDDEN-MIX
        elif self.mixup in ['feature', 'hidden']:
            return self(tf, True)
        # No mixup (self.mixup == `'none'`)
        out = self(tf)
        if self.out_channels > 1:
            one_hot_y = F.one_hot(tf.y, num_classes=self.out_channels)
            return out, one_hot_y
        return out, tf.y
