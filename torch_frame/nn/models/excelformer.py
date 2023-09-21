import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder
from torch_frame.nn.encoder.stype_encoder import ExcelFormerEncoder
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in
        https://arxiv.org/pdf/2301.02819.pdf

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channels dimensionality
        num_cols (int): Number of columns
        num_layers (int): Number of :class:`ExcelFormerConv` layers.
        num_heads (int): Number of attention heads used in :class:`DiaM`
        col_stats (Dict[str, Dict[StatType, Any]]): A dictionary that maps
            column name into stats.
        col_names_dict (Dict[torch_frame.stype, List[str]]): A dictionary that
            maps stype to a list of column names. The column names are sorted
            based on the ordering that appear in :obj:`tensor_frame.x_dict`.
        diam_dropout (float, optional): diam_dropout (default: :obj:`0.0`)
        aium_dropout (float, optional): aium_dropout (default: :obj:`0.0`)
        residual_dropout (float, optional): residual dropout (default: `0.0`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if len(col_names_dict[stype.categorical]) != 0:
            raise ValueError("ExcelFormer only accepts numerical"
                             " features.")
        self.excelformer_encoder = StypeWiseFeatureEncoder(
            out_channels=self.in_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: ExcelFormerEncoder(out_channels)
            },
        )
        self.excelformer_convs = ModuleList([
            ExcelFormerConv(in_channels, num_cols, num_heads, diam_dropout,
                            aium_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.excelformer_decoder = ExcelFormerDecoder(in_channels,
                                                      out_channels, num_cols)
        self.reset_parameters()

    def reset_parameters(self):
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.excelformer_decoder.reset_parameters()

    def feat_mix(self, tf: TensorFrame, beta: float) -> TensorFrame:
        r"""Mixup :obj: Tensor by swaping some feature elements of
                two samples. The shuffle rates for each row is sampled from
                the Beta distribution with shape parameter self.beta.

            Args:
                x (Tensor): Input :obj:`TensorFrame` object.
            Returns:
                tf (TensorFrame): Input :obj:`TensorFrame` object with
                    x_dict mixed with feat_mix.
                y (Tensor): Transformed target [batch_size, num_classes]
                    for classification and [batch_size, 1] for regression.
        """
        # TODO: Modularize this so other models can add mixup easily.
        x = tf.x_dict[stype.numerical]
        B, num_cols = x.shape
        beta_distribution = torch.distributions.beta.Beta(beta, beta)
        shuffle_rates = beta_distribution.sample((B, 1)).to(x.device)
        feat_masks = torch.rand((B, num_cols), device=x.device) < shuffle_rates
        shuffled_sample_ids = torch.randperm(B)

        x_shuffled = x[shuffled_sample_ids]
        x_mixup = feat_masks * x + ~feat_masks * x_shuffled
        tf_mixedup = copy.copy(tf)
        tf_mixedup.x_dict[stype.numerical] = x_mixup

        mix_rates = shuffle_rates[:, 0].float()
        y_shuffled = tf.y[shuffled_sample_ids]
        if tf.y.is_floating_point():
            y_mixedup = mix_rates * tf.y + (1 - mix_rates) * y_shuffled
        else:
            one_hot_y = F.one_hot(tf.y, num_classes=self.out_channels)
            y_shuffled = tf.y[shuffled_sample_ids]
            one_hot_y_shuffled = F.one_hot(y_shuffled,
                                           num_classes=self.out_channels)
            y_mixedup = torch.einsum(
                'i, ij-> ij', mix_rates, one_hot_y) + torch.einsum(
                    'i, ij->ij', (1 - mix_rates), one_hot_y_shuffled)
        return tf_mixedup, y_mixedup

    def forward(
            self, tf: TensorFrame, mixup: bool = False,
            beta: Optional[float] = 0.5
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Transform :obj:`TensorFrame` object into
            output predictions, return feature masks and shuffled
            ids as well if mixup is used.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame` object.
            mixup (bool): True if mixup is used during training otherwise
                False. (default: False)
            beta (float, optional): Shape parameter for beta distribution to
                calculate shuffle rate in mixup. Only useful when mixup is
                true. (default: 0.5)

        Returns:
            x (Tensor): [batch_size, out_channels].
            y_mixedup (Tensor): Output :obj:`Tensor` y_mixedup will be
                returned only when mixup is set to true. The size is
                [batch_size, num_classes] for classification and
                [batch_size, 1] for regression.
        """
        if mixup:
            tf, y_mixedup = self.feat_mix(tf, beta)
        x, _ = self.excelformer_encoder(tf)
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        x = self.excelformer_decoder(x)
        if mixup:
            return x, y_mixedup
        else:
            return x
