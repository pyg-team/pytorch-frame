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


def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: int = 0.5,
) -> TensorFrame:
    r"""Mixup :obj: input numerical feature tensor :obj:`x` by swaping some
    feature elements of two shuffled sample samples. The shuffle rates for
    each row is sampled from the Beta distribution. The target `y` is also
    linearly mixed up.

    Args:
        x (Tensor): The input numerical feature.
        y (Tensor): The target.
        num_classes (int): Number of classes.
        beta (float): The concentration parameter of the Beta distribution.

    Returns:
        x_mixedup (Tensor): The mixedup numerical feature.
        y_mixedup (Tensor): Transformed target of size
            [batch_size, num_classes]
    """
    assert num_classes > 0

    beta = torch.tensor(beta, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample((len(x), 1))
    feat_masks = torch.rand(x.shape, device=x.device) < shuffle_rates
    shuffled_idx = torch.randperm(len(x), device=x.device)
    x_mixedup = feat_masks * x + ~feat_masks * x[shuffled_idx]

    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        shuffle_rates = shuffle_rates.view(-1, )
        y_mixedup = shuffle_rates * y + (1 - shuffle_rates) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (shuffle_rates * one_hot_y +
                     (1 - shuffle_rates) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup


class ExcelFormer(Module):
    r"""The ExcelFormer model introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    ExcelFormer first converts the categorical features with a target
    statistics encoder into numerical features. Then it sorts the
    numerical features with mutual information sort. So the model
    itself limits to numerical features.

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
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:obj:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        diam_dropout (float, optional): diam_dropout. (default: :obj:`0.0`)
        aium_dropout (float, optional): aium_dropout. (default: :obj:`0.0`)
        residual_dropout (float, optional): residual dropout.
            (default: :obj:`0.0`)
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
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        if col_names_dict.keys() != set([stype.numerical]):
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
        self.excelformer_encoder.reset_parameters()
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.excelformer_decoder.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transform :obj:`TensorFrame` object into output embeddings.

        Args:
            tf (:class:`torch_frame.TensorFrame`):
                Input :obj:`TensorFrame` object.

        Returns:
            torch.Tensor: The output embeddings of size
                [batch_size, out_channels].
        """
        if stype.numerical not in tf.feat_dict or len(
                tf.feat_dict[stype.numerical]) == 0:
            raise ValueError(
                "Excelformer only takes in numerical features, but the input "
                "TensorFrame object does not have numerical features.")
        x, _ = self.excelformer_encoder(tf)
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)
        out = self.excelformer_decoder(x)
        return out

    def forward_mixup(
        self,
        tf: TensorFrame,
        beta: Optional[float] = 0.5,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Transform :obj:`TensorFrame` object into output embeddings. If
        `mixup` is :obj:`True`, it produces the output embeddings together with
        the mixed-up targets.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame` object.
            beta (float, optional): Shape parameter for beta distribution to
                calculate shuffle rate in mixup. Only useful when mixup is
                true. (default: :obj:`0.5`)

        Returns:
            (torch.Tensor, torch.Tensor): The first :obj:`Tensor` is the mixed
            up output embeddings of size [batch_size, out_channels].
            The second :obj:`Tensor` y_mixedup will be returned only when
            mixup is set to true. The size is [batch_size, num_classes] for
            classification and [batch_size, 1] for regression.
        """
        # Mixup numerical features
        x_mixedup, y_mixedup = feature_mixup(
            tf.feat_dict[stype.numerical],
            tf.y,
            num_classes=self.out_channels,
            beta=beta,
        )

        # Create a new `feat_dict`, where stype.numerical is swapped with
        # mixed up feature.
        feat_dict: Dict[stype, Tensor] = {}
        for stype_name, x in tf.feat_dict.items():
            if stype_name == stype.numerical:
                feat_dict[stype_name] = x_mixedup
            else:
                feat_dict[stype_name] = x
        tf_mixedup = TensorFrame(feat_dict, tf.col_names_dict, tf.y)

        # Call Excelformer forward function
        out_mixedup = self(tf_mixedup)

        return out_mixedup, y_mixedup
