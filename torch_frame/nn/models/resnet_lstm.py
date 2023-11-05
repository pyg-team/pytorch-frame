from typing import Any, Dict, List, Optional

from torch import Tensor
import torch
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
    LSTM,
    SELU,
)
from torch.nn.modules.module import Module

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.typing import NAStrategy


class FCResidualBlock(Module):
    r"""Fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str, optional): The type of normalization to use.
            :obj:`batchnorm`, :class:`torch.nn.LayerNorm`, or :obj:`None`.
            (default: :class:`torch.nn.LayerNorm`)
        dropout_prob (float): The dropout probability (default: `0.0`, i.e.,
            no dropout).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = 'layernorm',
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_prob)

        if normalization == 'batchnorm':
            self.norm1 = BatchNorm1d(out_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == 'layernorm':
            self.norm1 = LayerNorm(out_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        if in_channels != out_channels:
            self.shortcut = Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.shortcut is not None:
            self.shortcut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = self.lin1(x)
        out = self.norm1(out) if self.norm1 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out) if self.norm2 else out
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            x = self.shortcut(x)

        out += x
        out = self.relu(out)

        return out


class ResNetLSTM(Module):
    r"""The ResNet model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using ResNet, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): The number of channels in the backbone layers.
        out_channels (int): The number of output channels in the decoder.
        num_layers (int): The number of layers in the backbone.
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (Optional[Dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            Dictionary containing encoder type per column statistics
            (default: :obj:`None`, :obj:`EmbeddingEncoder()` for categorial
            feature and :obj:`LinearEncoder()` for numerical feature)
        normalization (str, optional): The type of normalization to use.
            :obj:`batchnorm`, :obj:`layernorm`, or :obj:`None`.
            (default: :obj:`layernorm`)
        dropout_prob (float): The dropout probability (default: `0.2`).
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        lstm_channels: int,
        lstm_hidden: int,
        lstm_num_layers: int,
        lstm_out_channels: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        lstm_col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        lstm_col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict: Optional[Dict[torch_frame.stype,
                                          StypeEncoder]] = None,
        normalization: Optional[str] = 'layernorm',
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        in_channels = channels * (len(col_stats) - 1)
        self.backbone = Sequential(*[
            FCResidualBlock(in_channels if i == 0 else channels, channels,
                            normalization=normalization,
                            dropout_prob=dropout_prob)
            for i in range(num_layers)
        ])

        self.decoder = Sequential(
            LayerNorm(channels + lstm_out_channels),
            BatchNorm1d(channels + lstm_out_channels), SELU(),
            Linear(channels + lstm_out_channels, out_channels),
        )
        lstm_col_stats_list = [lstm_col_stats[name] for name in lstm_col_names_dict[stype.numerical]]
        lstm_mean = torch.tensor(
            [stats[StatType.MEAN] for stats in lstm_col_stats_list])
        lstm_std = torch.tensor([stats[StatType.STD]
                            for stats in lstm_col_stats_list]) + 1e-6
        self.register_buffer('mean', lstm_mean)
        self.register_buffer('std', lstm_std)
        
        self.lstm = MultivariateLSTM(
                 lstm_channels,
                 lstm_hidden,
                 lstm_num_layers,
                 lstm_out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        self.decoder[0].reset_parameters()
        self.decoder[-1].reset_parameters()
        self.lstm.reset_parameters()

    def forward(self, tf: TensorFrame, tf_price: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            x (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        #print("after encoder x ", x)
        B = x.size(0)
        # Flattening the encoder output
        x = x.view(x.size(0), -1)

        x = self.backbone(x)
        #print(tf_price.feat_dict[stype.numerical].shape)
        x_price = tf_price.feat_dict[stype.numerical]
        x_price = (x_price - self.mean) / self.std
        mask = torch.isnan(x_price)
        x_price[mask] = 0
        #print("shape of x_price ", x_price.shape)
        x_price = self.lstm(x_price)
        x_price = x_price.repeat(B, 1)
        #print("x_price is ", x_price)
        x = torch.cat([x, x_price], dim=1)
        print(x)
        #print("x is ", x)
        #print("shape of x before decoder ", x.shape)
        out = self.decoder(x)
        #import pdb
        #pdb.set_trace()
        #print(out)
        return out


class MultivariateLSTM(Module):
    def __init__(self,
                 lstm_channels: int,
                 lstm_hidden: int,
                 lstm_num_layers: int,
                 lstm_out_channels: int):
        super().__init__()
        self.lstm = LSTM(lstm_channels, lstm_hidden, lstm_num_layers, batch_first=True)
        self.fc = Linear(lstm_hidden, lstm_out_channels)

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.fc.reset_parameters()
    
    def forward(self, x: Tensor):
        x = x.unsqueeze(0)
        #print("shape of x before lstm", x.shape)
        out, (hn, cn) = self.lstm(x)
        #print("shape of x after lstm", out.shape)
        out = torch.tanh(out)
        out = self.fc(out[:, -1, :])
        #print("shape of out ", out.shape)
        return out

