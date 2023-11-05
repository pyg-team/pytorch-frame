from typing import Any, Dict, List, Optional

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential, LSTM
from torch.nn.modules.module import Module
import torch
import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.nn.conv import FTTransformerConvs


class FTTransformerLSTM(Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Numner of layers.  (default: :obj:`3`)
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:obj:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (Optional[Dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            Dictionary containing encoder type per column statistics (default:
            :obj:`None`, will call
            :class:`torch_frame.nn.encoder.EmbeddingEncoder()` for categorial
            feature and :class:`torch_frame.nn.encoder.LinearEncoder()`
            for numerical feature)
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
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        self.decoder = Sequential(
            LayerNorm(channels + lstm_out_channels),
            ReLU(),
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
        self.backbone.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame, tf_price: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            x (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        B = x.size(0)
        x, x_cls = self.backbone(x)
        x_price = tf_price.feat_dict[stype.numerical]
        x_price = (x_price - self.mean) / self.std
        mask = torch.isnan(x_price)
        x_price[mask] = 0
        #print("shape of x_price ", x_price.shape)
        x_price = self.lstm(x_price)
        x_price = x_price.repeat(B, 1)
        x_price.zero_()
        #print("x_price is ", x_price)
        x_cls = torch.cat([x_cls, x_price], dim=1)
        out = self.decoder(x_cls)
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

