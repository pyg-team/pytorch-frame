from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleDict

import torch_frame
from torch_frame import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import FeatureEncoder
from torch_frame.nn.encoder.stype_encoder import StypeEncoder


class StypeWiseFeatureEncoder(FeatureEncoder):
    r"""Feature encoder that transforms each stype tensor into embeddings and
    performs the final concatenation.

    Args:
        out_channels (int): Output dimensionality.
        col_stats
            (Dict[str, Dict[:class:`torch_frame.data.stats.StatType`, Any]]):
            A dictionary that maps column name into stats. Available as
            :obj:`dataset.col_stats`.
        col_names_dict (Dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (Dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            A dictionary that maps :class:`torch_frame.stype` into
            :class:`torch_frame.nn.encoder.StypeEncoder` class.
    """
    def __init__(
        self,
        out_channels: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict: Dict[torch_frame.stype, StypeEncoder],
    ):
        super().__init__()

        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.encoder_dict = ModuleDict()
        for stype, stype_encoder in stype_encoder_dict.items():
            if stype not in stype_encoder.supported_stypes:
                raise ValueError(
                    f"{stype_encoder} does not support encoding {stype}.")

            if stype in col_names_dict:
                stats_list = [
                    self.col_stats[col_name]
                    for col_name in self.col_names_dict[stype]
                ]
                # Set lazy attributes
                stype_encoder.stype = stype
                stype_encoder.out_channels = out_channels
                stype_encoder.stats_list = stats_list
                self.encoder_dict[stype.value] = stype_encoder

    def forward(self, tf: TensorFrame) -> Tuple[Tensor, List[str]]:
        col_names = []
        xs = []
        for stype in tf.stypes:
            feat = tf.feat_dict[stype]
            x = self.encoder_dict[stype.value](feat)
            xs.append(x)
            col_names.extend(self.col_names_dict[stype])
        x = torch.cat(xs, dim=1)
        return x, col_names
