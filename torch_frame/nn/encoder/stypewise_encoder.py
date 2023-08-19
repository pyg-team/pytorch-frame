from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleDict

import torch_frame
from torch_frame import Stype, TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import FeatureEncoder
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)

DEFAULT_STYPE_ENCODER_DICT: Dict[Stype, StypeEncoder] = {
    Stype.categorical: EmbeddingEncoder(),
    Stype.numerical: LinearEncoder(),
}


class StypeWiseFeatureEncoder(FeatureEncoder):
    r"""Feature encoder that transforms each Stype tensor into embeddings and
    performs the final concatenatenation.

    Args:
        out_channels (int): Output dimensionality
        col_stats (Dict[str, Dict[StatType, Any]]): A dictionary that maps
            column name into stats. Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[torch_frame.Stype, List[str]]): A dictionary that
            maps stype to a list of column names. The column names are sorted
            based on the ordering that appear in :obj:`tensor_frame.x_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict (Dict[stype, StypeEncoder]): A dictionary that maps
            stype into :class:`StypeEncoder` class.
    """
    def __init__(
        self,
        out_channels: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.Stype, List[str]],
        stype_encoder_dict: Dict[Stype,
                                 StypeEncoder] = DEFAULT_STYPE_ENCODER_DICT,
    ):
        super().__init__()

        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.encoder_dict = ModuleDict()
        for stype, stype_encoder in stype_encoder_dict.items():
            if stype not in stype_encoder.stype_supported:
                raise ValueError(
                    f"{stype_encoder} does not support encoding {stype}.")

            if stype in col_names_dict:
                col_names = col_names_dict[stype]
                stats_list = [
                    self.col_stats[col_name] for col_name in col_names
                ]
                # Set LAZY_ATTRS
                stype_encoder.out_channels = out_channels
                stype_encoder.stats_list = stats_list
                self.encoder_dict[stype.value] = stype_encoder

    def forward(self, tf: TensorFrame) -> Tuple[Tensor, List[str]]:
        col_names = []
        xs = []
        for stype in tf.stype_list:
            x = tf.x_dict[stype]
            x = self.encoder_dict[stype.value](x)
            xs.append(x)
            col_names.extend(self.col_names_dict[stype])
        x = torch.cat(xs, dim=1)
        return x, col_names
