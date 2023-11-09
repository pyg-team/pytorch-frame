from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from torch_frame.data import MultiNestedTensor, TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.typing import TensorData


def serialize_feat_dict(feat_dict: Dict[str, TensorData]) -> Dict[str, Any]:
    feat_seriealized_dict = {}
    for stype, feat in feat_dict.items():
        if stype.use_multi_nested_tensor:
            assert isinstance(feat, MultiNestedTensor)
            feat_seriealized = asdict(feat)
        elif stype.use_dict_multi_nested_tensor:
            feat_seriealized = {}
            assert isinstance(feat, dict)
            for name, f in feat.items():
                assert isinstance(f, MultiNestedTensor)
                feat_seriealized[name] = asdict(f)
        else:
            assert isinstance(feat, Tensor)
            feat_seriealized = feat
        feat_seriealized_dict[stype] = feat_seriealized
    return feat_seriealized_dict


def deserialize_feat_dict(
        feat_serialized_dict: Dict[str, Any]) -> Dict[str, TensorData]:
    feat_dict = {}
    for stype, feat_seriealized in feat_serialized_dict.items():
        if stype.use_multi_nested_tensor:
            feat = MultiNestedTensor(**feat_seriealized)
        elif stype.use_dict_multi_nested_tensor:
            feat = {}
            for name, f_serialized in feat_seriealized.items():
                feat[name] = MultiNestedTensor(**f_serialized)
        else:
            assert isinstance(feat_seriealized, Tensor)
            feat = feat_seriealized
        feat_dict[stype] = feat
    return feat_dict


def save(tensor_frame: TensorFrame,
         col_stats: Optional[Dict[str, Dict[StatType, Any]]], path: str):
    r"""Save a :class:`TensorFrame` object and optional :obj:`col_stats`
    to specified path.

    Args:
        tensor_frame(TensorFrame): The :class:`TensorFrame` object
            to be saved.
        col_stats (dict, optional): The :obj:`col_stats` to be saved.
        path (str): Path to save the :class:`TensorFrame` object and
            :obj:`col_stats`.
    """

    tf_dict = {
        'y': tensor_frame.y,
        'col_names_dict': tensor_frame.col_names_dict,
        'feat_seriealized_dict': serialize_feat_dict(tensor_frame.feat_dict),
    }
    torch.save((tf_dict, col_stats), path)


def load(
    path: str, device: Optional[torch.device] = None
) -> Tuple[TensorFrame, Optional[Dict[str, Dict[StatType, Any]]]]:
    r"""Load saved :class:`TensorFrame` object and optional :obj:`col_stats`
    from a specified path.

    Args:
        path (str): Path to load the :class:`TensorFrame` object and
            :obj:`col_stats`.
        device (torch.device, optional): Device to load the
            :class:`TensorFrame` object. (default: :obj:`None`)

    Returns:
        tuple: A tuple of loaded :class:`TensorFrame` object and
            optional :obj:`col_stats`.
    """
    tf_dict, col_stats = torch.load(path)
    tf_dict['feat_dict'] = deserialize_feat_dict(
        tf_dict.pop('feat_seriealized_dict'))
    tensor_frame = TensorFrame(**tf_dict)
    tensor_frame.to(device)
    return tensor_frame, col_stats
