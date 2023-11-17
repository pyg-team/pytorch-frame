from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

import torch_frame
from torch_frame.data import MultiNestedTensor, TensorFrame
from torch_frame.data.multi_tensor import _MultiTensor
from torch_frame.data.stats import StatType
from torch_frame.typing import TensorData


def serialize_feat_dict(
    feat_dict: Dict[torch_frame.stype, TensorData]
) -> Dict[torch_frame.stype, Any]:
    feat_serialized_dict = {}
    for stype, feat in feat_dict.items():
        # TODO: Add stype.use_multi_tensor and use the same code path for
        # stype.embedding.
        if stype.use_multi_nested_tensor:
            assert isinstance(feat, _MultiTensor)
            feat_serialized = feat.to_dict()
        elif stype.use_dict_multi_nested_tensor:
            feat_serialized = {}
            assert isinstance(feat, dict)
            for name, f in feat.items():
                assert isinstance(f, MultiNestedTensor)
                feat_serialized[name] = f.to_dict()
        else:
            assert isinstance(feat, Tensor)
            feat_serialized = feat
        feat_serialized_dict[stype] = feat_serialized
    return feat_serialized_dict


def deserialize_feat_dict(
    feat_serialized_dict: Dict[torch_frame.stype, Any]
) -> Dict[torch_frame.stype, TensorData]:
    feat_dict = {}
    for stype, feat_serialized in feat_serialized_dict.items():
        if stype.use_multi_nested_tensor:
            feat = MultiNestedTensor(**feat_serialized)
        elif stype.use_dict_multi_nested_tensor:
            feat = {}
            for name, f_serialized in feat_serialized.items():
                feat[name] = MultiNestedTensor(**f_serialized)
        else:
            assert isinstance(feat_serialized, Tensor)
            feat = feat_serialized
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
        'feat_serialized_dict': serialize_feat_dict(tensor_frame.feat_dict),
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
        tf_dict.pop('feat_serialized_dict'))
    tensor_frame = TensorFrame(**tf_dict)
    tensor_frame.to(device)
    return tensor_frame, col_stats
