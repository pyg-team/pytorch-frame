from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import torch

from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType


def save(tensor_frame: TensorFrame,
         col_stats: Optional[Dict[str, Dict[StatType, Any]]], path: str):
    r"""Save a :obj:`TensorFrame` object and optional :obj:`col_stats`
    to specified path.

    Args:
        tensor_frame(TensorFrame): The :obj:`TensorFrame` object
            to be saved.
        col_stats (dict, optional): The :obj:`col_stats` to be saved.
        path (str): Path to save the :obj:`TensorFrame` object and
            :obj:`col_stats`.
    """
    tf_attrs = [field.name for field in fields(TensorFrame)]
    tf_dict = {attr: getattr(tensor_frame, attr) for attr in tf_attrs}
    torch.save((tf_dict, col_stats), path)


def load(
    path: str, device: Optional[torch.device] = None
) -> Tuple[TensorFrame, Optional[Dict[str, Dict[StatType, Any]]]]:
    r"""Load saved :obj:`TensorFrame` object and optional :obj:`col_stats`
    from a specified path.

    Args:
        path (str): Path to load the :obj:`TensorFrame` object and
            :obj:`col_stats`.
        device (torch.device, optional): Device to load the
            :obj:`TensorFrame` object. (default: :obj:`None`)

    Returns:
        tuple: A tuple of loaded :obj:`TensorFrame` object and
            optional :obj:`col_stats`.
    """
    tf_dict, col_stats = torch.load(path)
    tensor_frame = TensorFrame(**tf_dict)
    tensor_frame.to(device)
    return tensor_frame, col_stats
