from typing import Any, Dict, Optional, Tuple

import torch

from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType


def save_tf(path: str, tensor_frame: TensorFrame,
            col_stats: Optional[Dict[str, Dict[StatType, Any]]] = None):
    r"""Save a :obj:`TensorFrame` object and optional :obj:`col_stats`
    to specified path.

    Args:
        path (str): Path to save the :obj:`TensorFrame` object and
            :obj:`col_stats`.
        tensor_frame(TensorFrame): The :obj:`TensorFrame` object
            to be saved.
        col_stats (dict, optional): The :obj:`col_stats` to be saved.
            (default: :obj:`None`)
    """
    torch.save((tensor_frame, col_stats), path)


def load_tf(
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
    tensor_frame, col_stats = torch.load(path)
    tensor_frame.to(device)
    return tensor_frame, col_stats
