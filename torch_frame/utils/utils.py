from typing import Any, Dict, Optional, Tuple

import torch

from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType


def save_tf(path: str, tensor_frame,
            col_stats: Optional[Dict[str, Dict[StatType, Any]]] = None):
    torch.save((tensor_frame, col_stats), path)


def load_tf(
    path: str, device: Optional[torch.device] = None
) -> Tuple[TensorFrame, Optional[Dict[str, Dict[StatType, Any]]]]:
    tensor_frame, col_stats = torch.load(path)
    tensor_frame.to(device)
    return tensor_frame, col_stats
