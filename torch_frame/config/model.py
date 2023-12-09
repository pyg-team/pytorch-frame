from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from torch_frame.typing import TensorData


@dataclass
class ModelConfig:
    r"""Learnable model that maps :class:`TensorData` into embeddings of each
        row.

    Args:
        model (callable): A callable model that takes TensorData of shape
            :obj:`[batch_size, 1, *]` as input and outputs embeddings of shape
            :obj:`[batch_size, 1, out_channels]`.
        out_channels (int): Model output channels.

    """
    model: Callable[[TensorData], Tensor]
    out_channels: int
