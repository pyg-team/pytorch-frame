from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from torch_frame.typing import TensorData


@dataclass
class ModelConfig:
    r"""Learnable model that maps per-column :class:`TensorData` into
    row embeddings.

    Args:
        model (callable): A callable model that takes TensorData of shape
            :obj:`[batch_size, 1, *]` as input and outputs embeddings of shape
            :obj:`[batch_size, 1, out_channels]`.
        out_channels (int): Model output channels.

    """
    model: Callable[[TensorData], Tensor]
    out_channels: int
