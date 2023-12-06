from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from torch_frame.typing import TensorData


@dataclass
class TextModelConfig:
    r"""Text model that maps a dictionary of :class:`MultiNestedTensor`
    into embeddings of each row.

    Args:
        model (callable): A callable text model that takes a dictionary
            of :class:`MultiNestedTensor` as input and outputs embeddings
            for each row in the :class:`MultiNestedTensor`.
        out_channels (int): Text model output channels.

    """
    model: Callable[[TensorData], Tensor]
    out_channels: int
