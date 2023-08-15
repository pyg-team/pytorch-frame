from abc import ABC
from typing import List

from torch import Module, Tensor
from torch_frame import EmbeddingFrame


class Decoder(Module, ABC):
    r"""Base class for decoder that takes EmbeddingFrame as input and outputs
    tensor over which a prediction head can be applied.

    Args:
        in_channels (int): Input embedding dimensionality
        in_col_names (List[str]): Input column names.
        in_col_index (Tensor): Input column index.
    """
    def __init__(
        self,
        in_channels: int,
        in_col_names: List[str],
        in_col_index: Tensor,
    ):
        self.in_channels = in_channels
        self.in_col_names = in_col_names
        self.in_col_index = in_col_index

    def forward(self, ef: EmbeddingFrame) -> Tensor:
        raise NotImplementedError()
