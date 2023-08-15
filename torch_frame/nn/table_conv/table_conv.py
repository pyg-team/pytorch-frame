from abc import ABC
from typing import List, Optional

from torch import Module, Tensor
from torch_frame import EmbeddingFrame


class TableConv(Module, ABC):
    r"""Base class for table convolution that transforms EmbeddingFrame into
    another EmbeddingFrame

    Args:
        in_channels (int): Input embedding dimensionality
        out_channels (int): Output embedding dimensionality
        in_col_names (List[str]): Input column names.
        in_col_index (Tensor): Input column index.
        out_col_names (List[str], optional): Output column names. If None, it
            directly re-uses in_col_names.
        out_col_index (Tensor, optional): Output column index. If None, it
            directly re-uses in_col_index.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_col_names: List[str],
        in_col_index: Tensor,
        out_col_names: Optional[List[str]],
        out_col_index: Tensor,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_col_names = in_col_names
        self.in_col_index = in_col_index
        self.out_col_names = out_col_names or in_col_names
        self.out_col_index = out_col_index or in_col_index

    def forward(self, ef: EmbeddingFrame) -> EmbeddingFrame:
        raise NotImplementedError()
