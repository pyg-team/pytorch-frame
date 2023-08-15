from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor


@dataclass
class EmbeddingFrame:
    r"""EmbeddingFrame is the hidden embeddings processed by multiple layers of
    TableConv. It is produced by torch_frame.Encoder that takes TensorFrame as
    input. In general, TableConv produces EmbeddingFrame whose col_names is
    different from input.
    """
    # 3-dim tensor of shape [batch_size, num_cols, hidden_channels]
    x: Tensor
    # List of column names
    col_names: List[str]
    # i-th col_names corresponds to x[:, col_index[i]:col_index[i+1], :]
    # If None, we assume x[:, i, :] corresponds to i-th col_names
    col_index: Optional[Tensor] = None

    def __post_init__(self):
        if self.x.ndim != 3:
            raise ValueError("x must be 3-dim tensor")
        num_cols = len(self.col_names)
        if self.col_index is None:
            self.col_index = torch.arange(num_cols + 1, device=self.x.device)
        if len(self.col_index) != num_cols + 1:
            raise ValueError(
                "The length of col_index must be number of columns plus 1.")
        if not (self.col_index[1:] - self.col_index[:-1] > 0).all():
            raise ValueError("col_index must be monotonically increasing.")
        if self.col_index[-1] != self.x.size(1):
            raise ValueError(f"col_index[-1] must be equal to x.size(1)")

    def __len__(self) -> int:
        return self.x.size(0)

    @property
    def num_cols(self) -> int:
        return len(self.col_names)

    @property
    def dim(self) -> int:
        r"""The embedding dimensionality"""
        return self.x.size(2)

    # TODO implement concat / column selection utils for list for
    # EmbeddingFrame.
