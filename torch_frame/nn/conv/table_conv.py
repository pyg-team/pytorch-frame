from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.nn import Module


class TableConv(Module, ABC):
    r"""Base class for table convolution that transforms the input column-wise
    pytorch tensor.
    """
    @abstractmethod
    def forward(self, x: Tensor, *argv: Any, **kwargs: Any) -> Any:
        r"""Encode TensorFrame into (x, col_names).
        Args:
            x (Tensor): Input column-wise tensor of shape
                :obj:`[batch_size, num_cols, hidden_channels]`.
            argv (Any): Extra argument values.
            kwargs (Any): Extra keyward arguments.
        """
        raise NotImplementedError
