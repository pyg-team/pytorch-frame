from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.nn import Module


class TableConv(Module, ABC):
    r"""Base class for table convolution that transforms the input column-wise
    pytorch tensor.
    """
    @abstractmethod
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:
        r"""Process column-wise 3-dimensional tensor into another column-wise
        3-dimensional tensor.

        Args:
            x (torch.Tensor): Input column-wise tensor of shape
                :obj:`[batch_size, num_cols, hidden_channels]`.
            args (Any): Extra arguments.
            kwargs (Any): Extra keyword arguments.
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
