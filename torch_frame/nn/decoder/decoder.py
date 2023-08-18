from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.nn import Module


class Decoder(Module, ABC):
    r"""Base class for decoder that transforms the input column-wise pytorch
    tensor into output tensor on which prediction head is applied."""
    @abstractmethod
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:
        r"""Encode TensorFrame into (x, col_names).

        Args:
            x (torch.Tensor): Input column-wise tensor of shape
                :obj:`[batch_size, num_cols, hidden_channels]`.
            args (Any): Extra arguments.
            kwargs (Any): Extra keyward arguments.
        """
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass
