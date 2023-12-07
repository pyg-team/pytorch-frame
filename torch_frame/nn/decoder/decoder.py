from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.nn import Module


class Decoder(Module, ABC):
    r"""Base class for decoder that transforms the input column-wise PyTorch
    tensor into output tensor on which prediction head is applied.
    """
    @abstractmethod
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:
        r"""Decode :obj:`x` of shape :obj:`[batch_size, num_cols, channels]`
        into an output tensor of shape :obj:`[batch_size, out_channels]`.

        Args:
            x (torch.Tensor): Input column-wise tensor of shape
                :obj:`[batch_size, num_cols, hidden_channels]`.
            args (Any): Extra arguments.
            kwargs (Any): Extra keyward arguments.
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
