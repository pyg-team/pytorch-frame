from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class Encoding(Module, ABC):
    r"""Base class for feature encoding that transforms
    input :obj:`Tensor` into :obj:`Tensor` encoding.
    """
    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.isnan().any():
            raise ValueError(f"'{self.__class__.__name__}' cannot "
                             "handle tensors with nans.")
        return self._forward(tensor)

    @abstractmethod
    def _forward(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError
