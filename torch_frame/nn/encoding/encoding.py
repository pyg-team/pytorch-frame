from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class Encoding(Module, ABC):
    r"""Base class for feature encoding that transforms
    input :obj:`Tensor` into :obj:`Tensor` encoding.
    """
    def forward(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.isnan().any():
            raise ValueError(f"'{self.__class__.__name__}' cannot "
                             "handle tensors with nans.")
        return self._forward(input_tensor)

    @abstractmethod
    def _forward(self, input_tensor: Tensor) -> Tensor:
        raise NotImplementedError
