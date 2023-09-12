import copy
from abc import ABC
from typing import Union

from torch import Tensor
from torch.nn import Module

from torch_frame import TensorFrame


class BaseTransform(ABC, Module):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`TensorFrame`"""
    def __call__(self, tf: Union[TensorFrame,
                                 Tensor]) -> Union[TensorFrame, Tensor]:
        # Shallow-copy the data so that we prevent in-place data modification.
        if isinstance(tf, TensorFrame):
            return self.forward(copy.copy(tf))
        else:
            return self.forward(tf)

    def forward(self, tf: Union[TensorFrame,
                                Tensor]) -> Union[TensorFrame, Tensor]:
        r"""Process TensorFrame obj into another TensorFrame obj.
        Args:
            tf (TensorFrame): Input :obj:`TensorFrame`.
        Returns:
            tf (TensorFrame): Input :obj:`TensorFrame` after transform.
        """
        return tf

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
