import copy
from abc import ABC, abstractmethod

from torch_frame import TensorFrame


class BaseTransform(ABC):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`TensorFrame`"""
    def __call__(self, tf: TensorFrame) -> TensorFrame:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(tf))

    @abstractmethod
    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.
        Args:
            tf (TensorFrame): Input :obj:`TensorFrame`.
        Returns:
            tf (TensorFrame): Input :obj:`TensorFrame` after transform.
        """
        return tf

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
