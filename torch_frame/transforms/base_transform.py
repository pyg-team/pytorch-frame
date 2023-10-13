import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch_frame import TensorFrame
from torch_frame.data.stats import StatType


class BaseTransform(ABC):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`TensorFrame`"""
    def __init__(self):
        self._transformed_stats: Optional[Dict[str, Dict[StatType,
                                                         Any]]] = None

    def __call__(self, tf: TensorFrame) -> TensorFrame:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(tf))

    @abstractmethod
    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.
        Args:
            tf (TensorFrame): Input :obj:`TensorFrame`.
        Returns:
            TensorFrame: Input :obj:`TensorFrame` after transform.
        """
        return tf

    @property
    def transformed_stats(self) -> Dict[str, Dict[StatType, Any]]:
        r"""The column stats after the transform.

        Returns:
            transformed_stats (Dict[str, Dict[StatType, Any]]):
                Transformed column stats. The :obj:`TensorFrame` might be
                modified by the transform, so the returned transformed_stats
                would contain the column stats of the modified
                :obj:`TensorFrame`.
        """
        if self._transformed_stats is None:
            raise ValueError("Transformed column stats is not computed yet. "
                             "Please run necessary functions to compute this"
                             " first.")
        return self._transformed_stats

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
