from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

from torch_frame import TensorFrame
from torch_frame.data.stats import StatType


class BaseTransform(ABC):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`TensorFrame`
    """
    def __init__(self):
        self._transformed_stats: dict[str, dict[StatType, Any]] | None = None

    def __call__(self, tf: TensorFrame) -> TensorFrame:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(tf))

    @abstractmethod
    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame`.

        Returns:
            TensorFrame: Input :class:`TensorFrame` after transform.
        """
        return tf

    @property
    def transformed_stats(self) -> dict[str, dict[StatType, Any]]:
        r"""The column stats after the transform.

        Returns:
            transformed_stats (Dict[str, Dict[StatType, Any]]):
                Transformed column stats. The :class:`TensorFrame` object might
                be modified by the transform, so the returned
                :obj:`transformed_stats` would contain the column stats of the
                modified :class:`TensorFrame` object.
        """
        if self._transformed_stats is None:
            raise ValueError("Transformed column stats is not computed yet. "
                             "Please run necessary functions to compute this"
                             " first.")
        return self._transformed_stats

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
