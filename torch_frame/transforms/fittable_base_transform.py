import copy
from abc import abstractmethod

from torch_frame import TensorFrame
from torch_frame.transforms import BaseTransform


class FittableBaseTransform(BaseTransform):
    r"""An abstract base class for writing fittable transforms.

    Fittable transforms must be fitted on training data before transform.
    """
    def __init__(self):
        self._is_fitted: bool = False

    def __call__(self, tf: TensorFrame) -> TensorFrame:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(tf))

    @property
    def is_fitted(self) -> bool:
        r"""Whether the transform is already fitted."""
        return self._is_fitted

    def fit(self, tf: TensorFrame) -> TensorFrame:
        r"""Fit the transform with train data.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame` representing train data.
        """
        self._fit(tf)
        self._is_fitted = True

    def forward(self, tf: TensorFrame) -> TensorFrame:
        if not self.is_fitted:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fitted ."
                             f"Please run `fit()` first before attempting to "
                             f"transform the TensorFrame.")

        return self._forward(tf)

    @abstractmethod
    def _fit(self, tf: TensorFrame):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, tf: TensorFrame) -> TensorFrame:
        raise NotImplementedError
