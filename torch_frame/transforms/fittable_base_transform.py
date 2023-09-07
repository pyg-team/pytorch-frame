import copy
from abc import abstractmethod

from torch_frame import TaskType, TensorFrame
from torch_frame.transforms import BaseTransform


class FittableBaseTransform(BaseTransform):
    r"""An abstract base class for writing fittable transforms.

    Fittable transforms must be fitted on train data before transform.
    """
    def __init__(self, task_type: TaskType, *args, **kwargs):
        self.task_type = task_type
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
        if self._is_fitted:
            raise ValueError(f"'{self.__class__.__name__}' is already fitted. "
                             f"Call `reset_parameters()` before fitting again")
        self._fit(tf)
        self._is_fitted = True

    def forward(self, tf: TensorFrame) -> TensorFrame:
        if not self._is_fitted:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fitted ."
                             f"Please run `fit()` first before attempting to "
                             f"calibrate model outputs.")

        return self._forward(tf)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @abstractmethod
    def _fit(self, tf: TensorFrame):
        pass

    @abstractmethod
    def _forward(self, tf: TensorFrame) -> TensorFrame:
        pass
