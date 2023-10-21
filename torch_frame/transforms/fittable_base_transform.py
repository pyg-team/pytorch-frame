import copy
from abc import abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor

from torch_frame import NAStrategy, TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.transforms import BaseTransform


class FittableBaseTransform(BaseTransform):
    r"""An abstract base class for writing fittable transforms.
    Fittable transforms must be fitted on training data before transform.
    """
    def __init__(self):
        super().__init__()
        self._is_fitted: bool = False

    def __call__(self, tf: TensorFrame) -> TensorFrame:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(tf))

    @property
    def is_fitted(self) -> bool:
        r"""Whether the transform is already fitted."""
        return self._is_fitted

    def _replace_nans(self, x: Tensor, na_strategy: NAStrategy):
        r"""Replace NaNs based on NAStrategy.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame` whose NaN values
                in categorical columns are to be replaced.
            na_strategy (NAStrategy): The :class:`NAStrategy` used to
                replace NaN values.

        Returns:
            Tensor: Output :obj:`TensorFrame` with NaN values replaced.
        """
        x = x.clone()
        for col in range(x.size(1)):
            column_data = x[:, col]
            if na_strategy.is_numerical_strategy:
                nan_mask = torch.isnan(column_data)
            else:
                nan_mask = column_data < 0
            if not nan_mask.any():
                continue
            valid_data = column_data[~nan_mask]
            if na_strategy == NAStrategy.MEAN:
                fill_value = valid_data.mean()
            elif na_strategy in [NAStrategy.ZEROS, NAStrategy.MOST_FREQUENT]:
                fill_value = torch.tensor(0.)
            column_data[nan_mask] = fill_value
        return x

    def fit(
        self,
        tf: TensorFrame,
        col_stats: Dict[str, Dict[StatType, Any]],
    ):
        r"""Fit the transform with train data.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame` representing train data.
            col_stats (Dict[str, Dict[StatType, Any]], optional): The column
                stats of the input :obj:`TensorFrame`.
        """
        self._fit(tf, col_stats)
        self._is_fitted = True

    def forward(self, tf: TensorFrame) -> TensorFrame:
        if not self.is_fitted:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fitted ."
                             f"Please run `fit()` first before attempting to "
                             f"transform the TensorFrame.")

        transformed_tf = self._forward(tf)
        transformed_tf.validate()
        return transformed_tf

    @abstractmethod
    def _fit(self, tf: TensorFrame, col_stats: Dict[str, Dict[StatType, Any]]):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, tf: TensorFrame) -> TensorFrame:
        raise NotImplementedError
