from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import Tensor
from torch.nn import Module

from torch_frame import TensorFrame


class FeatureEncoder(Module, ABC):
    r"""Base class for feature encoder that transforms input
    :class:`torch_frame.TensorFrame` into :obj:`(x, col_names)`,
    where :obj:`x` is the colum-wise pytorch tensor and
    :obj:`col_names` is the names of the columns. This class can contain
    learnable parameters and missing value handling.
    """
    @abstractmethod
    def forward(self, tf: TensorFrame) -> Tuple[Tensor, List[str]]:
        r"""Encode :obj:`TensorFrame` into (x, col_names).

        Args:
            tf (:class:`torch_frame.TensorFrame`): Input :obj:`TensorFrame`.

        Returns:
            (torch.Tensor, List[str]): A tuple of an output column-wise
            :class:`torch.Tensor` of shape
            [batch_size, num_cols, hidden_channels]
            and a list of column names of  :obj:`x`. The length needs
            to be :obj:`num_cols`.
        """
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass
