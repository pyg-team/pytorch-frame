from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import Tensor
from torch.nn import Module

from torch_frame import TensorFrame


class FeatureEncoder(Module, ABC):
    r"""Base class for feature encoder that transforms input TensorFrame into
    :obj:`(x, col_names)`, where :obj:`x` is the colum-wise pytorch tensor and
    :obj:`col_names` is the names of the columns. This class can contain
    learnable parameters and missing value handling.
    """
    @abstractmethod
    def forward(self, tf: TensorFrame) -> Tuple[Tensor, List[str]]:
        r"""Encode TensorFrame into (x, col_names).
        Args:
            df (TensorFrame): Input TensorFrame

        Returns:
            x (Tensor): Output column-wise pytorch tensor of shape
                :obj:`[batch_size, num_cols, hidden_channels]`.
            col_names (List[str]): Column names of  :obj:`x`. The length needs
                to be :obj:`num_cols`.
        """
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass
