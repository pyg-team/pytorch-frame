from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

import pandas as pd
import torch
from torch import Tensor

from torch_frame.typing import Series


class TensorMapper(ABC):
    r"""A base class to handle the conversion from raw input data into a
    compact tensor representation, i.e., the identity for numerical values,
    indices for categorical values, etc."""
    @abstractmethod
    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        r"""Maps raw input data into a compact tensor representation."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, tensor: Tensor) -> pd.Series:
        r"""Maps a compact tensor representation back into the raw input data.
        The reverse operation of :meth:`forward`."""
        raise NotImplementedError


class NumericalTensorMapper(TensorMapper):
    r"""Maps any numerical series into a floating-point representation, with
    :obj:`float('NaN')` denoting N/A values."""
    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        # NOTE We are converting the default PyTorch dtype into a string
        # representation that can be understood by numpy.
        # TODO Think of a "less hacky" way to do this.
        dtype = str(torch.get_default_dtype()).split('.')[-1]
        value = ser.values.astype(dtype)
        return torch.from_numpy(value).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        return pd.Series(tensor.detach().cpu().numpy())


class CategoricalTensorMapper(TensorMapper):
    r"""Maps any categorical series into a index representation, with :obj:`-1`
    denoting N/A values."""
    def __init__(self, categories: Iterable[Any]):
        super().__init__()

        self.categories: pd.Series = pd.Series(
            index=categories,
            data=pd.RangeIndex(0, len(categories)),
            name='index',
        )

    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:

        index = pd.merge(
            ser.rename('data'),
            self.categories,
            how='left',
            left_on='data',
            right_index=True,
        )['index'].values
        index = torch.from_numpy(index).to(device)

        if index.is_floating_point():
            index[index.isnan()] = -1

        return index.to(torch.long)

    def backward(self, tensor: Tensor) -> pd.Series:
        index = tensor.cpu().numpy()
        ser = pd.Series(self.categories[index].index)
        ser[index < 0] = None
        return ser
