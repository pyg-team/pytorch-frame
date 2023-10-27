from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from torch_frame.data import MultiNestedTensor
from torch_frame.typing import Series, TensorData


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
    ) -> TensorData:
        r"""Maps raw input data into a compact tensor representation."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, tensor: TensorData) -> pd.Series:
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
    r"""Maps any categorical series into an index representation, with
    :obj:`-1` denoting N/A values."""
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


class MultiCategoricalTensorMapper(TensorMapper):
    r"""Maps any multi-categorical series into an index representation, with
    :obj:`-1` denoting missing values (NaN) and no value denoting not belonging
    to any categories. If the cell contains unseen categories, it will be
    ignored.

    Args:
        categories (List[Any]): A list of possible categories in the
        multi-categorical column sorted by occurence.
        sep (str): The delimiter for the categories in each cell.
        (default: :obj:`,`)
    """
    def __init__(
        self,
        categories: List[Any],
        sep: str = ',',
    ):
        super().__init__()
        self.categories = categories
        self.sep = sep
        self.index = pd.Series(
            index=categories,
            data=pd.RangeIndex(0, len(categories)),
        )
        self.index = pd.concat((self.index, (pd.Series([-1], index=[-1]))))

    def _row_to_tensor(self, row: str):
        if row is None:
            return [-1]
        else:
            return row.split(self.sep)

    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> MultiNestedTensor:
        if ser.dtype != 'object':
            raise ValueError('Multi-categorical types expect string as input')
        values = []
        original_index = ser.index
        ser = ser.apply(self._row_to_tensor)
        ser = ser.explode()
        ser = pd.merge(
            ser.rename('data'),
            self.index.rename('index'),
            how='left',
            left_on='data',
            right_index=True,
        ).dropna()
        ser['index'] = ser['index'].astype('int64')
        ser = ser.groupby(level=0)['index'].apply(list)
        ser = ser.reindex(original_index, fill_value=[])
        values = torch.tensor(sum(ser, []), device=device)
        ser = ser.apply(lambda x: len(x))
        offset = torch.tensor([0] + ser.tolist(), device=device)
        offset = torch.cumsum(offset, dim=0)
        return MultiNestedTensor(num_rows=len(ser), num_cols=1, values=values,
                                 offset=offset)

    def backward(self, tensor: MultiNestedTensor) -> pd.Series:
        values = tensor.values
        offset = tensor.offset
        values = values.tolist()
        ser = []
        for i in range(1, len(offset)):
            index = list([
                self.categories[item]
                for item in values[offset[i - 1]:offset[i]] if item != -1
            ])
            ser.append(self.sep.join(index))
        return pd.Series(ser)


class TextEmbeddingTensorMapper(TensorMapper):
    r"""Embed any text series into tensor.

    Args:
        text_embedder (callable): A callable function that takes list of
            strings and returns embedding for that list of strings. For heavy
            text embedding model (e.g., based on Transformer), we recommend
            using GPU.
        batch_size (int, optional): The mini-batch size used for the text
            embedder. If :obj:`None`, we will encode all text in a full-batch
            manner. If you use heavy text embedding model with GPU, we
            recommend you setting :obj:`batch_size` to a reasonable number to
            avoid the GPU OOM issue.
    """
    def __init__(
        self,
        text_embedder: Callable[[List[str]], Tensor],
        batch_size: Optional[int],
    ):
        super().__init__()
        self.text_embedder = text_embedder
        self.batch_size = batch_size

    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        ser_list = ser.tolist()
        if self.batch_size is None:
            emb = self.text_embedder(ser_list)
            return emb.to(device)

        emb_list = []
        for i in tqdm(range(0, len(ser_list), self.batch_size),
                      desc="Embedding texts in mini-batch"):
            emb = self.text_embedder(ser_list[i:i + self.batch_size])
            emb_list.append(emb.to(device))
        return torch.cat(emb_list, dim=0)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError
