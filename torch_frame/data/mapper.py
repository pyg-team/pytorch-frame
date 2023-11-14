from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from torch_frame.data import MultiNestedTensor
from torch_frame.typing import Series, TensorData, TextTokenizationOutputs


class TensorMapper(ABC):
    r"""A base class to handle the conversion from raw input data into a
    compact tensor representation, i.e., the identity for numerical values,
    indices for categorical values, etc.
    """
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
        The reverse operation of :meth:`forward`.
        """
        raise NotImplementedError


class NumericalTensorMapper(TensorMapper):
    r"""Maps any numerical series into a floating-point representation, with
    :obj:`float('NaN')` denoting N/A values.
    """
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
    :obj:`-1` denoting N/A values.
    """
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
    to any categories. Unseen categories will be ignored.

    Args:
        categories (List[Any]): A list of possible categories in the
        multi-categorical column sorted by counts.
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

    def _split_by_sep(self, row: str):
        if row is None:
            return set([-1])
        elif row == '':
            return set()
        else:
            return set(row.split(self.sep))

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
        ser = ser.apply(self._split_by_sep)
        ser = ser.explode()
        ser = pd.merge(
            ser.rename('data'),
            self.index.rename('index'),
            how='left',
            left_on='data',
            right_index=True,
        ).dropna()
        ser['index'] = ser['index'].astype('int64')
        values = torch.from_numpy(ser['index'].values)
        offset = ser.index.value_counts()
        offset = offset.reindex(original_index, fill_value=0)
        offset = pd.concat((pd.Series([0]), offset))
        offset = torch.from_numpy(offset.values)
        offset = torch.cumsum(offset, dim=0)
        return MultiNestedTensor(num_rows=len(original_index), num_cols=1,
                                 values=values, offset=offset)

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


class NumericalSequenceTensorMapper(TensorMapper):
    r"""Maps any sequence series into an :class:`MultiNestedTensor`."""
    def __init__(self, ):
        super().__init__()

    def get_sequence_length(self, row):
        if isinstance(row, List):
            return len(row)
        elif row is None or (isinstance(row, float) and pd.isna(row)):
            return 0
        else:
            raise ValueError(f"{type(row)} is not supported as"
                             " numerical sequence.")

    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> MultiNestedTensor:
        values = []
        num_rows = len(ser)
        offset = ser.apply(lambda row: self.get_sequence_length(row))
        ser = ser[offset != 0]
        offset = pd.concat((pd.Series([0]), offset))
        offset = torch.from_numpy(offset.values)
        offset = torch.cumsum(offset, dim=0)
        ser = ser.explode()
        values = torch.from_numpy(ser.values.astype('float32'))
        return MultiNestedTensor(num_rows=num_rows, num_cols=1, values=values,
                                 offset=offset)

    def backward(self, tensor: MultiNestedTensor) -> pd.Series:
        values = tensor.values.cpu().numpy()
        offset = tensor.offset
        ser = []
        for i in range(1, len(offset)):
            val = values[offset[i - 1]:offset[i]]
            ser.append(val if val.size > 0 else None)
        return pd.Series(ser)


class TimestampTensorMapper(TensorMapper):
    r"""Maps any sequence series into an :class:`MultiNestedTensor`."""
    def __init__(self, ):
        super().__init__()

    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        ser = pd.to_datetime(ser)
        tensors = [
            torch.tensor(ser.dt.year.values, device=device),
            torch.tensor(ser.dt.month.values, device=device),
            torch.tensor(ser.dt.day.values, device=device),
            torch.tensor(ser.dt.dayofweek, device=device),
            torch.tensor(ser.dt.hour, device=device),
            torch.tensor(ser.dt.minute, device=device),
            torch.tensor(ser.dt.second, device=device)
        ]
        return torch.cat(tensors, dim=1).reshape(len(ser), 1, len(tensors))

    def backward(self, tensor: Tensor) -> pd.Series:
        df = pd.DataFrame(
            tensor, columns=[
                'year', 'month', 'day', 'dayofweek', 'hour', 'minute', 'second'
            ])
        ser = pd.to_datetime(
            df[['year', 'month', 'day', 'hour', 'minute', 'second']])
        return ser


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
        ser = ser.astype(str)
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


class TextTokenizationTensorMapper(TensorMapper):
    r"""Tokenize any text series into a dictionary of
    :class:`MultiNestedTensor`.

    Args:
        text_tokenizer (callable): A callable function that takes list of
            strings and returns list of dictionary. The keys of the dictionary
            are arguments that will be put to the model, including
            :obj:`input_ids` and :obj:`attention_mask`. The values of the
            dictionary are tensors corresponding to keys.
        batch_size (int, optional): The mini-batch size used for the text
            tokenizer. If :obj:`None`, we will encode all text in a full-batch
            manner.
    """
    def __init__(
        self,
        text_tokenizer: Callable[[List[str]], TextTokenizationOutputs],
        batch_size: Optional[int],
    ):
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.batch_size = batch_size

    def forward(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, MultiNestedTensor]:
        ser = ser.astype(str)
        ser_list = ser.tolist()

        feat_dict = {}
        if self.batch_size is None:
            tokenized_list: TextTokenizationOutputs = self.text_tokenizer(
                ser_list)
            for key in tokenized_list[0]:
                xs = []
                for item in tokenized_list:
                    if item[key].ndim == 1:
                        xs.append([item[key]])
                    elif item[key].ndim == 2:
                        xs.append([row for row in item[key]])
                    else:
                        raise ValueError(f'{key} has `ndim` not '
                                         f'equal to 1 or 2.')
                feat_dict[key] = MultiNestedTensor.from_tensor_mat(xs).to(
                    device)
            return feat_dict

        tokenized_list: TextTokenizationOutputs = []
        for i in tqdm(range(0, len(ser_list), self.batch_size),
                      desc="Tokenizing texts in mini-batch"):
            tokenized_batch: TextTokenizationOutputs = self.text_tokenizer(
                ser_list[i:i + self.batch_size])
            tokenized_list.extend(tokenized_batch)
        for key in tokenized_list[0]:
            xs = []
            for item in tokenized_list:
                if item[key].ndim == 1:
                    xs.append([item[key]])
                elif item[key].ndim == 2:
                    for row in item[key]:
                        xs.append([row])
                else:
                    raise ValueError(f'{key} has `ndim` not '
                                     f'equal to 1 or 2.')
            feat_dict[key] = MultiNestedTensor.from_tensor_mat(xs).to(device)
        return feat_dict

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError
