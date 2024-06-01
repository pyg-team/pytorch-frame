from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.typing import Series, TensorData, TextTokenizationOutputs

NUM_MONTHS_PER_YEAR = 12
'''
TODO: We might want to revisit the normalization
constant for NUM_DAYS_PER_MONTH as some months only
have 28-30 days.
'''
NUM_DAYS_PER_MONTH = 31
NUM_DAYS_PER_WEEK = 7
NUM_HOURS_PER_DAY = 24
NUM_MINUTES_PER_HOUR = 60
NUM_SECONDS_PER_MINUTE = 60


def _get_default_numpy_dtype() -> np.dtype:
    r"""Returns the default numpy dtype."""
    # NOTE: We are converting the default PyTorch dtype into a string
    # representation that can be understood by numpy.
    # TODO: Think of a "less hacky" way to do this.
    dtype = str(torch.get_default_dtype()).split('.')[-1]
    return np.dtype(dtype)


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
        device: torch.device | None = None,
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
        device: torch.device | None = None,
    ) -> Tensor:
        dtype = _get_default_numpy_dtype()
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
        device: torch.device | None = None,
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
        ser = pd.Series(self.categories.iloc[index].index)
        ser[index < 0] = None
        return ser


class MultiCategoricalTensorMapper(TensorMapper):
    r"""Maps any multi-categorical series into an index representation, with
    :obj:`-1` denoting missing values (NaN) and no value denoting not belonging
    to any categories. Unseen categories will be ignored.

    Args:
        categories (List[Any]): A list of possible categories in the
        multi-categorical column sorted by counts.
        sep (str, optional): The delimiter for the categories in each cell.
            If :obj:`None`, it assumes each cell is directly represented as a
            list of categories. (default: :obj:`None`)
    """
    def __init__(
        self,
        categories: list[Any],
        sep: None | str = None,
    ):
        super().__init__()
        self.categories = categories
        self.sep = sep
        self.index = pd.Series(
            index=categories,
            data=pd.RangeIndex(0, len(categories)),
        )
        self.index = pd.concat((self.index, (pd.Series([-1], index=[-1]))))

    @staticmethod
    def split_by_sep(row: str | Iterable | None, sep: None | str) -> set[Any]:
        if row is None or row is np.nan:
            return {-1}
        elif isinstance(row, str):
            assert sep is not None
            if row.strip() == '':
                return set()
            else:
                return {cat.strip() for cat in row.split(sep)}
        elif isinstance(row, Iterable):
            assert sep is None
            return set(row)
        else:
            raise ValueError(
                f"MulticategoricalTensorMapper only supports str or list "
                f"types (got input: {row})")

    def forward(
        self,
        ser: Series,
        *,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        if ser.dtype != 'object':
            raise ValueError('Multi-categorical types expect string as input')
        values = []
        original_index = ser.index
        ser = ser.apply(lambda row: MultiCategoricalTensorMapper.split_by_sep(
            row, sep=self.sep))
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
        if isinstance(row, list):
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
        device: torch.device | None = None,
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
    TIME_TO_INDEX = {
        'YEAR': 0,
        'MONTH': 1,
        'DAY': 2,
        'DAYOFWEEK': 3,
        'HOUR': 4,
        'MINUTE': 5,
        'SECOND': 6
    }
    CYCLIC_VALUES_NORMALIZATION_CONSTANT = torch.tensor([
        NUM_MONTHS_PER_YEAR, NUM_DAYS_PER_MONTH, NUM_DAYS_PER_WEEK,
        NUM_HOURS_PER_DAY, NUM_MINUTES_PER_HOUR, NUM_SECONDS_PER_MINUTE
    ])

    def __init__(self, format: str):
        super().__init__()
        self.format = format

    @staticmethod
    def to_tensor(ser: Series) -> Tensor:
        # subtracting 1 so that the smallest months and days can
        # start from 0.
        tensors = [
            torch.from_numpy(ser.dt.year.values).unsqueeze(1),
            torch.from_numpy(ser.dt.month.values - 1).unsqueeze(1),
            torch.from_numpy(ser.dt.day.values - 1).unsqueeze(1),
            torch.from_numpy(ser.dt.dayofweek.values).unsqueeze(1),
            torch.from_numpy(ser.dt.hour.values).unsqueeze(1),
            torch.from_numpy(ser.dt.minute.values).unsqueeze(1),
            torch.from_numpy(ser.dt.second.values).unsqueeze(1)
        ]
        stacked = torch.cat(tensors, dim=1)
        return torch.nan_to_num(stacked, nan=-1).to(torch.long)

    def forward(
        self,
        ser: Series,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        ser = pd.to_datetime(ser, format=self.format, errors='coerce')
        tensor = TimestampTensorMapper.to_tensor(ser)
        return tensor.to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError


class TextTokenizationTensorMapper(TensorMapper):
    r"""Tokenize any text series into a dictionary of
    :class:`MultiNestedTensor`.

    Args:
        text_tokenizer (callable): A callable function that takes list of
            strings and returns list of dictionaries or dictionary. The keys
            of the dictionary are arguments that will be put to the model,
            such as :obj:`input_ids` and :obj:`attention_mask`. The values
            of the dictionary are tensors corresponding to keys.
        batch_size (int, optional): The mini-batch size used for the text
            tokenizer. If :obj:`None`, we will encode all text in a full-batch
            manner.
    """
    def __init__(
        self,
        text_tokenizer: Callable[[list[str]], TextTokenizationOutputs],
        batch_size: int | None,
    ):
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.batch_size = batch_size

    def forward(
        self,
        ser: Series,
        *,
        device: torch.device | None = None,
    ) -> dict[str, MultiNestedTensor]:
        ser = ser.astype(str)
        ser_list = ser.tolist()

        feat_dict = {}
        if self.batch_size is None:
            tokenized_outputs: TextTokenizationOutputs = self.text_tokenizer(
                ser_list)
            if isinstance(tokenized_outputs, Mapping):
                keys = tokenized_outputs.keys()
                for key in keys:
                    tensors = tokenized_outputs[key]
                    assert tensors.ndim == 2
                    xs = [[tensor] for tensor in tensors]
                    feat_dict[key] = MultiNestedTensor.from_tensor_mat(xs)
            else:
                keys = tokenized_outputs[0].keys()
                for key in keys:
                    xs = []
                    for tensor_dict in tokenized_outputs:
                        tensor = tensor_dict[key]
                        assert tensor.ndim == 1
                        xs.append([tensor])
                    feat_dict[key] = MultiNestedTensor.from_tensor_mat(xs)
            return feat_dict

        tokenized_outputs: list[TextTokenizationOutputs] = []
        for i in tqdm(range(0, len(ser_list), self.batch_size),
                      desc="Tokenizing texts in mini-batch"):
            tokenized_batch: TextTokenizationOutputs = self.text_tokenizer(
                ser_list[i:i + self.batch_size])
            tokenized_outputs.append(tokenized_batch)

        if isinstance(tokenized_outputs[0], Mapping):
            keys = tokenized_outputs[0].keys()
            for key in keys:
                xs = []
                for tokenized_batch in tokenized_outputs:
                    tensors = tokenized_batch[key]
                    assert tensors.ndim == 2
                    xs.extend([tensor] for tensor in tensors)
                feat_dict[key] = MultiNestedTensor.from_tensor_mat(xs)
        else:
            keys = tokenized_outputs[0][0].keys()
            for key in keys:
                xs = []
                for tokenized_batch in tokenized_outputs:
                    for tensor_dict in tokenized_batch:
                        tensor = tensor_dict[key]
                        assert tensor.ndim == 1
                        xs.append([tensor])
                feat_dict[key] = MultiNestedTensor.from_tensor_mat(xs)
        return feat_dict

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError


class EmbeddingTensorMapper(TensorMapper):
    r"""Maps embeddings columns into tensors. It can take either raw data,
    e.g. text, images and convert them into embeddings with the specified
    embedder, or embeddings as column values directly.

    Args:
        embedder (callable, optional): A callable function that takes list of
            raw data and returns embedding for that list of raw data. For heavy
            embedding model (e.g., based on Transformer), we recommend using
            GPU.
        batch_size (int, optional): The mini-batch size used for the embedder.
            If :obj:`None`, we will encode all text in a full-batch manner.
            If you use heavy embedding model with GPU, we recommend you setting
            :obj:`batch_size` to a reasonable number to avoid the GPU OOM
            issue.
    """
    def __init__(
        self,
        embedder: Callable[[list[Any]], Tensor] | None = None,
        batch_size: int | None = None,
    ):
        super().__init__()
        self.embedder = embedder
        self.batch_size = batch_size

    def forward(
        self,
        ser: Series,
        *,
        device: torch.device | None = None,
    ) -> MultiEmbeddingTensor:

        if self.embedder is not None:
            ser = ser.astype(str)
            ser_list = ser.tolist()
            if self.batch_size is None:
                values = self.embedder(ser_list)
            else:
                emb_list = []
                for i in tqdm(range(0, len(ser_list), self.batch_size),
                              desc="Embedding raw data in mini-batch"):
                    emb = self.embedder(ser_list[i:i + self.batch_size])
                    emb_list.append(emb.to(device))
                values = torch.cat(emb_list, dim=0)
        else:
            dtype = _get_default_numpy_dtype()
            values = torch.from_numpy(np.stack(ser.values).astype(dtype))
        return MultiEmbeddingTensor(
            num_rows=len(ser),
            num_cols=1,
            values=values,
            offset=torch.tensor([0, len(values[0])]),
        ).to(device)

    def backward(self, tensor: MultiEmbeddingTensor) -> pd.Series:
        if self.embedder:
            raise NotImplementedError
        values = tensor.values.cpu()
        offset = tensor.offset
        val = [v for v in values[:, 0:offset[1]].numpy()]
        return pd.Series(val)
