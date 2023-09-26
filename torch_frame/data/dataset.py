import copy
import functools
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import Tensor

import torch_frame
from torch_frame.data import TensorFrame
from torch_frame.data.mapper import (
    CategoricalTensorMapper,
    NumericalTensorMapper,
    TensorMapper,
    TextEmbeddingTensorMapper,
)
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.typing import (
    ColumnSelectType,
    DataFrame,
    IndexSelectType,
    TaskType,
)


def requires_pre_materialization(func):
    @functools.wraps(func)
    def _requires_pre_materialization(self, *args, **kwargs):
        if self.is_materialized:
            raise RuntimeError(
                f"'{self}' cannot be modified via '{func.__name__}' post "
                f"materialization")
        return func(self, *args, **kwargs)

    return _requires_pre_materialization


def requires_post_materialization(func):
    @functools.wraps(func)
    def _requires_post_materialization(self, *args, **kwargs):
        if not self.is_materialized:
            raise RuntimeError(
                f"'{func.__name__}' requires a materialized dataset. Please "
                f"call `dataset.materialize(...)` first.")
        return func(self, *args, **kwargs)

    return _requires_post_materialization


class DataFrameToTensorFrameConverter:
    r"""DataFrame to TensorFrame converter.

    Args:
        col_to_stype (Dict[str, torch_frame.stype]): A dictionary that maps
            each column in the data frame to a semantic type.
        col_stats (Dict[str, Dict[StatType, Any]]): A dictionary that maps
            column name into stats. Available as :obj:`dataset.col_stats`.
        target_col (str, optional): The column used as target.
            (default: :obj:`None`)
        text_embedder (callable, optional): A callable text embedder that
            takes a list of strings as input and returns corresponding text
            embedding tensor. This text embedder is only called when there
            is text stype data in the dataframe. Series data will call
            :obj:`tolist` before input to the function. (default: :obj:`None`)
    """
    def __init__(
        self,
        col_to_stype: Dict[str, torch_frame.stype],
        col_stats: Dict[str, Dict[StatType, Any]],
        target_col: Optional[str] = None,
        text_embedder: Optional[Callable[[List[str]], Tensor]] = None,
    ):
        self.col_to_stype = col_to_stype
        self.col_stats = col_stats
        self.target_col = target_col
        self.text_embedder = text_embedder

        # Pre-compute a canonical `col_names_dict` for tensor frame.
        self._col_names_dict: Dict[torch_frame.stype,
                                   List[str]] = defaultdict(list)
        for col, stype in self.col_to_stype.items():
            if col != self.target_col:
                self._col_names_dict[stype].append(col)
        if (torch_frame.text_embedded
                in self.col_names_dict) and (self.text_embedder is None):
            raise ValueError("`text_embedder` needs to be "
                             "specified when `text_embedded` "
                             "stype column exist.")
        for stype in self._col_names_dict.keys():
            # in-place sorting of col_names for each stype
            sorted(self._col_names_dict[stype])

    @property
    def col_names_dict(self) -> Dict[torch_frame.stype, List[str]]:
        return self._col_names_dict

    def _get_mapper(self, col: str) -> TensorMapper:
        r"""Get TensorMapper given a column name."""
        stype = self.col_to_stype[col]
        if stype == torch_frame.numerical:
            return NumericalTensorMapper()
        elif stype == torch_frame.categorical:
            index, _ = self.col_stats[col][StatType.COUNT]
            return CategoricalTensorMapper(index)
        elif stype == torch_frame.text_embedded:
            return TextEmbeddingTensorMapper(self.text_embedder)
        else:
            raise NotImplementedError(f"Unable to process the semantic "
                                      f"type '{stype.value}'")

    def __call__(
        self,
        df: DataFrame,
        device: Optional[torch.device] = None,
    ) -> TensorFrame:
        r"""Convert a given dataframe into :obj:`TensorFrame`."""

        xs_dict: Dict[torch_frame.stype, List[Tensor]] = defaultdict(list)

        for stype, col_names in self.col_names_dict.items():
            for col in col_names:
                out = self._get_mapper(col).forward(df[col], device=device)
                xs_dict[stype].append(out)
        x_dict = {
            stype: torch.stack(xs, dim=1)
            for stype, xs in xs_dict.items()
        }

        y: Optional[Tensor] = None
        if self.target_col is not None:
            y = self._get_mapper(self.target_col).forward(
                df[self.target_col], device=device)

        return TensorFrame(x_dict, self.col_names_dict, y)


class Dataset(ABC):
    r"""Base class for creating tabular datasets.

    Args:
        df (DataFrame): The tabular data frame.
        col_to_stype (Dict[str, torch_frame.stype]): A dictionary that maps
            each column in the data frame to a semantic type.
        target_col (str, optional): The column used as target.
            (default: :obj:`None`)
        split_col (str, optional): The column that stores the pre-defined split
            information. The column should only contain 'train', 'val', or
            'test'. (default: :obj:`None`).
        text_embedder (callable, optional): A callable text embedder that
            takes a list of strings as input and returns corresponding text
            embedding tensor. This text embedder is only called when there
            is text stype data in the dataframe. Series data will call
            :obj:`tolist` before input to the function. (default: :obj:`None`)
    """
    def __init__(
        self,
        df: DataFrame,
        col_to_stype: Dict[str, torch_frame.stype],
        target_col: Optional[str] = None,
        split_col: Optional[str] = None,
        text_embedder: Optional[Callable[[List[str]], Tensor]] = None,
    ):
        self.df = df
        self.target_col = target_col

        if split_col is not None:
            if split_col not in df.columns:
                raise ValueError(
                    f"Given split_col ({split_col}) does not match columns of "
                    f"the given df.")
            if split_col in col_to_stype:
                raise ValueError(
                    f"col_to_stype should not contain the split_col "
                    f"({col_to_stype}).")
            if not set(df[split_col]).issubset({'train', 'val', 'test'}):
                raise ValueError(
                    "split_col must only contain either 'train', 'val', or "
                    "'test'.")
        self.split_col = split_col
        self.col_to_stype = col_to_stype

        cols = self.feat_cols + ([] if target_col is None else [target_col])
        missing_cols = set(cols) - set(df.columns)
        if len(missing_cols) > 0:
            raise ValueError(f"The column(s) '{missing_cols}' are specified "
                             f"but missing in the data frame")

        self.text_embedder = text_embedder
        self._is_materialized: bool = False
        self._col_stats: Dict[str, Dict[StatType, Any]] = {}
        self._tensor_frame: Optional[TensorFrame] = None

    @staticmethod
    def download_url(
        url: str,
        root: str,
        filename: Optional[str] = None,
        *,
        log: bool = True,
    ) -> str:
        r"""Downloads the content of :obj:`url` to the specified folder
        :obj:`root`.

        Args:
            url (str): The URL.
            root (str): The root folder.
            filename (str, optional): If set, will rename the downloaded file.
                (default: :obj:`None`)
            log (bool, optional): If :obj:`False`, will not print anything to
                the console. (default: :obj:`True`)
        """
        return torch_frame.data.download_url(url, root, filename, log=log)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: IndexSelectType) -> 'Dataset':
        is_col_select = isinstance(index, str)
        is_col_select |= (isinstance(index, (list, tuple)) and len(index) > 0
                          and isinstance(index[0], str))

        if is_col_select:
            return self.col_select(index)

        return self.index_select(index)

    @property
    def feat_cols(self) -> List[str]:
        r"""The input feature columns of the dataset."""
        cols = list(self.col_to_stype.keys())
        if self.target_col is not None:
            cols.remove(self.target_col)
        return cols

    @property
    def task_type(self) -> TaskType:
        r"""The task type of the dataset."""
        raise NotImplementedError

    @property
    @requires_post_materialization
    def num_classes(self) -> int:
        if StatType.COUNT not in self.col_stats[self.target_col]:
            raise ValueError(
                f"num_classes attribute is only supported when the target "
                f"column ({self.target_col}) stats contains StatType.COUNT, "
                f"but only the following target column stats are calculated: "
                f"{list(self.col_stats[self.target_col].keys())}.")
        return len(self.col_stats[self.target_col][StatType.COUNT][0])

    # Materialization #########################################################

    def materialize(self, device: Optional[torch.device] = None) -> 'Dataset':
        r"""Materializes the dataset into a tensor representation. From this
        point onwards, the dataset should be treated as read-only."""
        if self.is_materialized:
            return self

        # 1. Fill column statistics:
        for col, stype in self.col_to_stype.items():
            self._col_stats[col] = compute_col_stats(self.df[col], stype)
            # For a target column, sort categories lexicographically such that
            # we do not accidentally swap labels in binary classification
            # tasks.
            if col == self.target_col and stype == torch_frame.categorical:
                index, value = self._col_stats[col][StatType.COUNT]
                if len(index) == 2:
                    ser = pd.Series(index=index, data=value).sort_index()
                    index, value = ser.index.tolist(), ser.values.tolist()
                    self._col_stats[col][StatType.COUNT] = (index, value)

        # 2. Create the `TensorFrame`:
        self._to_tensor_frame_converter = DataFrameToTensorFrameConverter(
            col_to_stype=self.col_to_stype,
            col_stats=self._col_stats,
            target_col=self.target_col,
            text_embedder=getattr(self, 'text_embedder', None),
        )
        self._tensor_frame = self._to_tensor_frame_converter(self.df, device)

        # 3. Mark the dataset as materialized:
        self._is_materialized = True

        return self

    @property
    def is_materialized(self) -> bool:
        r"""Whether the dataset is already materialized."""
        return self._is_materialized

    @property
    @requires_post_materialization
    def tensor_frame(self) -> TensorFrame:
        r"""Returns the :class:`TensorFrame` of the dataset."""
        return self._tensor_frame

    @property
    @requires_post_materialization
    def col_stats(self) -> Dict[str, Dict[StatType, Any]]:
        r"""Returns column-wise dataset statistics."""
        return self._col_stats

    # Indexing ################################################################

    @requires_post_materialization
    def index_select(self, index: IndexSelectType) -> 'Dataset':
        r"""Returns a subset of the dataset from specified indices
        :obj:`index`."""
        if isinstance(index, int):
            index = [index]

        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            index = slice(start, stop, step)

        dataset = copy.copy(self)

        iloc = index.cpu().numpy() if isinstance(index, Tensor) else index
        dataset.df = self.df.iloc[iloc]

        dataset._tensor_frame = self._tensor_frame[index]

        return dataset

    def shuffle(
        self, return_perm: bool = False
    ) -> Union['Dataset', Tuple['Dataset', Tensor]]:
        r"""Randomly shuffles the rows in the dataset."""
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    @requires_pre_materialization
    def col_select(self, cols: ColumnSelectType) -> 'Dataset':
        r"""Returns a subset of the dataset from specified columns
        :obj:`cols`."""
        cols = [cols] if isinstance(cols, str) else cols

        if self.target_col is not None and self.target_col not in cols:
            cols.append(self.target_col)

        dataset = copy.copy(self)

        dataset.df = self.df[cols]
        dataset.col_to_stype = {col: self.col_to_stype[col] for col in cols}

        return dataset

    def get_split_dataset(self, split: str) -> 'Dataset':
        r"""Get splitted dataset defined in `split_col` of :obj:`self.df`.

        Args:
            split (str): The split name. Should be 'train', 'val', or 'test'.
        """
        if self.split_col is None:
            raise ValueError(
                f"'get_split_dataset' is not supported for {self} "
                "since 'split_col' is not specified.")
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"The split named {split} is not available. "
                             f"Needs to either 'train', 'val', or 'test'.")
        indices = self.df.index[self.df[self.split_col] == split].tolist()
        return self[indices]

    @property
    @requires_post_materialization
    def convert_to_tensor_frame(self) -> DataFrameToTensorFrameConverter:
        return self._to_tensor_frame_converter
