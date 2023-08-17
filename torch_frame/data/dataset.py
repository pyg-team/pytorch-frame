from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor

import torch_frame
from torch_frame.data import TensorFrame
from torch_frame.data.mapper import (
    CategoricalTensorMapper,
    NumericalTensorMapper,
)
from torch_frame.typing import DataFrame


class Dataset(ABC):
    r"""Base class for creating tabular datasets.

    Args:
        df (DataFrame): The tabular data frame.
        stypes (Dict[str, torch_frame.stype]): A dictionary that maps each
            column in the data frame to a semantic type.
        target_col (str, optional): The column used as target.
    """
    def __init__(
        self,
        df: DataFrame,
        stypes: Dict[str, torch_frame.stype],
        target_col: Optional[str] = None,
    ):
        self.df = df
        self.stypes = stypes
        self.target_col = target_col

        cols = self.feat_cols + ([] if target_col is None else [target_col])
        missing_cols = set(cols) - set(df.columns)
        if len(missing_cols) > 0:
            raise ValueError(f"The column(s) '{missing_cols}' are missing in "
                             f"the data frame")

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

    @property
    def feat_cols(self) -> List[str]:
        r"""The input feature columns of the dataset."""
        columns = list(self.stypes.keys())
        if self.target_col is not None:
            columns.remove(self.target_col)
        return columns

    def to_tensor_frame(
        self,
        device: Optional[torch.device] = None,
    ) -> TensorFrame:
        r"""Converts the dataset into a :class:`TensorFrame`."""

        xs_dict: Dict[torch_frame.stype, List[Tensor]] = defaultdict(list)
        col_names_dict: Dict[torch_frame.stype, List[str]] = defaultdict(list)
        y: Optional[Tensor] = None

        for col_name, stype in self.stypes.items():

            if stype == torch_frame.numerical:
                mapper = NumericalTensorMapper()

            elif stype == torch_frame.categorical:
                # TODO For now, we simply use the set of unique values to
                # define the category mapping, but eventually we need a better
                # way to do this because we want to guarantee a consisting
                # mapping across different splits.
                count = self.df[col_name].value_counts()
                count = count.sort_values(ascending=False)
                mapper = CategoricalTensorMapper(categories=count.index)

            else:
                raise NotImplementedError(f"Unable to process the semantic "
                                          f"type '{stype.value}'")

            if col_name == self.target_col:
                y = mapper.forward(self.df[col_name])
            else:
                xs_dict[stype].append(mapper.forward(self.df[col_name]))
                col_names_dict[stype].append(col_name)

        x_dict = {
            stype: torch.stack(xs, dim=1)
            for stype, xs in xs_dict.items()
        }

        return TensorFrame(x_dict, col_names_dict, y)
