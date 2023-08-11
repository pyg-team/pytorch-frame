import os
import os.path as osp
import ssl
import sys
import urllib.request
from abc import ABC
from typing import Dict, List, Optional

import torch_frame
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
        for col_name in stypes.keys():
            if col_name not in df:
                raise ValueError(f"The column '{col_name}' is missing the "
                                 f"data frame")

        if target_col is not None:
            if target_col not in df:
                raise ValueError(f"The column '{col_name}' is missing the "
                                 f"data frame")
            if target_col not in stypes.keys():
                raise ValueError(f"The column '{col_name}' is missing in the "
                                 f"semantic type information")

        self.df = df
        self.stypes = stypes
        self.target_col = target_col

    @staticmethod
    def download(
        url: str,
        root: str,
        filename: Optional[str] = None,
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
        if filename is None:
            filename = url.rpartition('/')[2]
            if filename[0] != '?':
                filename.split('?')[0]

        path = osp.join(root, filename)

        if osp.exists(path):
            return path

        if log and 'pytest' not in sys.modules:
            print(f'Downloading {url}', file=sys.stderr)

        os.makedirs(root, exist_ok=True)

        context = ssl._create_unverified_context()
        data = urllib.request.urlopen(url, context=context)

        with open(path, 'wb') as f:
            while True:
                chunk = data.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        return path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> List[str]:
        r"""The input column names of the dataset."""
        columns = list(self.stypes.keys())
        if self.target_col is not None:
            columns.remove(self.target_col)
        return columns
