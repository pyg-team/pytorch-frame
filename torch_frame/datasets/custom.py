from typing import Dict, Optional

import pandas as pd

import torch_frame
from torch_frame import FileType, stype


class CustomDataset(torch_frame.data.Dataset):
    r"""Custom dataset loaded from CSV or PARQUET files

    Args:

    """
    def __init__(self, path: str, col_to_stype: Dict[str, stype],
                 target: Optional[str] = None,
                 file_type: Optional[FileType] = FileType.CSV):
        if file_type == FileType.CSV:
            df = pd.read_csv(path)
        elif file_type == FileType.PARQUET:
            df = pd.read_parquet(path)
        if not set(col_to_stype.keys()).issubset(df.columns):
            raise ValueError("The columns of the dataset does not contain all "
                             f"{col_to_stype.keys()}")
        super().__init__(df, col_to_stype, target_col=target)
