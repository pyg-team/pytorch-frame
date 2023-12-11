from typing import Dict, Optional

import pandas as pd

import torch_frame
from torch_frame import stype
from torch_frame.utils.split import SPLIT_TO_NUM


class HuggingFaceDatasetDict(torch_frame.data.Dataset):
    def __init__(
        self,
        path: str,
        col_to_stype: Dict[str, stype],
        name: Optional[str] = None,
        target_col: Optional[str] = None,
        **kwargs,
    ):
        try:
            from datasets import DatasetDict, load_dataset
        except ImportError:
            raise ImportError("Please run `pip install datasets` at first.")
        dataset = load_dataset(path, name)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"{self.__class__} only supports `DatasetDict`")
        # Convert dataset to pandas format
        dataset.set_format(type="pandas")
        dfs = []
        for split_name in dataset:
            # Load pandas dataframe for each split
            df: pd.DataFrame = dataset[split_name][:]
            # Add the split column
            if "val" in split_name:
                split_name = "val"
            df = df.assign(split=SPLIT_TO_NUM[split_name])
            dfs.append(df)
        df = pd.concat(dfs)
        super().__init__(df, col_to_stype, target_col=target_col,
                         split_col='split', **kwargs)
