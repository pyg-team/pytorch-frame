from typing import List

import numpy as np
import pandas as pd

import torch_frame
from torch_frame import stype


class FakeDataset(torch_frame.data.Dataset):
    r"""A fake regression dataset.

    Args:
        num_rows (int): Number of rows.
        with_nan (bool): Whether include nan in the dataset.
        stypes (List[stype]): List of stype columns to include
                in the dataset. Particularly useful, when you want to
                create a dataset with only numerical or categorical
                feature columns.
        create_split (bool): Whether to create a train, val and test
                split for the fake dataset.
    """
    def __init__(
        self,
        num_rows: int,
        with_nan: bool = False,
        stypes: List[stype] = [stype.categorical, stype.numerical],
        create_split: bool = False,
    ):
        assert len(stypes) > 0
        df_dict = {
            'target': np.random.randn(num_rows),
        }
        col_to_stype = {
            'target': stype.numerical,
        }
        if stype.numerical in stypes:
            for col_name in ['a', 'b', 'c']:
                df_dict[col_name] = np.random.randn(num_rows)
                col_to_stype[col_name] = stype.numerical
        if stype.categorical in stypes:
            for col_name in ['x', 'y']:
                df_dict[col_name] = np.random.randint(0, 3, size=(num_rows, ))
                col_to_stype[col_name] = stype.categorical
        df = pd.DataFrame(df_dict)
        if with_nan:
            df.iloc[0] = df.iloc[-1] = np.nan
        if create_split:
            if num_rows < 3:
                raise ValueError("Dataframe needs at least 3 rows to include"
                                 " each of train, val and test split.")
            split = ['train'] * num_rows
            split[1] = 'val'
            split[2] = 'test'
            df['split'] = split
        super().__init__(df, col_to_stype, target_col='target')
