from typing import List

import numpy as np
import pandas as pd

import torch_frame
from torch_frame import TaskType, stype


class FakeDataset(torch_frame.data.Dataset):
    r"""A fake regression dataset.

    Args:
        num_rows (int): Number of rows.
        with_nan (bool): Whether include nan in the dataset.
        stypes (List[stype]): List of stype columns to include
                in the dataset. Particularly useful, when you want to
                create a dataset with only numerical or categorical
                feature columns. (default: [stype.categorical,
                stype.numerical])
        create_split (bool): Whether to create a train, val and test
                split for the fake dataset. (default: :obj:`False`)
        task_type (TaskType): Task type (default: :obj:`TaskType.REGRESSION`)

    """
    def __init__(self, num_rows: int, with_nan: bool = False,
                 stypes: List[stype] = [stype.categorical, stype.numerical],
                 create_split: bool = False,
                 task_type: TaskType = TaskType.REGRESSION):
        assert len(stypes) > 0
        if task_type == TaskType.REGRESSION:
            df_dict = {'target': np.random.randn(num_rows)}
            col_to_stype = {'target': stype.numerical}
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            df_dict = {'target': np.random.randint(0, 3, size=(num_rows, ))}
            col_to_stype = {'target': stype.categorical}
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            df_dict = {'target': np.random.randint(0, 2, size=(num_rows, ))}
            col_to_stype = {'target': stype.categorical}
        else:
            raise ValueError(
                "FakeDataset only support binary classification, "
                "multiclass classification or regression type, but"
                f" got {task_type}")
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
