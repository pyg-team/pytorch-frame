import numpy as np
import pandas as pd

import torch_frame
from torch_frame import stype


class FakeDataset(torch_frame.data.Dataset):
    r"""A fake regression dataset.

    Args:
        num_rows (int): Number of rows.
        with_nan (bool): Whether include nan in the dataset.
    """
    def __init__(
        self,
        num_rows: int,
        with_nan: bool,
    ):
        df = pd.DataFrame({
            'a': np.random.randn(num_rows),
            'b': np.random.randn(num_rows),
            'c': np.random.randn(num_rows),
            'x': np.random.randint(0, 3, size=(num_rows, )),
            'y': np.random.randint(0, 3, size=(num_rows, )),
            'target': np.random.randn(num_rows),
        })
        if with_nan:
            df.iloc[0] = df.iloc[-1] = np.nan

        col_to_stype = {
            'a': stype.numerical,
            'b': stype.numerical,
            'c': stype.numerical,
            'x': stype.categorical,
            'y': stype.categorical,
            'target': stype.numerical,
        }

        super().__init__(df, col_to_stype, target_col='target')


class FakeRegressionDatasetWithNumericalColsOnly(torch_frame.data.Dataset):
    r"""A fake regression dataset with only numerical features.

    Args:
        num_rows (int): Number of rows.
        with_nan (bool): Whether include nan in the dataset.
    """
    def __init__(
        self,
        num_rows: int,
        with_nan: bool,
    ):
        df = pd.DataFrame({
            'a': np.random.randn(num_rows),
            'b': np.random.randn(num_rows),
            'c': np.random.randn(num_rows),
            'target': np.random.randn(num_rows),
        })
        if with_nan:
            df.iloc[0] = df.iloc[-1] = np.nan

        col_to_stype = {
            'a': stype.numerical,
            'b': stype.numerical,
            'c': stype.numerical,
            'target': stype.numerical,
        }

        super().__init__(df, col_to_stype, target_col='target')


class FakeClassificationDatasetWithNumericalColsOnly(torch_frame.data.Dataset):
    r"""A fake classification dataset with only numerical features.

    Args:
        num_rows (int): Number of rows.
        with_nan (bool): Whether include nan in the dataset.
    """
    def __init__(
        self,
        num_rows: int,
        with_nan: bool,
    ):
        df = pd.DataFrame({
            'a': np.random.randn(num_rows),
            'b': np.random.randn(num_rows),
            'c': np.random.randn(num_rows),
            'target': np.random.randint(0, 3, size=(num_rows, )),
        })
        if with_nan:
            df.iloc[0] = df.iloc[-1] = np.nan

        col_to_stype = {
            'a': stype.numerical,
            'b': stype.numerical,
            'c': stype.numerical,
            'target': stype.categorical,
        }

        super().__init__(df, col_to_stype, target_col='target')
