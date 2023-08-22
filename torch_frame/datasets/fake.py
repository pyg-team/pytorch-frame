import numpy as np
import pandas as pd

import torch_frame
from torch_frame import stype


class FakeDataset(torch_frame.data.Dataset):
    r"""The Titanic dataset from the `Titanic ML competition
    <https://www.kaggle.com/competitions/titanic>`_.
    The Titanic dataset is known as the MNIST equivalent for tabular learning.
    The goal is to predict which passenger survived using passenger data
    (*i.e.* gender, age, etc)."""

    url = 'https://github.com/datasciencedojo/datasets/raw/master/titanic.csv'

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
            df['a'][np.random.randint(0, 2, (num_rows, ), dtype=bool)] = np.nan
            df['x'][np.random.randint(0, 2, (num_rows, ), dtype=bool)] = np.nan
            df['target'][np.random.randint(0, 2, (num_rows, ),
                                           dtype=bool)] = np.nan

        col_to_stype = {
            'a': stype.numerical,
            'b': stype.numerical,
            'c': stype.numerical,
            'x': stype.categorical,
            'y': stype.categorical,
            'target': stype.numerical,
        }

        super().__init__(df, col_to_stype, target_col='target')
