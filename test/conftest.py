from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data import Dataset


@pytest.fixture()
def get_fake_tensor_frame() -> Callable:
    def _get_fake_tensor_frame(num_rows: int) -> TensorFrame:
        x_dict = {
            torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
            torch_frame.numerical: torch.randn(size=(num_rows, 2)),
        }
        col_names_dict = {
            torch_frame.categorical: ['a', 'b', 'c'],
            torch_frame.numerical: ['x', 'y'],
        }
        y = torch.randn(num_rows)

        return TensorFrame(
            x_dict=x_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    return _get_fake_tensor_frame


@pytest.fixture()
def get_fake_dataset() -> Callable:
    def _get_fake_dataset(num_rows: int, with_nan: bool = False) -> Dataset:
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

        return Dataset(
            df=df,
            col_to_stype=col_to_stype,
            target_col='target',
        )

    return _get_fake_dataset
