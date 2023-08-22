import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame.data import Dataset


def get_dataset(num_rows: int):  # TODO Use `FakeDataset` once available
    df = pd.DataFrame({'A': np.arange(num_rows), 'B': np.arange(num_rows)})
    col_to_stype = {'A': torch_frame.categorical, 'B': torch_frame.categorical}
    return Dataset(df, col_to_stype, target_col='B')


def test_index_select():
    dataset = get_dataset(num_rows=10)
    assert len(dataset) == 10

    with pytest.raises(RuntimeError, match="requires a materialized dataset"):
        dataset = dataset[:5]

    dataset = dataset.materialize()

    assert len(dataset[0]) == 1
    assert len(dataset[:5]) == 5
    assert len(dataset[[1, 2, 3]]) == 3
    assert len(dataset[torch.tensor([1, 2, 3])]) == 3
    assert len(dataset[torch.tensor(5 * [True] + 5 * [False])]) == 5


def test_shuffle():
    dataset = get_dataset(num_rows=10).materialize()

    dataset, perm = dataset.shuffle(return_perm=True)
    assert torch.equal(torch.from_numpy(dataset.df['A'].values), perm)
    x = dataset.tensor_frame.x_dict[torch_frame.categorical].view(-1)
    assert torch.equal(x, perm)


def test_col_select():
    dataset = get_dataset(num_rows=10)

    assert dataset[['B']].feat_cols == []

    with pytest.raises(RuntimeError, match="post materialization"):
        dataset.materialize()[['B']]
