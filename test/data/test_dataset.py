import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType


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
    assert len(dataset[0.2:0.7]) == 5
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


def test_categorical_target_order():
    # Ensures that we do not swap labels in binary classification tasks.
    df = pd.DataFrame({'A': [0, 1, 1], 'B': [0, 1, 1]})
    col_to_stype = {'A': torch_frame.categorical, 'B': torch_frame.categorical}
    dataset = Dataset(df, col_to_stype, target_col='B').materialize()

    assert dataset.col_stats['A'][StatType.COUNT] == ([1, 0], [2, 1])
    assert dataset.col_stats['B'][StatType.COUNT] == ([0, 1], [1, 2])

    assert torch.equal(
        dataset.tensor_frame.x_dict[torch_frame.categorical],
        torch.tensor([[1], [0], [0]]),
    )
    assert torch.equal(dataset.tensor_frame.y, torch.tensor([0, 1, 1]))
