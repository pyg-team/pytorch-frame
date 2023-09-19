import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame.data import Dataset, TensorFrameConverter
from torch_frame.data.stats import StatType
from torch_frame.datasets import FakeDataset


def test_index_select():
    dataset = FakeDataset(num_rows=10)
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
    df = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10)})
    col_to_stype = {'A': torch_frame.categorical, 'B': torch_frame.categorical}
    dataset = Dataset(df, col_to_stype, target_col='B').materialize()

    dataset, perm = dataset.shuffle(return_perm=True)
    assert torch.equal(torch.from_numpy(dataset.df['A'].values), perm)
    x = dataset.tensor_frame.x_dict[torch_frame.categorical].view(-1)
    assert torch.equal(x, perm)


def test_col_select():
    dataset = FakeDataset(num_rows=10)

    assert dataset[['target']].feat_cols == []

    with pytest.raises(RuntimeError, match="post materialization"):
        dataset.materialize()[['target']]


def test_categorical_target_order():
    # Ensures that we do not swap labels in binary classification tasks.
    df = pd.DataFrame({'A': [0, 1, 1, 1], 'B': [0, 1, 1, 1]})
    col_to_stype = {'A': torch_frame.categorical, 'B': torch_frame.categorical}
    dataset = Dataset(df, col_to_stype, target_col='B').materialize()

    assert dataset.col_stats['A'][StatType.COUNT] == ([1, 0], [3, 1])
    assert dataset.col_stats['B'][StatType.COUNT] == ([0, 1], [1, 3])

    assert torch.equal(
        dataset.tensor_frame.x_dict[torch_frame.categorical],
        torch.tensor([[1], [0], [0], [0]]),
    )
    assert torch.equal(dataset.tensor_frame.y, torch.tensor([0, 1, 1, 1]))


def test_dataset_inductive_transform():
    dataset = FakeDataset(num_rows=10).materialize()

    df = dataset.df
    converter = dataset.get_tensor_frame_converter()
    assert converter.col_names_dict[torch_frame.numerical] == ['a', 'b', 'c']
    assert converter.col_names_dict[torch_frame.categorical] == ['x', 'y']
    mapped_tensor_frame = converter(df)
    for key in dataset.tensor_frame.x_dict.keys():
        assert torch.equal(mapped_tensor_frame.x_dict[key],
                           dataset.tensor_frame.x_dict[key])

    # A new dataframe with an unseen categorical column
    df['x'] = 999
    unseen_tensor_frame = converter(df)
    assert torch.eq(unseen_tensor_frame.x_dict[torch_frame.categorical][:, 0],
                    -1).all()


def test_converter():
    dataset = FakeDataset(num_rows=10).materialize()
    converter = TensorFrameConverter(
        col_to_stype=dataset.col_to_stype,
        target_col=dataset.target_col,
        col_stats=dataset.col_stats,
    )
    tf = converter(dataset.df)
    assert tf.col_names_dict == converter.col_names_dict
    assert len(tf) == len(dataset)
