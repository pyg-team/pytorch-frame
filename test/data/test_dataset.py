import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame import stype
from torch_frame.data import DataFrameToTensorFrameConverter, Dataset
from torch_frame.data.stats import StatType
from torch_frame.datasets import FakeDataset
from torch_frame.typing import TaskType


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
    feat = dataset.tensor_frame.feat_dict[torch_frame.categorical].view(-1)
    assert torch.equal(feat, perm)


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
        dataset.tensor_frame.feat_dict[torch_frame.categorical],
        torch.tensor([[1], [0], [0], [0]]),
    )
    assert torch.equal(dataset.tensor_frame.y, torch.tensor([0, 1, 1, 1]))


def test_dataset_inductive_transform():
    dataset = FakeDataset(num_rows=10).materialize()

    df = dataset.df
    assert dataset.convert_to_tensor_frame.col_names_dict[
        torch_frame.numerical] == ['a', 'b', 'c']
    assert dataset.convert_to_tensor_frame.col_names_dict[
        torch_frame.categorical] == ['x', 'y']
    mapped_tensor_frame = dataset.convert_to_tensor_frame(df)
    for key in dataset.tensor_frame.feat_dict.keys():
        assert torch.equal(mapped_tensor_frame.feat_dict[key],
                           dataset.tensor_frame.feat_dict[key])

    # A new dataframe with an unseen categorical column
    df['x'] = 999
    unseen_tensor_frame = dataset.convert_to_tensor_frame(df)
    assert torch.eq(
        unseen_tensor_frame.feat_dict[torch_frame.categorical][:, 0],
        -1).all()


def test_converter():
    dataset = FakeDataset(
        num_rows=10,
        stypes=[stype.categorical, stype.numerical,
                stype.multicategorical]).materialize()
    convert_to_tensor_frame = DataFrameToTensorFrameConverter(
        col_to_stype=dataset.col_to_stype,
        col_stats=dataset.col_stats,
        target_col=dataset.target_col,
    )
    tf = convert_to_tensor_frame(dataset.df)
    assert tf.col_names_dict == convert_to_tensor_frame.col_names_dict
    assert len(tf) == len(dataset)


def test_multicategorical_materialization():
    data = {'a': ['A|B', 'B|C|A', '', '', 'B', 'B|A', None]}
    df = pd.DataFrame(data)
    dataset = Dataset(df, {'a': stype.multicategorical}, sep={'a': '|'})
    dataset.materialize()
    feat = dataset.tensor_frame.feat_dict[stype.multicategorical]
    assert torch.equal(feat[0, 0], torch.tensor([1, 0], device=feat.device))
    assert torch.equal(feat[1, 0], torch.tensor([0, 3, 1], device=feat.device))
    assert torch.equal(feat[2, 0], torch.tensor([2], device=feat.device))
    assert torch.equal(feat[6, 0], torch.tensor([-1], device=feat.device))
    assert StatType.MULTI_COUNT in dataset.col_stats['a']
    assert dataset.col_stats['a'][StatType.MULTI_COUNT][0] == [
        'B', 'A', '', 'C'
    ]
    assert dataset.col_stats['a'][StatType.MULTI_COUNT][1] == [4, 3, 2, 1]


@pytest.mark.parametrize('with_nan', [True, False])
def test_num_classes(with_nan):
    num_classes = 10
    target = np.arange(10)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    x = np.random.randn(10)
    if with_nan:
        target = target.astype(np.float32)
        num_classes_with_nan = 2
        target[num_classes_with_nan:] = np.nan
        num_classes = num_classes_with_nan
        task_type = TaskType.BINARY_CLASSIFICATION
    df = pd.DataFrame({"target": target, "x": x})
    dataset = Dataset(
        df,
        col_to_stype={
            "target": torch_frame.categorical,
            "x": torch_frame.numerical
        },
        target_col="target",
    ).materialize()
    assert dataset.num_classes == num_classes
    assert dataset.task_type == task_type
