import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import DataFrameToTensorFrameConverter, Dataset
from torch_frame.data.dataset import canonicalize_col_to_pattern
from torch_frame.data.stats import StatType
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder
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
    df = pd.DataFrame({'cat_1': np.arange(10), 'cat_2': np.arange(10)})
    col_to_stype = {
        'cat_1': torch_frame.categorical,
        'cat_2': torch_frame.categorical
    }
    dataset = Dataset(df, col_to_stype, target_col='cat_2').materialize()

    dataset, perm = dataset.shuffle(return_perm=True)
    assert torch.equal(torch.from_numpy(dataset.df['cat_1'].values), perm)
    feat = dataset.tensor_frame.feat_dict[torch_frame.categorical].view(-1)
    assert torch.equal(feat, perm)


def test_col_select():
    dataset = FakeDataset(num_rows=10)

    assert dataset[['target']].feat_cols == []

    with pytest.raises(RuntimeError, match="post materialization"):
        dataset.materialize()[['target']]


def test_categorical_target_order():
    # Ensures that we do not swap labels in binary classification tasks.
    df = pd.DataFrame({'cat_1': [0, 1, 1, 1], 'cat_2': [0, 1, 1, 1]})
    col_to_stype = {
        'cat_1': torch_frame.categorical,
        'cat_2': torch_frame.categorical
    }
    dataset = Dataset(df, col_to_stype, target_col='cat_2').materialize()

    assert dataset.col_stats['cat_1'][StatType.COUNT] == ([1, 0], [3, 1])
    assert dataset.col_stats['cat_2'][StatType.COUNT] == ([0, 1], [1, 3])

    assert torch.equal(
        dataset.tensor_frame.feat_dict[torch_frame.categorical],
        torch.tensor([[1], [0], [0], [0]]),
    )
    assert torch.equal(dataset.tensor_frame.y, torch.tensor([0, 1, 1, 1]))


def test_dataset_inductive_transform():
    dataset = FakeDataset(num_rows=10).materialize()

    df = dataset.df
    assert dataset.convert_to_tensor_frame.col_names_dict[
        torch_frame.numerical] == ['num_1', 'num_2', 'num_3']
    assert dataset.convert_to_tensor_frame.col_names_dict[
        torch_frame.categorical] == ['cat_1', 'cat_2']
    mapped_tensor_frame = dataset.convert_to_tensor_frame(df)
    for key in dataset.tensor_frame.feat_dict.keys():
        assert torch.equal(mapped_tensor_frame.feat_dict[key],
                           dataset.tensor_frame.feat_dict[key])

    # A new dataframe with an unseen categorical column
    df['cat_1'] = 999
    unseen_tensor_frame = dataset.convert_to_tensor_frame(df)
    assert torch.eq(
        unseen_tensor_frame.feat_dict[torch_frame.categorical][:, 0],
        -1).all()


def test_materalization_and_converter():
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(1),
        batch_size=8,
    )
    dataset_stypes = [
        torch_frame.categorical,
        torch_frame.numerical,
        torch_frame.multicategorical,
        torch_frame.sequence_numerical,
        torch_frame.timestamp,
        torch_frame.text_embedded,
        torch_frame.embedding,
    ]
    dataset = FakeDataset(
        num_rows=10,
        stypes=dataset_stypes,
        col_to_text_embedder_cfg=text_embedder_cfg,
    ).materialize()
    expected_parent_feat_size = dict()
    stype_num_cols = dict()
    for col, stype in dataset.col_to_stype.items():
        if col != dataset.target_col:
            stype_num_cols[stype] = stype_num_cols.get(stype, 0) + 1
    for stype in dataset_stypes:
        expected_parent_feat_size[
            stype.parent] = expected_parent_feat_size.get(
                stype.parent, 0) + stype_num_cols[stype]

    tf = dataset.tensor_frame
    assert len(tf) == len(dataset)
    for stype in dataset_stypes:
        # Test child stypes are mapped to parent stypes
        if stype.parent != stype:
            assert stype not in tf.feat_dict
            assert stype not in tf.col_names_dict
            assert stype.parent in tf.feat_dict
            assert stype.parent in tf.col_names_dict
        # Test parent stypes contain columns from all child types
        else:
            assert tf.feat_dict[stype].size(
                1) == expected_parent_feat_size[stype]
            assert len(
                tf.col_names_dict[stype]) == expected_parent_feat_size[stype]
    convert_to_tensor_frame = DataFrameToTensorFrameConverter(
        col_to_stype=dataset.col_to_stype,
        col_stats=dataset.col_stats,
        target_col=dataset.target_col,
        col_to_sep=dataset.col_to_sep,
        col_to_time_format=dataset.col_to_time_format,
        col_to_text_embedder_cfg=dataset.col_to_text_embedder_cfg,
    )
    tf = convert_to_tensor_frame(dataset.df)
    assert tf.col_names_dict == convert_to_tensor_frame.col_names_dict
    assert len(tf) == len(dataset)
    for stype in dataset_stypes:
        # Test child stypes are mapped to parent stypes
        if stype.parent != stype:
            assert stype not in tf.feat_dict
            assert stype not in tf.col_names_dict
            assert stype.parent in tf.feat_dict
            assert stype.parent in tf.col_names_dict
        # Test parent stypes contain columns from all child types
        else:
            assert tf.feat_dict[stype].size(
                1) == expected_parent_feat_size[stype]
            assert len(
                tf.col_names_dict[stype]) == expected_parent_feat_size[stype]
    assert tf.y is not None

    df_without_target = dataset.df.drop(dataset.target_col, axis=1)
    tf_without_target = convert_to_tensor_frame(df_without_target)
    assert tf.col_names_dict == convert_to_tensor_frame.col_names_dict
    assert len(tf) == len(dataset)
    assert tf_without_target.y is None


def test_multicategorical_materialization():
    data = {'multicat': ['A|B', 'B|C|A', '', 'B', 'B|A|A', None]}
    df = pd.DataFrame(data)
    dataset = Dataset(df, {'multicat': torch_frame.multicategorical},
                      col_to_sep={'multicat': '|'})
    dataset.materialize()
    feat = dataset.tensor_frame.feat_dict[torch_frame.multicategorical]
    assert torch.equal(feat[0, 0].sort().values,
                       torch.tensor([1, 0]).sort().values)
    assert torch.equal(feat[1, 0].sort().values,
                       torch.tensor([0, 2, 1]).sort().values)
    assert feat[2, 0].numel() == 0
    assert torch.equal(feat[3, 0].sort().values,
                       torch.tensor([0]).sort().values)
    assert torch.equal(feat[4, 0].sort().values,
                       torch.tensor([0, 1]).sort().values)
    assert torch.equal(feat[5, 0].sort().values,
                       torch.tensor([-1]).sort().values)
    assert StatType.MULTI_COUNT in dataset.col_stats['multicat']
    assert (dataset.col_stats['multicat'][StatType.MULTI_COUNT][0] == [
        'B', 'A', 'C'
    ])
    assert dataset.col_stats['multicat'][StatType.MULTI_COUNT][1] == [4, 3, 1]


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
    df = pd.DataFrame({"target": target, "num": x})
    dataset = Dataset(
        df,
        col_to_stype={
            "target": torch_frame.categorical,
            "num": torch_frame.numerical
        },
        target_col="target",
    ).materialize()
    assert dataset.num_classes == num_classes
    assert dataset.task_type == task_type


def test_canonicalize_col_to_pattern():
    col_to_sep = '|'
    columns = ['col_1', 'col_2']
    assert {
        'col_1': '|',
        'col_2': '|'
    } == canonicalize_col_to_pattern("col_to_sep", col_to_sep, columns,
                                     requires_all_inclusive=True)

    col_to_sep = {'col_1': '|', 'col_2': ','}
    columns = ['col_1', 'col_2']
    assert {
        'col_1': '|',
        'col_2': ','
    } == canonicalize_col_to_pattern("col_to_sep", col_to_sep, columns,
                                     requires_all_inclusive=True)

    col_to_sep = {'col_1': '|'}
    columns = ['col_1', 'col_2']
    with pytest.raises(ValueError, match='col_to_sep requires all columns'):
        canonicalize_col_to_pattern("col_to_sep", col_to_sep, columns,
                                    requires_all_inclusive=True)

    col_to_sep = {'col_1': '|'}
    columns = ['col_1', 'col_2']
    assert {
        'col_1': '|',
        'col_2': None
    } == canonicalize_col_to_pattern("col_to_sep", col_to_sep, columns,
                                     requires_all_inclusive=False)


def test_col_to_pattern_raise_error():
    with pytest.raises(TypeError, match="col_to_sep"):
        dataset = FakeDataset(num_rows=10,
                              stypes=[torch_frame.multicategorical])
        Dataset(dataset.df, dataset.col_to_stype, dataset.target_col,
                col_to_sep=2)

    with pytest.raises(TypeError, match="col_to_text_embedder_cfg"):
        FakeDataset(num_rows=10, stypes=[torch_frame.text_embedded])

    with pytest.raises(TypeError, match="col_to_text_tokenizer_cfg"):
        FakeDataset(num_rows=10, stypes=[torch_frame.text_tokenized])

    with pytest.raises(TypeError, match="col_to_time_format"):
        dataset = FakeDataset(num_rows=10, stypes=[torch_frame.timestamp])
        Dataset(dataset.df, dataset.col_to_stype, dataset.target_col,
                col_to_time_format=2)
