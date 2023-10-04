import pytest
import torch

from torch_frame import TensorFrame, stype, TaskType
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import CatToNumTransform


@pytest.mark.parametrize('with_nan', [True, False])
def test_ordered_target_statistics_encoder_on_categorical_only_dataset(
        with_nan):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=with_nan,
                                   stypes=[stype.categorical],
                                   create_split=True)
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    categorical_features = dataset.tensor_frame.col_names_dict[
        stype.categorical]
    # the dataset only contains categorical features
    assert (total_cols == len(categorical_features))

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    for col in dataset.feat_cols:
        col_stats = train_dataset.col_stats[col]
        # all the original columns are categorical
        for stat in StatType.stats_for_stype(stype.numerical):
            assert (stat not in col_stats)
        for stat in StatType.stats_for_stype(stype.categorical):
            assert (stat in col_stats)
    transform = CatToNumTransform()
    transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)
    transformed_col_stats = transform.transformed_stats
    for col in transformed_col_stats:
        # ensure that the transformed col stats contain
        # only numerical col stats
        for stat in StatType.stats_for_stype(stype.numerical):
            assert (stat in transformed_col_stats[col])
        for stat in StatType.stats_for_stype(stype.categorical):
            assert (stat not in transformed_col_stats[col])
    out = transform(tensor_frame)
    # assert that there are no categorical features
    assert (stype.categorical not in out.col_names_dict)
    assert (stype.categorical not in out.feat_dict)

    # assert that all features are numerical
    assert (len(out.col_names_dict[stype.numerical]) == total_cols)

    # assert that all categorical features are turned into numerical features
    assert (out.col_names_dict[stype.numerical] == categorical_features)


@pytest.mark.parametrize('task_type', [TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION])
def test_ordered_target_statistics_encoder(task_type):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical, stype.categorical],
                                   task_type= task_type,
                                   create_split=True)
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    total_num_cols = len(dataset.tensor_frame.col_names_dict[stype.numerical])
    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = CatToNumTransform()
    transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)
    transformed_col_stats = transform.transformed_stats
    for col in transformed_col_stats:
        # ensure that the transformed col stats contain
        # only numerical col stats
        for stat in StatType.stats_for_stype(stype.numerical):
            assert (stat in transformed_col_stats[col])
        for stat in StatType.stats_for_stype(stype.categorical):
            assert (stat not in transformed_col_stats[col])
    out = transform(tensor_frame)
    # assert that there are no categorical features
    assert (stype.categorical not in out.col_names_dict)
    assert (stype.categorical not in out.feat_dict)

    if task_type != TaskType.MULTICLASS_CLASSIFICATION:
        # assert that all features are numerical
        assert (len(out.col_names_dict[stype.numerical]) == total_cols)
        assert (dataset.tensor_frame.col_names_dict[stype.categorical] ==
        out.col_names_dict[stype.numerical][total_num_cols:])
    else:
        assert (len(out.col_names_dict[stype.numerical]) == total_num_cols + (dataset.num_classes - 1) * (total_cols - total_num_cols))

    # assert that the numerical features are unchanged
    assert (torch.eq(dataset.tensor_frame.feat_dict[stype.numerical],
                     out.feat_dict[stype.numerical][:, :total_num_cols]).all())
    assert (dataset.tensor_frame.col_names_dict[stype.numerical] ==
            out.col_names_dict[stype.numerical][:total_num_cols])
