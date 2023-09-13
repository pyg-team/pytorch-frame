import pytest
import torch

from torch_frame import TaskType, TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import MutualInformationSort


@pytest.mark.parametrize('with_nan', [True, False])
def test_mutual_information_sort_classification(with_nan):
    dataset: Dataset = FakeDataset(
        num_rows=10, with_nan=with_nan, stypes=[stype.numerical],
        create_split=True, task_type=TaskType.MULTICLASS_CLASSIFICATION)
    # modify the FakeDataset so column c would have highest mutual information
    # score
    dataset.df['c'] = dataset.df['target'].astype(float)

    dataset.materialize()

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = MutualInformationSort(TaskType.MULTICLASS_CLASSIFICATION)
    transform.fit(train_dataset.tensor_frame)
    out = transform(tensor_frame)

    # column c ranks the first
    assert (out.col_names_dict[stype.numerical][0] == 'c')
    actual_highest_mi_score_col = out.x_dict[stype.numerical][:, 0]
    actual_highest_mi_score_col_nan_mask = torch.isnan(
        actual_highest_mi_score_col)
    expected_highest_mi_score_col = torch.tensor(dataset.df['c'].values,
                                                 dtype=torch.float32)
    expected_highest_mi_score_col_nan_mask = torch.isnan(
        expected_highest_mi_score_col)
    assert (torch.allclose(actual_highest_mi_score_col_nan_mask,
                           expected_highest_mi_score_col_nan_mask))
    actual = actual_highest_mi_score_col[~actual_highest_mi_score_col_nan_mask]
    expected = expected_highest_mi_score_col[
        ~expected_highest_mi_score_col_nan_mask]
    assert (torch.allclose(actual, expected))

    # make sure the shapes are unchanged
    assert (set(out.col_names_dict[stype.numerical]) == set(
        tensor_frame.col_names_dict[stype.numerical]))
    assert (out.x_dict[stype.numerical].size() == tensor_frame.x_dict[
        stype.numerical].size())


@pytest.mark.parametrize('with_nan', [True, False])
def test_mutual_information_sort_regression(with_nan):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=with_nan,
                                   stypes=[stype.numerical], create_split=True,
                                   task_type=TaskType.REGRESSION)
    # modify the FakeDataset so column c would have highest mutual information
    # score
    dataset.df['c'] = dataset.df['target'].astype(float)
    dataset.materialize()

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = MutualInformationSort(TaskType.REGRESSION)
    transform.fit(train_dataset.tensor_frame)
    out = transform(tensor_frame)

    # column c ranks the first
    assert (out.col_names_dict[stype.numerical][0] == 'c')
    actual_highest_mi_score_col = out.x_dict[stype.numerical][:, 0]
    actual_highest_mi_score_col_nan_mask = torch.isnan(
        actual_highest_mi_score_col)
    expected_highest_mi_score_col = torch.tensor(dataset.df['c'].values,
                                                 dtype=torch.float32)
    expected_highest_mi_score_col_nan_mask = torch.isnan(
        expected_highest_mi_score_col)
    assert (torch.allclose(actual_highest_mi_score_col_nan_mask,
                           expected_highest_mi_score_col_nan_mask))
    actual = actual_highest_mi_score_col[~actual_highest_mi_score_col_nan_mask]
    expected = expected_highest_mi_score_col[
        ~expected_highest_mi_score_col_nan_mask]
    assert (torch.allclose(actual, expected))

    # make sure the column names are unchanged
    assert (set(out.col_names_dict[stype.numerical]) == set(
        tensor_frame.col_names_dict[stype.numerical]))
    # make sure the tensor shapes are unchanged
    assert (out.x_dict[stype.numerical].size() == tensor_frame.x_dict[
        stype.numerical].size())
