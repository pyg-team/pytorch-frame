import pytest
import torch

from torch_frame import TaskType, TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import MutualInformationSort


@pytest.mark.parametrize('with_nan', [True, False])
def test_mutual_information_sort(with_nan):
    task_type = TaskType.REGRESSION
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=with_nan,
                                   stypes=[stype.numerical], create_split=True,
                                   task_type=task_type)
    # modify the FakeDataset so column num_1 would have highest mutual
    # information score
    dataset.df['num_1'] = dataset.df['target'].astype(float)
    dataset.materialize()

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split('train')
    transform = MutualInformationSort(task_type)
    transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)
    out = transform(tensor_frame)

    # column num_1 ranks the first
    assert (out.col_names_dict[stype.numerical][0] == 'num_1')
    actual_first_col = out.feat_dict[stype.numerical][:, 0]
    actual_first_col_nan_mask = torch.isnan(actual_first_col)
    expected_first_col = torch.tensor(dataset.df['num_1'].values,
                                      dtype=torch.float32)
    expected_first_col_nan_mask = torch.isnan(expected_first_col)
    # if the tensor on first column contains NaNs, make sure the NaNs
    # are unchanged
    assert (torch.equal(actual_first_col_nan_mask,
                        expected_first_col_nan_mask))
    actual = actual_first_col[~actual_first_col_nan_mask]
    expected = expected_first_col[~expected_first_col_nan_mask]
    # make sure that the non NaN values are the same on first column
    assert (torch.allclose(actual, expected))

    # make sure the shapes are unchanged
    assert (set(out.col_names_dict[stype.numerical]) == set(
        tensor_frame.col_names_dict[stype.numerical]))
    assert (out.feat_dict[stype.numerical].size() == tensor_frame.feat_dict[
        stype.numerical].size())
