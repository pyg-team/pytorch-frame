import torch

from torch_frame import TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import MutualInformationSort


def test_mutual_information_sort_classification():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical], create_split=True,
                                   task_type="classification")
    # modify the FakeDataset so column c would have highest mutual information
    # score
    dataset.df['c'] = dataset.df['target'].astype(float)

    dataset.materialize()

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = MutualInformationSort(train_dataset.tensor_frame,
                                      "classification")
    out = transform(tensor_frame)

    # column c ranks the first
    assert (out.col_names_dict[stype.numerical][0] == 'c')
    assert (torch.eq(out.x_dict[stype.numerical][:, 0],
                     torch.tensor(dataset.df['c'].values,
                                  dtype=torch.float32)).all())

    # make sure the shapes are unchanged
    assert (set(out.col_names_dict[stype.numerical]) == set(
        tensor_frame.col_names_dict[stype.numerical]))
    assert (out.x_dict[stype.numerical].size() == tensor_frame.x_dict[
        stype.numerical].size())


def test_mutual_information_sort_regression():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical], create_split=True,
                                   task_type="regression")
    # modify the FakeDataset so column c would have highest mutual information
    # score
    dataset.df['c'] = dataset.df['target'].astype(float)
    dataset.materialize()

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = MutualInformationSort(train_dataset.tensor_frame, "regression")
    out = transform(tensor_frame)

    # column c ranks the first
    assert (out.col_names_dict[stype.numerical][0] == 'c')
    assert (torch.eq(out.x_dict[stype.numerical][:, 0],
                     torch.tensor(dataset.df['c'].values,
                                  dtype=torch.float32)).all())

    # make sure the column names are unchanged
    assert (set(out.col_names_dict[stype.numerical]) == set(
        tensor_frame.col_names_dict[stype.numerical]))
    # make sure the tensor shapes are unchanged
    assert (out.x_dict[stype.numerical].size() == tensor_frame.x_dict[
        stype.numerical].size())
