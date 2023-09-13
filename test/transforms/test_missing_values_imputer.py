import pytest
import torch

from torch_frame import ImputingStrategy, TaskType, TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import MissingValuesImputer


@pytest.mark.parametrize(
    'strategy', [(ImputingStrategy.ZEROS, ImputingStrategy.MEAN),
                 (ImputingStrategy.MOST_FREQUENT, ImputingStrategy.MEDIAN),
                 (None, ImputingStrategy.ZEROS),
                 (ImputingStrategy.ZEROS, None)])
def test_missing_values_imputer(strategy):
    dataset: Dataset = FakeDataset(
        num_rows=10, with_nan=True,
        stypes=[stype.numerical, stype.categorical], create_split=True,
        task_type=TaskType.MULTICLASS_CLASSIFICATION)
    dataset.materialize()

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    nan_masks = dict()
    for col_type in tensor_frame.col_names_dict:
        nan_masks[col_type] = torch.isnan(tensor_frame.x_dict[col_type])
    strategy = {stype.categorical: strategy[0], stype.numerical: strategy[1]}
    transform = MissingValuesImputer(
        categorical_strategy=strategy[stype.categorical],
        numerical_strategy=strategy[stype.numerical])
    transform.fit(train_dataset.tensor_frame)
    out = transform(tensor_frame)

    for col_type in out.col_names_dict:
        if strategy[col_type] is None:
            continue
        # the output tensor does not contain any nans
        assert (not torch.isnan(out.x_dict[col_type]).any())
        nan_mask = nan_masks[col_type]
        # the non-nan values are unchanged.
        assert (out.x_dict[col_type][nan_mask] == tensor_frame.x_dict[col_type]
                [nan_mask]).all()
        if col_type == stype.categorical:
            # categorical features does not contain -1's
            assert (not (out.x_dict[stype.categorical] == -1).any().item())
