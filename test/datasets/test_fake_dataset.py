import pytest
import torch

import torch_frame
from torch_frame.datasets import FakeDataset


@pytest.mark.parametrize('with_nan', [True, False])
def test_fake_dataset(with_nan):
    num_rows = 20
    dataset = FakeDataset(num_rows=num_rows, with_nan=with_nan)
    assert str(dataset) == 'FakeDataset()'
    assert len(dataset) == num_rows
    assert dataset.feat_cols == ['a', 'b', 'c', 'x', 'y']
    assert dataset.target_col == 'target'

    dataset = dataset.materialize()
    tensor_frame = dataset.tensor_frame
    x_num = tensor_frame.x_dict[torch_frame.numerical]
    assert x_num.dtype == torch.float
    assert x_num.size() == (num_rows, 3)
    if with_nan:
        assert torch.isnan(x_num).any()
    else:
        assert (~torch.isnan(x_num)).all()

    x_cat = tensor_frame.x_dict[torch_frame.categorical]
    assert x_cat.dtype == torch.long
    assert x_cat.size() == (num_rows, 2)
    if with_nan:
        assert (x_cat == -1).any()
    else:
        assert (x_cat >= 0).all()
