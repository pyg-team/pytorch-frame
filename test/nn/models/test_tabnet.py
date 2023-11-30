import pytest

from torch_frame.data import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import TabNet


@pytest.mark.parametrize('batch_size', [0, 5])
def test_tabnet(batch_size):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame[:batch_size]
    out_channels = 12
    model = TabNet(out_channels=out_channels, num_layers=3,
                   split_feat_channels=8, split_attn_channels=8, gamma=1.2,
                   col_stats=dataset.col_stats,
                   col_names_dict=tensor_frame.col_names_dict)
    model.reset_parameters()
    out, reg = model(tensor_frame, return_reg=True)
    assert out.shape == (len(tensor_frame), out_channels)
    assert reg >= 0
