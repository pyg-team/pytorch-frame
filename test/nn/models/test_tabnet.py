from torch_frame.data import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import TabNet


def test_tabnet():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    tabnet = TabNet(out_channels=8, col_stats=dataset.col_stats,
                    col_names_dict=tensor_frame.col_names_dict)
    out, reg = tabnet(tensor_frame, return_reg=True)
    assert out.shape == (10, 8)
    assert reg > 0
