from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeNumericalDataset
from torch_frame.nn import ExcelFormer


def test_excelformer():
    batch_size = 10
    in_channels = 8
    out_channels = 1
    num_heads = 2
    num_layers = 6
    dataset: Dataset = FakeNumericalDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    num_cols = len(dataset.col_stats) - 1
    tensor_frame = dataset.tensor_frame
    model = ExcelFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        num_cols=num_cols,
        num_layers=num_layers,
        num_heads=num_heads,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    out = model(tensor_frame)
    assert out.shape == (batch_size, num_cols, in_channels)
