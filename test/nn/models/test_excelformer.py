from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.nn import ExcelFormer
from torch_frame.stype import stype


def test_excelformer():
    batch_size = 10
    in_channels = 8
    out_channels = 1
    num_heads = 2
    num_layers = 6
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical])
    dataset.materialize()
    num_cols = len(dataset.col_stats) - 1
    tensor_frame = dataset.tensor_frame
    model = ExcelFormer(in_channels=in_channels, out_channels=out_channels,
                        num_cols=num_cols, num_layers=num_layers,
                        num_heads=num_heads, col_stats=dataset.col_stats,
                        col_names_dict=tensor_frame.col_names_dict)
    out_mixedup, y_mixedup = model(tensor_frame, mixup=True)
    assert out_mixedup.shape == (batch_size, out_channels)
    assert y_mixedup.shape == tensor_frame.y.shape
