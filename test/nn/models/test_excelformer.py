import torch

from torch_frame.nn import ExcelFormer


def test_excelformer():
    batch_size = 10
    in_channels = 8
    out_channels = 1
    num_cols = 5
    num_heads = 2
    num_layers = 6
    x = torch.randn(size=(batch_size, num_cols, in_channels))
    model = ExcelFormer(in_channels=in_channels, out_channels=out_channels,
                        num_layers=num_layers, num_heads=num_heads)
    out = model(x)
    assert out.shape == (batch_size, num_cols, in_channels)


test_excelformer()
