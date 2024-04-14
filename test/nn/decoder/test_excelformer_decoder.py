import torch

from torch_frame.nn import ExcelFormerDecoder


def test_excelformer_decoder():
    batch_size = 10
    num_cols = 8
    in_channels = 8
    out_channels = 3
    x = torch.randn(batch_size, num_cols, in_channels)
    decoder = ExcelFormerDecoder(in_channels, out_channels, num_cols)
    y = decoder(x)
    assert y.shape == (batch_size, out_channels)
