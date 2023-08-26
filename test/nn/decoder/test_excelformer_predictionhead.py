import torch

from torch_frame.nn import ExcelFormerPredictionHead


def test_excelformer_predictionhead():
    batch_size = 10
    num_cols = 18
    in_channels = 8
    out_channels = 3
    x = torch.randn(batch_size, num_cols, in_channels)
    decoder = ExcelFormerPredictionHead(in_channels, out_channels, num_cols)
    y = decoder(x)
    assert y.shape == (batch_size, out_channels)
