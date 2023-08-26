import torch

from torch_frame.nn import ExcelFormerConv


def test_trompt_conv():
    batch_size = 10
    channels = 8
    num_cols = 15
    num_heads = 8
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, channels))

    conv = ExcelFormerConv(channels=8, num_cols=num_cols, num_heads=num_heads)
    x_prompt = conv(x)
    assert x_prompt.shape == (batch_size, num_cols, channels)