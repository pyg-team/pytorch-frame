import torch

from torch_frame.nn import TabTransformerConv


def test_tab_transformer_conv():
    batch_size = 10
    channels = 16
    num_cols = 15
    num_heads = 8
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, channels))
    conv = TabTransformerConv(channels, num_cols, num_heads=num_heads,
                              attn_dropout=0.)
    x_out = conv(x)
    assert x_out.shape == (batch_size, num_cols, channels)
