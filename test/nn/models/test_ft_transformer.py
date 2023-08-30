import torch

from torch_frame.nn import FTTransformer


def test_trompt():
    batch_size = 10
    in_channels = 8
    out_channels = 1
    num_cols = 5
    num_layers = 3
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, in_channels))
    model = FTTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        num_cols=num_cols,
        num_layers=num_layers,
    )
    out = model(x)
    assert out.shape == (batch_size, out_channels)
