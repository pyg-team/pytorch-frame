import torch

from torch_frame.nn import Trompt


def test_trompt():
    batch_size = 10
    in_channels = 8
    out_channels = 1
    num_cols = 5
    num_prompts = 2
    num_layers = 6
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, in_channels))
    model = Trompt(
        in_channels=in_channels,
        out_channels=out_channels,
        num_cols=num_cols,
        num_prompts=num_prompts,
        num_layers=num_layers,
    )
    out = model(x)
    assert out.shape == (batch_size, num_layers, out_channels)
