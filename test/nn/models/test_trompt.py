import torch

from torch_frame.nn import Trompt


def test_trompt():
    batch_size = 10
    in_channels = 8
    out_channels = 2
    num_cols = 5
    num_prompts = 2
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, in_channels))
    model = Trompt(
        in_channels=in_channels,
        out_channels=out_channels,
        num_cols=num_cols,
        num_prompts=num_prompts,
        num_layers=6,
    )
    outs = model(x)
    for out in outs:
        assert out.shape == (batch_size, out_channels)
