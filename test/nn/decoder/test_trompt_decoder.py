import torch

from torch_frame.nn import TromptDecoder


def test_trompt_decoder():
    batch_size = 10
    num_prompts = 2
    in_channels = 8
    out_channels = 1
    x_prompt = torch.randn(batch_size, num_prompts, in_channels)
    decoder = TromptDecoder(in_channels, out_channels, num_prompts)
    y = decoder(x_prompt)
    assert y.shape == (batch_size, out_channels)
