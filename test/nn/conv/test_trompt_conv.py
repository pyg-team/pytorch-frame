import torch

from torch_frame.nn import TromptConv


def test_trompt_conv():
    batch_size = 10
    channels = 8
    num_cols = 5
    num_prompts = 2
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, channels))
    # Prompt embeddings
    x_prompt = torch.randn(size=(batch_size, num_prompts, channels))
    conv = TromptConv(channels=8, num_cols=num_cols, num_prompts=num_prompts)
    x_prompt = conv(x, x_prompt)
    assert x_prompt.shape == (batch_size, num_prompts, channels)
