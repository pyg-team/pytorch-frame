import torch

from torch_frame.nn import PositionalEncoding


def test_positional_encoding_shape():
    encoding_size = 8
    input_tensor = torch.arange(10)
    positional_encoding = PositionalEncoding(encoding_size)
    out = positional_encoding(input_tensor)
    assert out.shape == input_tensor.shape + (encoding_size, )
