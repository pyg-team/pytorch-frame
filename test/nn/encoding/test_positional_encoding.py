import torch

from torch_frame.nn import PositionalEncoding


def test_positional_encoding():
    encoding_size = 8
    for size in [(10, ), (10, 4), (10, 5, 8)]:
        input_tensor = torch.randint(0, 10, size=size)
        positional_encoding = PositionalEncoding(encoding_size)
        out_tensor = positional_encoding(input_tensor)
        assert out_tensor.shape == input_tensor.shape + (encoding_size, )
