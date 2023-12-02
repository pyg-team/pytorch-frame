import torch

from torch_frame.nn import CyclicEncoding


def test_cyclic_encoding_shape():
    out_size = 8
    for size in [(10, ), (10, 4), (10, 5, 8)]:
        input_tensor = torch.rand(size)
        cyclic_encoding = CyclicEncoding(out_size)
        out_tensor = cyclic_encoding(input_tensor)
        assert out_tensor.shape == input_tensor.shape + (out_size, )


def test_cyclic_encoding_values():
    out_size = 8
    for size in [(10, ), (10, 4), (10, 5, 8)]:
        cyclic_encoding = CyclicEncoding(out_size)
        input_zeros_tensor = torch.zeros(size)
        input_ones_tensor = torch.ones(size)
        out_zeros_tensor = cyclic_encoding(input_zeros_tensor)
        out_ones_tensor = cyclic_encoding(input_ones_tensor)
        assert torch.allclose(out_zeros_tensor, out_ones_tensor, atol=1e-5)
