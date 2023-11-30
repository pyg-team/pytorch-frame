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
        input_tensor = torch.randint(0, 2, size=size)
        cyclic_encoding = CyclicEncoding(out_size)
        out_tensor = cyclic_encoding(input_tensor)
        assert torch.allclose(out_tensor[..., :4],
                              torch.zeros_like(out_tensor[..., :4]), atol=1e-5)
        assert torch.all(torch.abs(out_tensor[..., 4:] - 1) < 1e-5)
        assert out_tensor.shape == input_tensor.shape + (out_size, )
