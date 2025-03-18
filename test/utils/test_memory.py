import pytest
import torch

from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_frame.utils import num_bytes


def test_num_bytes(get_fake_tensor_frame):
    data = torch.randn(4, 8)
    assert num_bytes(data) == 4 * 8 * 4

    data = MultiNestedTensor(
        num_rows=3,
        num_cols=2,
        values=torch.randn(12),
        offset=torch.tensor([0, 2, 4, 6, 8, 10, 12]),
    )
    assert num_bytes(data) == 12 * 4 + 7 * 8

    data = MultiEmbeddingTensor(
        num_rows=2,
        num_cols=3,
        values=torch.randn(2, 10),
        offset=torch.tensor([0, 3, 5, 10]),
    )
    assert num_bytes(data) == 2 * 10 * 4 + 4 * 8

    mapping = {
        'A': data,
        'B': data,
    }
    assert num_bytes(mapping) == 2 * num_bytes(data)

    with pytest.raises(NotImplementedError):
        num_bytes("unsupported")
