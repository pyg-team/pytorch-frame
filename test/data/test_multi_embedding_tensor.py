import random
from typing import List

import pytest
import torch

from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor


def assert_equal(
    tensor_list: List[torch.Tensor],
    met: MultiEmbeddingTensor,
) -> None:
    assert len(tensor_list) == met.num_cols
    assert len(tensor_list[0]) == met.num_rows
    for i in range(met.num_rows):
        for j in range(met.num_cols):
            # Note: tensor_list[j] is a tensor of j-th column of size
            # [num_rows, dim_emb_j]. See the docs for more info.
            assert torch.allclose(tensor_list[j][i], met[i, j])


def test_size():
    num_rows = 8
    num_cols = 3
    lengths = torch.tensor([1, 2, 3])
    offset = torch.tensor([0, 1, 3, 6])
    values = torch.rand((num_rows, lengths.sum().item()))
    met = MultiEmbeddingTensor(
        num_rows=num_rows,
        num_cols=num_cols,
        values=values,
        offset=offset,
    )

    assert met.size(0) == num_rows
    assert met.size(1) == num_cols
    with pytest.raises(IndexError, match="not have a fixed length"):
        met.size(2)
    with pytest.raises(IndexError, match="Dimension out of range"):
        met.size(3)

    assert met.shape[0] == num_rows
    assert met.shape[1] == num_cols
    assert met.shape[2] == -1


def test_from_list():
    num_rows = 2
    num_cols = 3
    tensor_list = [
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        torch.tensor([[6, 7], [8, 9]]),
        torch.tensor([[10], [11]]),
    ]
    met = MultiEmbeddingTensor.from_list(tensor_list)
    assert met.num_rows == num_rows
    assert met.num_cols == num_cols
    expected_values = torch.tensor([
        [0, 1, 2, 6, 7, 10],
        [3, 4, 5, 8, 9, 11],
    ])
    assert torch.allclose(met.values, expected_values)
    expected_offset = torch.tensor([0, 3, 5, 6])
    assert torch.allclose(met.offset, expected_offset)
    assert_equal(tensor_list, met)

    # case: empty list
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_list([])

    # case: list of non-2d tensors
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_list([torch.rand(1)])

    # case: list of tensors having different num_rows
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_list([torch.rand(2, 1), torch.rand(3, 1)])

    # case: list of tensors on different devices
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_list([
            torch.rand(2, 1, device="cpu"),
            torch.rand(2, 1, device="meta"),
        ])
