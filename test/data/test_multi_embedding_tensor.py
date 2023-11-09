import random
from typing import List, Optional, Tuple

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


def get_fake_multi_embedding_tensor(
    num_rows: int,
    num_cols: int,
) -> Tuple[MultiEmbeddingTensor, List[torch.Tensor]]:
    tensor_list = []
    for _ in range(num_cols):
        embedding_dim = random.randint(1, 5)
        print(embedding_dim)
        tensor = torch.randn((num_rows, embedding_dim))
        tensor_list.append(tensor)
    return MultiEmbeddingTensor.from_tensor_list(tensor_list), tensor_list


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


def test_from_tensor_list():
    num_rows = 2
    num_cols = 3
    tensor_list = [
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        torch.tensor([[6, 7], [8, 9]]),
        torch.tensor([[10], [11]]),
    ]
    met = MultiEmbeddingTensor.from_tensor_list(tensor_list)
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
        MultiEmbeddingTensor.from_tensor_list([])

    # case: list of non-2d tensors
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_tensor_list([torch.rand(1)])

    # case: list of tensors having different num_rows
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_tensor_list(
            [torch.rand(2, 1), torch.rand(3, 1)])

    # case: list of tensors on different devices
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.from_tensor_list([
            torch.rand(2, 1, device="cpu"),
            torch.rand(2, 1, device="meta"),
        ])


def test_index():
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=2,
        num_cols=3,
    )
    # case met[i, j]: a tuple of two integers
    assert_equal(tensor_list, met)


def test_clone():
    met, _ = get_fake_multi_embedding_tensor(
        num_rows=2,
        num_cols=3,
    )
    met_clone = met.clone()
    met.values[0, 0] = 12345.
    assert met_clone.values[0, 0] != 12345.
    met.offset[0] = -1
    assert met_clone.offset[0] != -1


def test_cat():
    # case: dim=0
    tensor_list1 = [
        torch.tensor([[0, 1, 2], [6, 7, 8]]),
        torch.tensor([[3, 4], [9, 10]]),
        torch.tensor([[5], [11]]),
    ]
    met1 = MultiEmbeddingTensor.from_tensor_list(tensor_list1)
    tensor_list2 = [
        torch.tensor([[12, 13, 14]]),
        torch.tensor([[15, 16]]),
        torch.tensor([[17]]),
    ]
    met2 = MultiEmbeddingTensor.from_tensor_list(tensor_list2)
    for xs in [[met1, met2], (met1, met2)]:
        met_cat = MultiEmbeddingTensor.cat(xs, dim=0)
        assert met_cat.num_rows == met1.num_rows + met2.num_rows
        assert met_cat.num_cols == met1.num_cols == met2.num_cols
        expected_values = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
            ]
        )
        assert torch.allclose(met_cat.values, expected_values)
        assert torch.allclose(met_cat.offset, met1.offset)
        assert torch.allclose(met_cat.offset, met2.offset)

        met_cat = MultiEmbeddingTensor.cat((met1, met2), dim=0)
        assert met_cat.num_rows == met1.num_rows + met2.num_rows
        assert met_cat.num_cols == met1.num_cols == met2.num_cols
        expected_values = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
            ]
        )
        assert torch.allclose(met_cat.values, expected_values)
        assert torch.allclose(met_cat.offset, met1.offset)
        assert torch.allclose(met_cat.offset, met2.offset)

    # case: dim=1
    tensor_list1 = [
        torch.tensor([[0, 1, 2], [10, 11, 12]]),
        torch.tensor([[3, 4], [13, 14]]),
        torch.tensor([[5], [15]]),
    ]
    met1 = MultiEmbeddingTensor.from_tensor_list(tensor_list1)
    tensor_list2 = [
        torch.tensor([[6, 7, 8, 9], [16, 17, 18, 19]]),
    ]
    met2 = MultiEmbeddingTensor.from_tensor_list(tensor_list2)
    for xs in [[met1, met2], (met1, met2)]:
        met_cat = MultiEmbeddingTensor.cat(xs, dim=1)
        assert met_cat.num_rows == met1.num_rows
        assert met_cat.num_rows == met2.num_rows
        assert met_cat.num_cols == met1.num_cols + met2.num_cols
        expected_values = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            ]
        )
        assert torch.allclose(met_cat.values, expected_values)
        expected_offset = torch.tensor([0, 3, 5, 6, 10])
        assert torch.allclose(met_cat.offset, expected_offset)
