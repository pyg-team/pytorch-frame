import random
from typing import List

import torch
from torch import Tensor

from torch_frame.data import MultiNestedTensor


def assert_equal(tensor_mat: List[List[Tensor]],
                 multi_nested_tensor: MultiNestedTensor):
    assert len(tensor_mat) == multi_nested_tensor.num_rows
    assert len(tensor_mat[0]) == multi_nested_tensor.num_cols
    for i in range(multi_nested_tensor.num_rows):
        for j in range(multi_nested_tensor.num_cols):
            tensor = multi_nested_tensor[i, j]
            assert torch.allclose(tensor_mat[i][j], tensor)


def test_multi_nested_tensor_basic():
    num_rows = 8
    num_cols = 10
    tensor_mat = []
    for _ in range(num_rows):
        tensor_list = []
        for _ in range(num_cols):
            length = random.randint(0, 10)
            tensor_list.append(torch.randint(0, 100, size=(length, )))
        tensor_mat.append(tensor_list)

    multi_nested_tensor = MultiNestedTensor.from_tensor_mat(tensor_mat)
    assert str(multi_nested_tensor
               ) == "MultiNestedTensor(num_rows=8, num_cols=10, device=cpu)"
    assert multi_nested_tensor.num_rows == num_rows
    assert multi_nested_tensor.num_cols == num_cols

    # Test multi_nested_tensor[i, j] indexing
    for i in range(-num_rows, num_rows):
        for j in range(num_cols):
            tensor = multi_nested_tensor[i, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test multi_nested_tensor[i] indexing
    for i in range(-num_rows, num_rows):
        multi_nested_tensor_row = multi_nested_tensor[i]
        assert multi_nested_tensor_row.num_rows == 1
        assert multi_nested_tensor_row.num_cols == num_cols
        for j in range(-num_cols, num_cols):
            tensor = multi_nested_tensor_row[0, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test row slicing
    assert_equal(tensor_mat, multi_nested_tensor[:])
    assert_equal(tensor_mat[:3], multi_nested_tensor[:3])
    assert_equal(tensor_mat[3:], multi_nested_tensor[3:])
    assert_equal(tensor_mat[3:5], multi_nested_tensor[3:5])
    assert_equal(tensor_mat[-7:5], multi_nested_tensor[-7:5])
    assert_equal(tensor_mat[-7:-1], multi_nested_tensor[-7:-1])
    assert_equal(tensor_mat[1::2], multi_nested_tensor[1::2])
    empty_multi_nested_tensor = multi_nested_tensor[5:3]
    assert empty_multi_nested_tensor.num_rows == 0
    assert empty_multi_nested_tensor.num_cols == num_cols

    # Test multi_nested_tensor[List[int]] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0]]:
        multi_nested_tensor_indexed = multi_nested_tensor[index]
        assert multi_nested_tensor_indexed.num_rows == len(index)
        assert multi_nested_tensor_indexed.num_cols == num_cols
        for i, idx in enumerate(index):
            for j in range(num_cols):
                tensor = multi_nested_tensor_indexed[i, j]
                assert torch.allclose(tensor_mat[idx][j], tensor)

    # Test multi_nested_tensor[:, i] indexing
    for j in range(-num_cols, num_cols):
        multi_nested_tensor_col = multi_nested_tensor[:, j]
        assert multi_nested_tensor_col.num_rows == num_rows
        assert multi_nested_tensor_col.num_cols == 1
        for i in range(-num_rows, num_rows):
            tensor = multi_nested_tensor_col[i, 0]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # TODO: Test column List[int] indexing and slicing once supported.
