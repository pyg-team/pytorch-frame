import random
from typing import List

import pytest
import torch
from torch import Tensor

from torch_frame.data import MultiNestedTensor


def assert_equal(tensor_mat: List[List[Tensor]],
                 multi_nested_tensor: MultiNestedTensor):
    assert len(tensor_mat) == multi_nested_tensor.shape[0]
    assert len(tensor_mat[0]) == multi_nested_tensor.shape[1]
    for i in range(multi_nested_tensor.shape[0]):
        for j in range(multi_nested_tensor.shape[1]):
            tensor = multi_nested_tensor[i, j]
            assert torch.allclose(tensor_mat[i][j], tensor)


def test_multi_nested_tensor_basic():
    num_rows = 8
    num_cols = 10
    max_value = 100
    tensor_mat = []
    for _ in range(num_rows):
        tensor_list = []
        for _ in range(num_cols):
            length = random.randint(0, 10)
            tensor_list.append(torch.randint(0, max_value, size=(length, )))
        tensor_mat.append(tensor_list)

    multi_nested_tensor = MultiNestedTensor.from_tensor_mat(tensor_mat)
    assert (str(multi_nested_tensor) ==
            "MultiNestedTensor(num_rows=8, num_cols=10, device='cpu')")

    # Test sizes
    assert multi_nested_tensor.shape[0] == num_rows
    assert multi_nested_tensor.size(0) == num_rows
    assert multi_nested_tensor.shape[1] == num_cols
    assert multi_nested_tensor.size(1) == num_cols
    with pytest.raises(ValueError, match="not have a fixed length"):
        multi_nested_tensor.size(2)
    with pytest.raises(IndexError, match="Dimension out of range"):
        multi_nested_tensor.size(3)

    # Test multi_nested_tensor[i, j] indexing
    for i in range(-num_rows, num_rows):
        for j in range(num_cols):
            tensor = multi_nested_tensor[i, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test multi_nested_tensor[i] indexing
    for i in range(-num_rows, num_rows):
        multi_nested_tensor_row = multi_nested_tensor[i]
        assert multi_nested_tensor_row.shape[0] == 1
        assert multi_nested_tensor_row.shape[1] == num_cols
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
    assert empty_multi_nested_tensor.shape[0] == 0
    assert empty_multi_nested_tensor.shape[1] == num_cols

    # Test row narrow
    assert_equal(tensor_mat[3:3 + 2],
                 multi_nested_tensor.narrow(dim=0, start=3, length=2))

    # Test multi_nested_tensor[List[int]] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0]]:
        multi_nested_tensor_indexed = multi_nested_tensor[index]
        assert multi_nested_tensor_indexed.shape[0] == len(index)
        assert multi_nested_tensor_indexed.shape[1] == num_cols
        for i, idx in enumerate(index):
            for j in range(num_cols):
                tensor = multi_nested_tensor_indexed[i, j]
                assert torch.allclose(tensor_mat[idx][j], tensor)

    # Test multi_nested_tensor[:, i] indexing
    for j in range(-num_cols, num_cols):
        multi_nested_tensor_col = multi_nested_tensor[:, j]
        assert multi_nested_tensor_col.shape[0] == num_rows
        assert multi_nested_tensor_col.shape[1] == 1
        for i in range(-num_rows, num_rows):
            tensor = multi_nested_tensor_col[i, 0]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test column List[int] indexing
    with pytest.raises(NotImplementedError):
        # TODO: Add proper test once implemented
        multi_nested_tensor[:, [2, 4, 6]]

    # Test column slicing
    with pytest.raises(NotImplementedError):
        # TODO: Add proper test once implemented
        multi_nested_tensor[:, 4:8]

    # Test column narrow
    with pytest.raises(NotImplementedError):
        # TODO: Add proper test once implemented
        multi_nested_tensor.narrow(dim=1, start=3, length=2)

    # Testing row concat
    assert_equal(
        tensor_mat,
        MultiNestedTensor.cat(
            (multi_nested_tensor[:2], multi_nested_tensor[2:4],
             multi_nested_tensor[4:]), dim=0),
    )
    assert_equal(
        tensor_mat,
        MultiNestedTensor.cat([
            multi_nested_tensor[i] for i in range(multi_nested_tensor.size(0))
        ], dim=0),
    )
    assert_equal(tensor_mat, MultiNestedTensor.cat([multi_nested_tensor]))
    with pytest.raises(RuntimeError, match="num_cols must be the same"):
        MultiNestedTensor.cat([
            multi_nested_tensor[:2],
            multi_nested_tensor[2:4, 0],
        ], dim=0)

    # Testing col concat
    with pytest.raises(NotImplementedError):
        # TODO: Add proper test once implemented
        MultiNestedTensor.cat([
            multi_nested_tensor[:, j]
            for j in range(multi_nested_tensor.size(1))
        ], dim=1),

    # Testing clone
    cloned_multi_nested_tensor = multi_nested_tensor.clone()
    multi_nested_tensor.values[0] = max_value + 1.0
    assert cloned_multi_nested_tensor.values[0] != max_value + 1.0
    multi_nested_tensor.offset[0] = -1
    assert cloned_multi_nested_tensor.values[0] != -1


def test_multi_nested_tensor_different_num_rows():
    tensor_mat = [
        [torch.tensor([1, 2, 3]),
         torch.tensor([4, 5])],
        [torch.tensor([6, 7]),
         torch.tensor([8, 9, 10]),
         torch.tensor([11])],
    ]

    with pytest.raises(
            RuntimeError,
            match="The length of each row must be the same",
    ):
        MultiNestedTensor.from_tensor_mat(tensor_mat)
