import random

import torch
import pytest

from torch_frame.data import MultiNestedTensor


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
    assert str(multi_nested_tensor
               ) == "MultiNestedTensor(num_rows=8, num_cols=10, device='cpu')"

    # Test sizes
    assert multi_nested_tensor.num_rows == num_rows
    assert multi_nested_tensor.size(0) == num_rows
    assert multi_nested_tensor.num_cols == num_cols
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
        assert multi_nested_tensor_row.num_rows == 1
        assert multi_nested_tensor_row.num_cols == num_cols
        for j in range(-num_cols, num_cols):
            tensor = multi_nested_tensor_row[0, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test multi_nested_tensor[List[int]] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0]]:
        multi_nested_tensor_indexed = multi_nested_tensor[index]
        assert multi_nested_tensor_indexed.num_rows == len(index)
        assert multi_nested_tensor_indexed.num_cols == num_cols
        for i, idx in enumerate(index):
            for j in range(num_cols):
                tensor = multi_nested_tensor_indexed[i, j]
                assert torch.allclose(tensor_mat[idx][j], tensor)

    cloned_multi_nested_tensor = multi_nested_tensor.clone()

    multi_nested_tensor.values[0] = max_value + 1.0
    assert cloned_multi_nested_tensor.values[0] != max_value + 1.0
    multi_nested_tensor.offset[0] = -1
    assert cloned_multi_nested_tensor.values[0] != -1


def test_multi_nested_tensor_different_num_rows():
    tensor_mat = [
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        [torch.tensor([6, 7]), torch.tensor([8, 9, 10]), torch.tensor([11])],
    ]

    with pytest.raises(
        RuntimeError,
        match="The length of each row must be the same",
    ):
        MultiNestedTensor.from_tensor_mat(tensor_mat)
