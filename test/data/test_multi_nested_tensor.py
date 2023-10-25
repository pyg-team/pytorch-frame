import random

import torch

from torch_frame.data import MultiNestedTensor


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
    assert str(
        multi_nested_tensor) == "MultiNestedTensor(num_rows: 8, num_cols: 10)"
    assert multi_nested_tensor.num_rows == num_rows
    assert multi_nested_tensor.num_cols == num_cols

    # Test multi_nested_tensor[i, j] indexing
    for i in range(num_rows):
        for j in range(num_cols):
            tensor = multi_nested_tensor[i, j]
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test multi_nested_tensor[i] indexing
    for i in range(num_rows):
        multi_nested_tensor_row = multi_nested_tensor[i]
        for j in range(num_cols):
            tensor = multi_nested_tensor_row[0, j]
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test multi_nested_tensor[List[int]] indexing
    for index in [[4], [2, 2], [4, 1, 7], [3, 7, 1, 0]]:
        multi_nested_tensor_indexed = multi_nested_tensor[index]

        for i, idx in enumerate(index):
            for j in range(num_cols):
                tensor = multi_nested_tensor_indexed[i, j]
                assert torch.allclose(tensor_mat[idx][j], tensor)
