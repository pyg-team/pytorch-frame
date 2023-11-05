import random
from typing import List, Union

import pytest
import torch
from torch import Tensor

from torch_frame.data import MultiEmbeddingTensor


def assert_equal(
    tensor_mat: List[List[Tensor]],
    multi_embedding_tensor: MultiEmbeddingTensor,
) -> None:
    assert len(tensor_mat) == multi_embedding_tensor.shape[0]
    assert len(tensor_mat[0]) == multi_embedding_tensor.shape[1]
    for i in range(multi_embedding_tensor.shape[0]):
        for j in range(multi_embedding_tensor.shape[1]):
            tensor = multi_embedding_tensor[i, j]
            assert torch.allclose(tensor_mat[i][j], tensor)


def column_select(
    tensor_mat: List[List[Tensor]],
    index: Union[List[int], slice],
) -> List[List[Tensor]]:
    new_tensor_mat = []
    for tensor_vec in tensor_mat:
        if isinstance(index, slice):
            new_tensor_mat.append(tensor_vec[index])
        else:
            new_tensor_mat.append([tensor_vec[idx] for idx in index])
    return new_tensor_mat


def test_multi_embedding_tensor_basic():
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

    multi_embedding_tensor = MultiEmbeddingTensor.from_tensor_mat(tensor_mat)
    assert (str(multi_embedding_tensor) ==
            "MultiEmbeddingTensor(num_rows=8, num_cols=10, device='cpu')")

    # Test sizes
    assert multi_embedding_tensor.shape[0] == num_rows
    assert multi_embedding_tensor.size(0) == num_rows
    assert multi_embedding_tensor.shape[1] == num_cols
    assert multi_embedding_tensor.size(1) == num_cols
    with pytest.raises(ValueError, match="not have a fixed length"):
        multi_embedding_tensor.size(2)
    with pytest.raises(IndexError, match="Dimension out of range"):
        multi_embedding_tensor.size(3)

    dense_multi_embedding_tensor = multi_embedding_tensor.to_dense(fill_value=-1)
    max_len = 0
    for i in range(multi_embedding_tensor.num_rows):
        for j in range(multi_embedding_tensor.num_cols):
            tensor = tensor_mat[i][j]
            if len(tensor) > max_len:
                max_len = len(tensor)
            assert (torch.allclose(
                dense_multi_embedding_tensor[i, j][:len(tensor)], tensor))
            assert (dense_multi_embedding_tensor[i, j][len(tensor):] == -1).all()
    assert dense_multi_embedding_tensor.shape == (multi_embedding_tensor.shape[:-1] +
                                               (max_len, ))

    # Test multi_embedding_tensor[i, j] indexing
    for i in range(-num_rows, num_rows):
        for j in range(num_cols):
            tensor = multi_embedding_tensor[i, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test multi_embedding_tensor[i] indexing
    for i in range(-num_rows, num_rows):
        multi_embedding_tensor_row = multi_embedding_tensor[i]
        assert multi_embedding_tensor_row.shape[0] == 1
        assert multi_embedding_tensor_row.shape[1] == num_cols
        for j in range(-num_cols, num_cols):
            tensor = multi_embedding_tensor_row[0, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_mat[i][j], tensor)

    # Test row slicing
    assert_equal(tensor_mat, multi_embedding_tensor[:])
    assert_equal(tensor_mat[:3], multi_embedding_tensor[:3])
    assert_equal(tensor_mat[3:], multi_embedding_tensor[3:])
    assert_equal(tensor_mat[3:5], multi_embedding_tensor[3:5])
    assert_equal(tensor_mat[-7:5], multi_embedding_tensor[-7:5])
    assert_equal(tensor_mat[-7:-1], multi_embedding_tensor[-7:-1])
    assert_equal(tensor_mat[1::2], multi_embedding_tensor[1::2])
    empty_multi_embedding_tensor = multi_embedding_tensor[5:3]
    assert empty_multi_embedding_tensor.shape[0] == 0
    assert empty_multi_embedding_tensor.shape[1] == num_cols

    # Test row narrow
    assert_equal(tensor_mat[3:3 + 2],
                 multi_embedding_tensor.narrow(dim=0, start=3, length=2))

    # Test multi_embedding_tensor[List[int]] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0], []]:
        multi_embedding_tensor_indexed = multi_embedding_tensor[index]
        assert multi_embedding_tensor_indexed.shape[0] == len(index)
        assert multi_embedding_tensor_indexed.shape[1] == num_cols
        for i, idx in enumerate(index):
            for j in range(num_cols):
                tensor = multi_embedding_tensor_indexed[i, j]
                assert torch.allclose(tensor_mat[idx][j], tensor)

    # Test multi_embedding_tensor[:, i] indexing
    for j in range(-num_cols, num_cols):
        multi_embedding_tensor_col = multi_embedding_tensor[:, j]
        assert multi_embedding_tensor_col.shape[0] == num_rows
        assert multi_embedding_tensor_col.shape[1] == 1
        assert_equal(column_select(tensor_mat, [j]), multi_embedding_tensor_col)

    # Test column List[int] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0], []]:
        assert_equal(column_select(tensor_mat, index),
                     multi_embedding_tensor[:, index])

    # Test column slicing
    assert_equal(tensor_mat, multi_embedding_tensor[:, :])
    assert_equal(column_select(tensor_mat, slice(None, 3)),
                 multi_embedding_tensor[:, :3])
    assert_equal(column_select(tensor_mat, slice(3, None)),
                 multi_embedding_tensor[:, 3:])
    assert_equal(column_select(tensor_mat, slice(3, 5)),
                 multi_embedding_tensor[:, 3:5])
    assert_equal(column_select(tensor_mat, slice(-7, 5)),
                 multi_embedding_tensor[:, -7:5])
    assert_equal(column_select(tensor_mat, slice(-7, -1)),
                 multi_embedding_tensor[:, -7:-1])
    assert_equal(column_select(tensor_mat, slice(1, None, 2)),
                 multi_embedding_tensor[:, 1::2])
    empty_multi_embedding_tensor = multi_embedding_tensor[:, 5:3]
    assert empty_multi_embedding_tensor.shape[0] == num_rows
    assert empty_multi_embedding_tensor.shape[1] == 0

    # Test column narrow
    assert_equal(column_select(tensor_mat, slice(3, 3 + 2)),
                 multi_embedding_tensor.narrow(dim=1, start=3, length=2))

    # Testing row concat
    assert_equal(
        tensor_mat,
        MultiEmbeddingTensor.cat(
            (multi_embedding_tensor[:2], multi_embedding_tensor[2:4],
             multi_embedding_tensor[4:]), dim=0),
    )
    assert_equal(
        tensor_mat,
        MultiEmbeddingTensor.cat([
            multi_embedding_tensor[i] for i in range(multi_embedding_tensor.size(0))
        ], dim=0),
    )
    assert_equal(tensor_mat, MultiEmbeddingTensor.cat([multi_embedding_tensor],
                                                   dim=0))
    with pytest.raises(RuntimeError, match="num_cols must be the same"):
        MultiEmbeddingTensor.cat([
            multi_embedding_tensor[:2],
            multi_embedding_tensor[2:4, 0],
        ], dim=0)
    with pytest.raises(RuntimeError, match="Cannot concatenate"):
        MultiEmbeddingTensor.cat([], dim=0)

    # Testing col concat
    assert_equal(
        tensor_mat,
        MultiEmbeddingTensor.cat(
            (multi_embedding_tensor[:, :2], multi_embedding_tensor[:, 2:4],
             multi_embedding_tensor[:, 4:]), dim=1),
    )
    assert_equal(
        tensor_mat,
        MultiEmbeddingTensor.cat([
            multi_embedding_tensor[:, i]
            for i in range(multi_embedding_tensor.size(1))
        ], dim=1),
    )
    assert_equal(tensor_mat, MultiEmbeddingTensor.cat([multi_embedding_tensor],
                                                   dim=1))
    with pytest.raises(RuntimeError, match="num_rows must be the same"):
        MultiEmbeddingTensor.cat([
            multi_embedding_tensor[1:],
            multi_embedding_tensor,
        ], dim=1)
    with pytest.raises(RuntimeError, match="Cannot concatenate"):
        MultiEmbeddingTensor.cat([], dim=1)

    # Testing set item
    with pytest.raises(RuntimeError, match="read-only"):
        multi_embedding_tensor[0, 0] = torch.zeros(3)

    # Testing clone
    cloned_multi_embedding_tensor = multi_embedding_tensor.clone()
    multi_embedding_tensor.values[0] = max_value + 1.0
    assert cloned_multi_embedding_tensor.values[0] != max_value + 1.0
    multi_embedding_tensor.offset[0] = -1
    assert cloned_multi_embedding_tensor.values[0] != -1


def test_multi_embedding_tensor_different_num_rows():
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
        MultiEmbeddingTensor.from_tensor_mat(tensor_mat)
