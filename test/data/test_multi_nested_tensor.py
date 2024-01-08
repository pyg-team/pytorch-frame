import random
from typing import List, Union

import pytest
import torch
from torch import Tensor

from torch_frame.data import MultiNestedTensor
from torch_frame.testing import withCUDA


def assert_equal(tensor_mat: List[List[Tensor]],
                 multi_nested_tensor: MultiNestedTensor):
    assert len(tensor_mat) == multi_nested_tensor.shape[0]
    assert len(tensor_mat[0]) == multi_nested_tensor.shape[1]
    for i in range(multi_nested_tensor.shape[0]):
        for j in range(multi_nested_tensor.shape[1]):
            tensor = multi_nested_tensor[i, j]
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


def test_fillna_col():
    # Creat a MultiNestedTensor containing all -1's
    # In MultiNestedTensor with torch.long dtype,
    # -1's are considered as NaNs.
    tensor_list = [
        [torch.tensor([100, -1]),
         torch.tensor([100, -1, -1])],
        [torch.tensor([-1]), torch.tensor([-1, -1])],
    ]
    multi_nested_tensor_with_nan = MultiNestedTensor.from_tensor_mat(
        tensor_list)
    columnwise_nan_mask = dict()
    for col in range(multi_nested_tensor_with_nan.num_cols):
        columnwise_nan_mask[col] = (
            multi_nested_tensor_with_nan[:, col].values == -1)

    # Test fillna_col
    for col in range(multi_nested_tensor_with_nan.num_cols):
        multi_nested_tensor_with_nan.fillna_col(col, col)
    assert not torch.all(multi_nested_tensor_with_nan.values == -1).any()
    for col in range(multi_nested_tensor_with_nan.num_cols):
        column = multi_nested_tensor_with_nan[:, col]
        assert torch.all(column.values[columnwise_nan_mask[col]] == col)
        assert torch.all(column.values[~columnwise_nan_mask[col]] == 100)

    tensor_list = [
        [
            torch.tensor([100., torch.nan]),
            torch.tensor([torch.nan, 100., torch.nan])
        ],
        [torch.tensor([torch.nan]),
         torch.tensor([torch.nan, 100.])],
    ]
    multi_nested_tensor_with_nan = MultiNestedTensor.from_tensor_mat(
        tensor_list)
    columnwise_nan_mask = dict()
    for col in range(multi_nested_tensor_with_nan.num_cols):
        columnwise_nan_mask[col] = torch.isnan(
            multi_nested_tensor_with_nan[:, col].values)

    # Test fillna_col
    for col in range(multi_nested_tensor_with_nan.num_cols):
        multi_nested_tensor_with_nan.fillna_col(col, float(col))
    assert not torch.isnan(multi_nested_tensor_with_nan.values).any()
    for col in range(multi_nested_tensor_with_nan.num_cols):
        column = multi_nested_tensor_with_nan[:, col]
        assert torch.all(
            torch.isclose(column.values[columnwise_nan_mask[col]],
                          torch.tensor([col], dtype=torch.float32)))
        assert torch.all(
            torch.isclose(column.values[~columnwise_nan_mask[col]],
                          torch.tensor([100], dtype=torch.float32)))


@withCUDA
def test_multi_nested_tensor_basics(device):
    num_rows = 8
    num_cols = 10
    max_value = 100
    tensor_mat = []
    for _ in range(num_rows):
        tensor_list = []
        for _ in range(num_cols):
            length = random.randint(0, 10)
            tensor = torch.randint(
                0,
                max_value,
                size=(length, ),
                device=device,
            )
            tensor_list.append(tensor)
        tensor_mat.append(tensor_list)

    multi_nested_tensor = MultiNestedTensor.from_tensor_mat(tensor_mat)
    assert (str(multi_nested_tensor) ==
            f"MultiNestedTensor(num_rows=8, num_cols=10, device='{device}')")
    assert multi_nested_tensor.device == device
    assert multi_nested_tensor.values.device == device
    assert multi_nested_tensor.offset.device == device

    # Test sizes
    assert multi_nested_tensor.shape[0] == num_rows
    assert multi_nested_tensor.size(0) == num_rows
    assert multi_nested_tensor.shape[1] == num_cols
    assert multi_nested_tensor.size(1) == num_cols
    with pytest.raises(IndexError, match="not have a fixed length"):
        multi_nested_tensor.size(2)
    with pytest.raises(IndexError, match="Dimension out of range"):
        multi_nested_tensor.size(3)

    dense_multi_nested_tensor = multi_nested_tensor.to_dense(fill_value=-1)
    max_len = 0
    for i in range(multi_nested_tensor.num_rows):
        for j in range(multi_nested_tensor.num_cols):
            tensor = tensor_mat[i][j]
            if len(tensor) > max_len:
                max_len = len(tensor)
            assert (torch.allclose(
                dense_multi_nested_tensor[i, j][:len(tensor)], tensor))
            assert (dense_multi_nested_tensor[i, j][len(tensor):] == -1).all()
    assert dense_multi_nested_tensor.shape == (multi_nested_tensor.shape[:-1] +
                                               (max_len, ))

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

    # Test row range
    multi_nested_tensor_range = multi_nested_tensor[range(2, 6)]
    for idx, i in enumerate(range(2, 6)):
        for j in range(num_cols):
            assert torch.allclose(tensor_mat[i][j],
                                  multi_nested_tensor_range[idx, j])

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
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0], []]:
        multi_nested_tensor_indexed = multi_nested_tensor[index]
        assert multi_nested_tensor_indexed.shape[0] == len(index)
        assert multi_nested_tensor_indexed.shape[1] == num_cols
        for i, idx in enumerate(index):
            for j in range(num_cols):
                tensor = multi_nested_tensor_indexed[i, j]
                assert torch.allclose(tensor_mat[idx][j], tensor)

    # Test row-wise Boolean masking
    for index in [[4], [2, 3], [0, 1, 7], []]:
        mask = torch.zeros((num_rows, ), dtype=torch.bool, device=device)
        mask[index] = True
        multi_nested_tensor_indexed = multi_nested_tensor[mask]
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
        assert_equal(column_select(tensor_mat, [j]), multi_nested_tensor_col)

    # Test column List[int] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0], []]:
        assert_equal(column_select(tensor_mat, index),
                     multi_nested_tensor[:, index])

    # Test column-wise Boolean masking
    for index in [[4], [2, 3], [0, 1, 7], []]:
        mask = torch.zeros((num_cols,), dtype=torch.bool, device=device)
        mask[index] = True
        assert_equal(column_select(tensor_mat, index),
                     multi_nested_tensor[:, mask])

    # Test row range
    multi_nested_tensor_range = multi_nested_tensor[:, range(2, 6)]
    for i in range(num_rows):
        for idx, j in enumerate(range(2, 6)):
            assert torch.allclose(tensor_mat[i][j],
                                  multi_nested_tensor_range[i, idx])

    # Test column slicing
    assert_equal(tensor_mat, multi_nested_tensor[:, :])
    assert_equal(column_select(tensor_mat, slice(None, 3)),
                 multi_nested_tensor[:, :3])
    assert_equal(column_select(tensor_mat, slice(3, None)),
                 multi_nested_tensor[:, 3:])
    assert_equal(column_select(tensor_mat, slice(3, 5)),
                 multi_nested_tensor[:, 3:5])
    assert_equal(column_select(tensor_mat, slice(-7, 5)),
                 multi_nested_tensor[:, -7:5])
    assert_equal(column_select(tensor_mat, slice(-7, -1)),
                 multi_nested_tensor[:, -7:-1])
    assert_equal(column_select(tensor_mat, slice(1, None, 2)),
                 multi_nested_tensor[:, 1::2])
    empty_multi_nested_tensor = multi_nested_tensor[:, 5:3]
    assert empty_multi_nested_tensor.shape[0] == num_rows
    assert empty_multi_nested_tensor.shape[1] == 0

    # Test column narrow
    assert_equal(column_select(tensor_mat, slice(3, 3 + 2)),
                 multi_nested_tensor.narrow(dim=1, start=3, length=2))

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
    assert_equal(tensor_mat, MultiNestedTensor.cat([multi_nested_tensor],
                                                   dim=0))
    with pytest.raises(RuntimeError, match="num_cols must be the same"):
        MultiNestedTensor.cat([
            multi_nested_tensor[:2],
            multi_nested_tensor[2:4, 0],
        ], dim=0)
    with pytest.raises(RuntimeError, match="Cannot concatenate"):
        MultiNestedTensor.cat([], dim=0)

    # Testing col concat
    assert_equal(
        tensor_mat,
        MultiNestedTensor.cat(
            (multi_nested_tensor[:, :2], multi_nested_tensor[:, 2:4],
             multi_nested_tensor[:, 4:]), dim=1),
    )
    assert_equal(
        tensor_mat,
        MultiNestedTensor.cat([
            multi_nested_tensor[:, i]
            for i in range(multi_nested_tensor.size(1))
        ], dim=1),
    )
    assert_equal(tensor_mat, MultiNestedTensor.cat([multi_nested_tensor],
                                                   dim=1))
    with pytest.raises(RuntimeError, match="num_rows must be the same"):
        MultiNestedTensor.cat([
            multi_nested_tensor[1:],
            multi_nested_tensor,
        ], dim=1)
    with pytest.raises(RuntimeError, match="Cannot concatenate"):
        MultiNestedTensor.cat([], dim=1)

    # Testing set item
    with pytest.raises(RuntimeError, match="not currently supported"):
        multi_nested_tensor[0, 0] = torch.zeros(3)

    # Testing clone
    cloned_multi_nested_tensor = multi_nested_tensor.clone()
    assert MultiNestedTensor.allclose(multi_nested_tensor,
                                      cloned_multi_nested_tensor)
    cloned_multi_nested_tensor.values[0] = max_value + 1.0
    assert multi_nested_tensor.values[0] != max_value + 1.0
    cloned_multi_nested_tensor.values[0] = -1
    assert multi_nested_tensor.values[0] != -1
    assert not MultiNestedTensor.allclose(multi_nested_tensor,
                                          cloned_multi_nested_tensor)


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
