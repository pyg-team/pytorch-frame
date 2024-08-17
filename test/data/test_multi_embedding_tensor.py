import random
from typing import List, Optional, Tuple, Union

import pytest
import torch

from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.testing import withCUDA


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


def row_select(
    tensor_list: List[torch.Tensor],
    index: Union[List[int], slice],
) -> List[torch.Tensor]:
    """Selects rows from a list of column tensors.

    Args:
        tensor_list (list[torch.Tensor]): A list of tensors of size
            [num_rows, dim_emb_j].
        index (Union[list[int], slice]): A list of row indices or a slice to
            apply to each tensor in tensor_list.

    Returns:
        List[torch.Tensor]: A list of tensors of size
            [num_rows_indexed, dim_emb_j].
    """
    return [col_tensor[index] for col_tensor in tensor_list]


def get_fake_multi_embedding_tensor(
    num_rows: int,
    num_cols: int,
    embedding_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MultiEmbeddingTensor, List[torch.Tensor]]:
    tensor_list = []
    for _ in range(num_cols):
        embedding_dim = embedding_dim or random.randint(1, 5)
        tensor = torch.randn((num_rows, embedding_dim), device=device)
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


def test_fillna_col():
    # Creat a MultiEmbeddingTensor containing all -1's
    # In MultiEmbeddingTensor with torch.long dtype,
    # -1's are considered as NaNs.
    tensor_list = [
        torch.tensor([[-1, 100, -1], [-1, 100, -1]]),
        torch.tensor([[-1, 100], [100, -1]]),
    ]
    met_with_nan = MultiEmbeddingTensor.from_tensor_list(tensor_list)
    columnwise_nan_mask = dict()
    for col in range(met_with_nan.num_cols):
        columnwise_nan_mask[col] = (met_with_nan[:, col].values == -1)

    # Test fillna_col
    for col in range(met_with_nan.num_cols):
        met_with_nan.fillna_col(col, col)
    assert not torch.all(met_with_nan.values == -1).any()
    for col in range(met_with_nan.num_cols):
        column = met_with_nan[:, col]
        assert torch.all(column.values[columnwise_nan_mask[col]] == col)
        assert torch.all(column.values[~columnwise_nan_mask[col]] == 100)

    tensor_list = [
        torch.tensor([[100., torch.nan, torch.nan],
                      [torch.nan, 100., torch.nan]]),
        torch.tensor([[torch.nan, 100.], [torch.nan, 100.]]),
    ]
    met_with_nan = MultiEmbeddingTensor.from_tensor_list(tensor_list)
    columnwise_nan_mask = dict()
    for col in range(met_with_nan.num_cols):
        columnwise_nan_mask[col] = torch.isnan(met_with_nan[:, col].values)

    # Test fillna_col
    for col in range(met_with_nan.num_cols):
        met_with_nan.fillna_col(col, col)
    assert not torch.isnan(met_with_nan.values).any()
    for col in range(met_with_nan.num_cols):
        column = met_with_nan[:, col]
        assert torch.all(
            torch.isclose(column.values[columnwise_nan_mask[col]],
                          torch.tensor([col], dtype=torch.float32)))
        assert torch.all(
            torch.isclose(column.values[~columnwise_nan_mask[col]],
                          torch.tensor([100], dtype=torch.float32)))


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


@withCUDA
def test_row_index(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )

    # Test [i, j] indexing
    for i in range(-num_rows, num_rows):
        for j in range(num_cols):
            tensor = met[i, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_list[j][i], tensor)

    # Test [i] indexing
    for i in range(-num_rows, num_rows):
        met_row = met[i]
        assert isinstance(met_row, MultiEmbeddingTensor)
        assert met_row.shape[0] == 1
        assert met_row.shape[1] == num_cols
        for j in range(-num_cols, num_cols):
            tensor = met_row[0, j]
            assert isinstance(tensor, torch.Tensor)
            assert torch.allclose(tensor_list[j][i], tensor)

    # Test [list[int]] indexing
    # Test [Tensor] indexing
    for index in [[4], [2, 2], [-4, 1, 7], [3, -7, 1, 0], []]:
        for index_type in ["list", "tensor"]:
            if index_type == "tensor":
                index = torch.tensor(index, dtype=torch.long)
            met_indexed = met[index]
            assert isinstance(met_indexed, MultiEmbeddingTensor)
            assert met_indexed.shape[0] == len(index)
            assert met_indexed.shape[1] == num_cols
            for i, idx in enumerate(index):
                for j in range(num_cols):
                    assert torch.allclose(
                        tensor_list[j][idx],
                        met_indexed[i, j],
                    )

    # test selection with Boolean mask
    # only ordered selection without duplicates is possible
    for index in [[4], [2, 3], [0, 1, 7], []]:
        mask = torch.zeros((num_rows, ), dtype=torch.bool, device=device)
        mask[index] = True

        met_indexed = met[mask]
        assert isinstance(met_indexed, MultiEmbeddingTensor)
        assert met_indexed.shape[0] == len(index)
        assert met_indexed.shape[1] == num_cols
        for i, idx in enumerate(index):
            for j in range(num_cols):
                assert torch.allclose(
                    tensor_list[j][idx],
                    met_indexed[i, j],
                )


@withCUDA
def test_row_index_range(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    # Test [range] indexing
    for index in [range(2, 6), range(2, 6, 2), range(6, 2, -1)]:
        met_indexed = met[index]
        assert isinstance(met_indexed, MultiEmbeddingTensor)
        assert met_indexed.shape[0] == len(index)
        assert met_indexed.shape[1] == num_cols
        for idx, i in enumerate(index):
            for j in range(num_cols):
                assert torch.allclose(tensor_list[j][i], met_indexed[idx, j])


@withCUDA
def test_row_index_slice(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    # Test [slice] indexing
    assert_equal(tensor_list, met[:])
    assert_equal(row_select(tensor_list, slice(None, 3)), met[:3])
    assert_equal(row_select(tensor_list, slice(3, None)), met[3:])
    assert_equal(row_select(tensor_list, slice(3, 5)), met[3:5])
    assert_equal(row_select(tensor_list, slice(-7, 5)), met[-7:5])
    assert_equal(row_select(tensor_list, slice(-7, -1)), met[-7:-1])
    assert_equal(row_select(tensor_list, slice(1, None, 2)), met[1::2])
    empty_met = met[5:3]
    assert empty_met.shape[0] == 0
    assert empty_met.shape[1] == num_cols


@withCUDA
def test_col_index_int(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    # Test [:, int] indexing
    for index in [4, 2, -4, 1, 7, 3, -7, 1, 0]:
        met_indexed = met[:, index]
        assert isinstance(met_indexed, MultiEmbeddingTensor)
        assert met_indexed.shape[0] == num_rows
        assert met_indexed.shape[1] == 1
        for i in range(num_rows):
            assert torch.allclose(
                tensor_list[index][i],
                met_indexed[i, 0],
            )


@withCUDA
def test_col_index_slice(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    # Test [:, slice] indexing
    assert_equal(tensor_list, met[:, :])
    assert_equal(tensor_list[:3], met[:, :3])
    assert_equal(tensor_list[3:], met[:, 3:])
    assert_equal(tensor_list[3:5], met[:, 3:5])
    assert_equal(tensor_list[-7:5], met[:, -7:5])
    assert_equal(tensor_list[-7:-1], met[:, -7:-1])
    assert_equal(tensor_list[1::2], met[:, 1::2])
    empty_met = met[:, 5:3]
    assert empty_met.shape[0] == num_rows
    assert empty_met.shape[1] == 0


@withCUDA
def test_col_index_list(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    # Test [:, list] indexing
    # Test [:, Tensor] indexing
    for index in [[4], [2, 2], [-4, 1, 9], [3, -9, 1, 0], []]:
        for index_type in ["list", "tensor"]:
            if index_type == "tensor":
                index = torch.tensor(index, dtype=torch.long)
            met_indexed = met[:, index]
            assert isinstance(met_indexed, MultiEmbeddingTensor)
            assert met_indexed.shape[0] == num_rows
            assert met_indexed.shape[1] == len(index)
            for i in range(num_rows):
                for j, idx in enumerate(index):
                    assert torch.allclose(
                        tensor_list[idx][i],
                        met_indexed[i, j],
                    )

    # test selection with Boolean mask
    # only ordered selection without duplicates is possible
    for index in [[4], [2, 3], [0, 1, 7], []]:
        mask = torch.zeros((num_cols, ), dtype=torch.bool, device=device)
        mask[index] = True
        met_indexed = met[:, mask]
        assert isinstance(met_indexed, MultiEmbeddingTensor)
        assert met_indexed.shape[0] == num_rows
        assert met_indexed.shape[1] == len(index)
        for i in range(num_rows):
            for j, idx in enumerate(index):
                assert torch.allclose(
                    tensor_list[idx][i],
                    met_indexed[i, j],
                )


@withCUDA
def test_col_index_range(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    # Test [:, range] indexing
    for index in [range(2, 6), range(2, 6, 2), range(6, 2, -1)]:
        met_indexed = met[:, index]
        assert isinstance(met_indexed, MultiEmbeddingTensor)
        assert met_indexed.shape[0] == num_rows
        assert met_indexed.shape[1] == len(index)
        for j, idx in enumerate(index):
            for i in range(num_rows):
                assert torch.allclose(
                    tensor_list[idx][i],
                    met_indexed[i, j],
                )


@withCUDA
def test_col_index_empty(device):
    met, _ = get_fake_multi_embedding_tensor(
        num_rows=0,
        num_cols=10,
        embedding_dim=3,
        device=device,
    )
    assert met[:, 1].shape == (0, 1, -1)
    assert met[:, [1, 2]].shape == (0, 2, -1)


@withCUDA
def test_narrow(device):
    num_rows = 8
    num_cols = 10
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=num_rows,
        num_cols=num_cols,
        device=device,
    )
    assert_equal(tensor_list[2:2 + 4], met.narrow(1, 2, 4))
    assert_equal(row_select(tensor_list, slice(2, 2 + 4)), met.narrow(0, 2, 4))
    empty_met = met.narrow(0, 2, 0)
    assert empty_met.shape[0] == 0
    assert empty_met.shape[1] == num_cols
    empty_met = met.narrow(1, 2, 0)
    assert empty_met.shape[0] == num_rows
    assert empty_met.shape[1] == 0


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


@withCUDA
def test_cat(device):
    met, tensor_list = get_fake_multi_embedding_tensor(
        num_rows=8,
        num_cols=10,
        embedding_dim=3,
        device=device,
    )
    # case: dim=0
    assert_equal(
        tensor_list,
        MultiEmbeddingTensor.cat([met[:2], met[2:4], met[4:]], dim=0),
    )
    assert_equal(
        tensor_list,
        MultiEmbeddingTensor.cat([met[i] for i in range(met.size(0))], dim=0),
    )
    assert_equal(tensor_list, MultiEmbeddingTensor.cat([met], dim=0))
    with pytest.raises(RuntimeError, match="num_cols must be the same"):
        MultiEmbeddingTensor.cat([met[:2], met[2:4, 0]], dim=0)
    with pytest.raises(RuntimeError, match="Cannot concatenate"):
        MultiEmbeddingTensor.cat([], dim=0)

    # case: dim=1
    assert_equal(
        tensor_list,
        MultiEmbeddingTensor.cat([met[:, :2], met[:, 2:4], met[:, 4:]], dim=1),
    )
    assert_equal(
        tensor_list,
        MultiEmbeddingTensor.cat(
            [met[:, i] for i in range(met.size(1))],
            dim=1,
        ),
    )
    assert_equal(tensor_list, MultiEmbeddingTensor.cat([met], dim=1))
    with pytest.raises(RuntimeError, match="num_rows must be the same"):
        MultiEmbeddingTensor.cat([met[1:], met], dim=1)
    with pytest.raises(RuntimeError, match="Cannot concatenate"):
        MultiEmbeddingTensor.cat([], dim=1)

    # case: different devices should raise error
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.cat([met.to("cpu"), met.to("meta")], dim=0)

    # case: unsupported dim should raise error
    with pytest.raises(IndexError, match="Dimension out of range"):
        MultiEmbeddingTensor.cat([met], dim=3)

    # case: list of non-MultiEmbeddingTensor should raise error
    with pytest.raises(AssertionError):
        MultiEmbeddingTensor.cat([object()], dim=0)


def test_pin_memory():
    met, _ = get_fake_multi_embedding_tensor(
        num_rows=2,
        num_cols=3,
    )
    assert not met.is_pinned()
    assert not met.values.is_pinned()
    assert not met.offset.is_pinned()
    met = met.pin_memory()
    assert met.is_pinned()
    assert met.values.is_pinned()
    assert met.offset.is_pinned()
