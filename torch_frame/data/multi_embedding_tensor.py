import copy
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor


class MultiEmbeddingTensor:
    r"""asdf

    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
        values (torch.Tensor): The values :class:`torch.Tensor` of size
            :obj:`[num_rows, dim0+dim1+...+dimN]`.
        offset (torch.Tensor): The offset :class:`torch.Tensor` of size
            :obj:`[num_cols+1,]`.
    """
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        values: Tensor,
        offset: Tensor,
    ) -> None:
        assert offset[0] == 0
        assert offset[-1] == len(values)
        assert len(offset) == num_rows * num_cols + 1
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.values = values
        self.offset = offset

    def __setitem__(self, index: Any, values: Any):
        raise RuntimeError(
            "MultiEmbeddingTensor object does not support setting values. "
            "It should be used for read-only. ")

    def __getitem__(
        self,
        index: Any,
    ) -> Union['MultiEmbeddingTensor', Tensor]:

        if isinstance(index, tuple):
            # index[0] for row indexing, index[1] for column indexing
            assert len(index) == 2
            if isinstance(index[0], int) and isinstance(index[1], int):
                # Returns Tensor
                return self.get_value(index[0], index[1])
            else:
                # Returns MultiEmbeddingTensor
                out = self
                for dim, idx in enumerate(index):
                    out = out.select(idx, dim)
                return out
        else:
            # Returns MultiEmbeddingTensor
            return self.select(index, dim=0)

    def get_value(self, i: int, j: int) -> Tensor:
        r"""Get :obj:`(i, j)`-th :class:`Tensor` object.

        Args:
            i (int): The row integer index.
            j (int): The column integer index.
        """
        i = self._to_positive_index(i, dim=0)
        j = self._to_positive_index(j, dim=1)
        idx = i * self.num_cols + j
        start_idx = self.offset[idx]
        end_idx = self.offset[idx + 1]
        out = self.values[start_idx:end_idx]
        return out

    def select(
        self,
        index: Union[int, Tensor, List, slice],
        dim: int,
    ) -> 'MultiEmbeddingTensor':
        r"""Supports all types of row/column-level advanced indexing.

        Args:
            index (Union[int, Tensor, List, slice]): Input :obj:`index`.
            dim (int): row (:obj:`dim = 0`) or column (:obj:`dim = 1`)
        """
        if isinstance(index, int):
            return self.single_index_select(index, dim=dim)
        elif isinstance(index, slice):
            return self.slice(index, dim=dim)
        elif isinstance(index, Tensor) and index.ndim == 1:
            return self.index_select(index, dim=dim)
        elif isinstance(index, List):
            return self.index_select(torch.tensor(index, device=self.device),
                                     dim=dim)
        else:
            raise NotImplementedError

    def narrow(self, dim: int, start: int, length: int) -> 'MultiEmbeddingTensor':
        assert start >= 0
        dim = MultiEmbeddingTensor._check_dim(dim)
        num_data = self.num_rows if dim == 0 else self.num_cols
        if start == 0 and start + length >= num_data:
            # Do nothing, just return the full data
            return self
        elif length > 0:
            if dim == 0:
                return self._row_narrow(start, length)
            else:
                return self._col_narrow(start, length)
        else:
            # Return empty MultiEmbeddingTensor if length is 0 or negative
            if dim == 0:
                num_rows = 0
                num_cols = self.num_cols
            else:
                num_rows = self.num_rows
                num_cols = 0
            return MultiEmbeddingTensor(
                num_rows=num_rows, num_cols=num_cols,
                values=torch.zeros(0, device=self.device,
                                   dtype=self.values.dtype),
                offset=torch.zeros(1, device=self.device, dtype=torch.long))

    def slice(self, slice: slice, dim: int) -> 'MultiEmbeddingTensor':
        dim = MultiEmbeddingTensor._check_dim(dim)

        num_data = self.num_rows if dim == 0 else self.num_cols
        if slice.step is not None and slice.step > 1:
            # If step is larger than 1, we reuse index_select along rows.
            idx = torch.tensor(range(num_data)[slice], device=self.device)
            return self.index_select(idx, dim=dim)
        else:
            start_idx: int = self._to_positive_index(slice.start or 0, dim=dim)
            end_idx: int = self._to_positive_index(slice.stop or num_data,
                                                   dim=dim, is_slice_end=True)
            return self.narrow(dim=dim, start=start_idx,
                               length=end_idx - start_idx)

    def _row_narrow(self, start: int, length: int) -> 'MultiEmbeddingTensor':
        r"""Helper function called by :obj:`narrow`."""
        assert start >= 0
        assert length > 0
        end = start + length
        assert not (start == 0 and end >= self.num_rows)
        offset = self.offset[start * self.num_cols:end * self.num_cols + 1]
        values = self.values[offset[0]:offset[-1]]
        offset = offset - offset[0]
        return MultiEmbeddingTensor(num_rows=end - start, num_cols=self.num_cols,
                                 values=values, offset=offset)

    def _col_narrow(self, start: int, length: int) -> 'MultiEmbeddingTensor':
        r"""Helper function called by :obj:`narrow`."""
        assert start >= 0
        assert length > 0
        end = start + length
        if start == 0:
            assert end < self.num_cols
            offset_mat = (self.offset[:-1].reshape(
                self.num_rows, self.num_cols)[:, start:end + 1])
        else:
            offset_mat = (self.offset[1:].reshape(
                self.num_rows, self.num_cols)[:, start - 1:end])

        offset_start = offset_mat[:, 0]
        count = offset_mat[:, -1] - offset_start
        batch, arange = batched_arange(count)
        values = self.values[offset_start[batch] + arange]

        offset_mat_zero_start = offset_mat - offset_start.view(-1, 1)
        accum = torch.cumsum(offset_mat_zero_start[:, -1], dim=0)
        offset_mat_zero_start[1:] += accum[:-1].view(-1, 1)
        num_cols = end - start
        offset = torch.full((self.num_rows * num_cols + 1, ), accum[-1])
        offset[:-1] = offset_mat_zero_start[:, :-1].flatten()
        return MultiEmbeddingTensor(num_rows=self.num_rows, num_cols=num_cols,
                                 values=values, offset=offset)

    def index_select(self, index: Tensor, dim: int) -> 'MultiEmbeddingTensor':
        dim = MultiEmbeddingTensor._check_dim(dim)
        index = self._to_positive_index(index, dim=dim)
        if dim == 0:
            return self._row_index_select(index)
        else:
            return self._col_index_select(index)

    def _row_index_select(self, index: Tensor) -> 'MultiEmbeddingTensor':
        r"""Helper function called by :obj:`index_select`."""
        # Calculate values
        if index.numel() == 0:
            return MultiEmbeddingTensor(
                num_rows=0, num_cols=self.num_cols,
                values=torch.tensor([], device=self.device),
                offset=torch.tensor([0], device=self.device))
        index_right = (index + 1) * self.num_cols
        index_left = index * self.num_cols
        diff = self.offset[index_right] - self.offset[index_left]
        batch, arange = batched_arange(diff)
        idx = self.offset[index_left][batch] + arange
        values = self.values[idx]

        # Calculate offset
        count = torch.full(size=(len(index), ), fill_value=self.num_cols,
                           dtype=torch.long, device=self.device)
        count[-1] += 1
        batch, arange = batched_arange(count)
        idx = index_left[batch] + arange
        offset = self.offset[idx] - self.offset[index_left][batch]
        diff_cumsum = torch.cumsum(diff, dim=0)
        diff_cumsum = torch.roll(diff_cumsum, 1)
        diff_cumsum[0] = 0
        offset = offset + diff_cumsum[batch]
        return MultiEmbeddingTensor(num_rows=len(index), num_cols=self.num_cols,
                                 values=values, offset=offset)

    def _to_positive_index(
        self,
        index: Union[int, Tensor],
        dim: int,
        is_slice_end: bool = False,
    ):
        """Helper function to map negative indices to positive indices and
        raise :obj:`IndexError` when necessary.

        Args:
            index: Union[int, Tensor]: Input :obj:`index` with potentially
                negative elements.
            is_slice_end (bool): Whether a given index (int) is slice or not.
                If :obj:`True`, we have more lenient :obj:`IndexError`.
                (default: :obj:`False`)
        """
        assert dim in [0, 1]
        max_entries = self.num_rows if dim == 0 else self.num_cols
        idx_name = "Row" if dim == 0 else "Col"
        if isinstance(index, int):
            if index < 0:
                index = index + max_entries
            if is_slice_end and index < 0 or index > max_entries:
                raise IndexError(f"{idx_name} index out of bounds!")
            elif (not is_slice_end) and (index < 0 or index >= max_entries):
                raise IndexError(f"{idx_name} index out of bounds!")
        elif isinstance(index, Tensor):
            assert not is_slice_end
            assert index.ndim == 1
            neg_idx = index < 0
            if neg_idx.any():
                index = index.clone()
                index[neg_idx] = max_entries + index[neg_idx]
            if index.numel() != 0 and (index.min() < 0
                                       or index.max() >= max_entries):
                raise IndexError(f"{idx_name} index out of bounds!")
        return index

    @classmethod
    def _check_dim(self, dim: int) -> int:
        r"""Check :obj:`dim` argument and make it 0 or 1."""
        if dim < 0:
            dim = dim + 3
        if dim not in [0, 1]:
            raise ValueError(
                f"Advanced indexing with dim={dim} is unsupported in "
                f"MultiEmbeddingTensor. Please use dim=0 or 1.")
        return dim

    @property
    def device(self) -> torch.device:
        return self.values.device

    def __len__(self):
        return self.num_rows

    def clone(self) -> 'MultiEmbeddingTensor':
        return MultiEmbeddingTensor(self.num_rows, self.num_cols,
                                 self.values.clone(), self.offset.clone())

    def __repr__(self) -> str:
        return ' '.join([
            f"{self.__class__.__name__}(num_rows={self.num_rows},",
            f"num_cols={self.num_cols},",
            f"device='{self.device}')",
        ])

    # Device Transfer #########################################################

    def to(self, *args, **kwargs):
        return self._apply(lambda x: x.to(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return self._apply(lambda x: x.cpu(*args, **kwargs))

    def cuda(self, *args, **kwargs):
        return self._apply(lambda x: x.cuda(*args, **kwargs))

    # Helper Functions ########################################################

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> 'MultiEmbeddingTensor':
        out = copy.copy(self)
        out.values = fn(out.values)
        out.offset = fn(out.offset)

        return out

    # Properties ##############################################################

    @property
    def ndim(self) -> int:
        return 3

    def dim(self) -> int:
        return self.ndim

    def size(self, dim: int) -> int:
        r"""Dimension of the :class:`torch_frame.data.MultiEmbeddingTensor`"""
        # Handle negative dim
        if dim < 0:
            dim = self.ndim - dim

        if dim == 0:
            return self.num_rows
        elif dim == 1:
            return self.num_cols
        elif dim == 2:
            raise ValueError(
                "MultiEmbeddingTensor does not have a fixed length on the third"
                " dimension.")
        else:
            raise IndexError(
                "Dimension out of range (expected to be in range of [0, 2],"
                f" but got {dim}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.num_rows, self.num_cols, -1)

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    # Static methods ##########################################################
    @staticmethod
    def cat(xs: Union[Tuple['MultiEmbeddingTensor', ...],
                      List['MultiEmbeddingTensor']],
            dim: int = 0) -> 'MultiEmbeddingTensor':
        pass