import copy
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor


class MultiNestedTensor:
    r"""A PyTorch tensor-based data structure that stores
    :obj:`[num_rows, num_cols, *]`, where the size of last dimension can be
    different for different data/column. Internally, we store the object in an
    efficient flattened format: :obj:`(values, offset)`, where the PyTorch
    Tensor at :obj:`(i, j)` is accessed by
    :obj:`values[offset[i*num_cols+j]:offset[i*num_cols+j+1]]`

    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
        values (torch.Tensor): The values Tensor that has
        offset (torch.Tensor): The offset Tensor.

    Example:
        >>> import torch
        >>> tensor_mat = [
        ...    [torch.tensor([1, 2]), torch.tensor([3])],
        ...    [torch.tensor([4]), torch.tensor([5, 6, 7])],
        ...    [torch.tensor([8, 9]), torch.tensor([10])],
        ... ]
        >>> out = MultiNestedTensor.from_tensor_mat(tensor_mat)
        >>> out.size(0)
        3
        >>> out.size(1)
        2
        >>> out.size(2)
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        ValueError: MultiNestedTensor does not have a fixed length on the third dimension.  # noqa
    """
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        values: Tensor,
        offset: Tensor,
    ):
        assert offset[0] == 0
        assert offset[-1] == len(values)
        assert len(offset) == num_rows * num_cols + 1
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.values = values
        self.offset = offset

    @classmethod
    def from_tensor_mat(
        cls,
        tensor_mat: List[List[Tensor]],
    ) -> 'MultiNestedTensor':
        r"""Construct :class:`MultiNestedTensor` object from
        :obj:`tensor_mat`.

        Args:
            tensor_mat (List[List[Tensor]]): A matrix of
                :class:`torch.Tensor` objects. :obj:`tensor_mat[i][j]`
                contains 1-dim :class:`torch.Tensor` of :obj:`i`-th row
                and :obj:`j`-th column, varying in size.

        Returns:
            MultiNestedTensor: A :class:`MultiNestedTensor` instance.

        Example:
            >>> tensor_mat = [
            ...    [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            ...    [torch.tensor([6, 7]), torch.tensor([8, 9, 10])],
            ... ]
            >>> out = MultiNestedTensor.from_tensor_mat(tensor_mat)
            >>> out
            MultiNestedTensor(num_rows=2, num_cols=2, device='cpu')
            >>> out.values
            tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
            >>> out.offset
            tensor([ 0,  3,  5,  7, 10])
            >>> tensor_mat[2][0]
            tensor([8, 9])
            >>> out[2, 0]
            tensor([8, 9])
        """
        num_rows = len(tensor_mat)
        num_cols = len(tensor_mat[0])

        offset_list = []
        accum_idx = 0
        offset_list.append(accum_idx)
        values_list = []
        for i in range(num_rows):
            if len(tensor_mat[i]) != num_cols:
                raise RuntimeError(
                    f"The length of each row must be the same."
                    f" tensor_mat[0] has length {num_cols}, but"
                    f" tensor_mat[{i}] has length {len(tensor_mat[i])}")

            for j in range(num_cols):
                tensor = tensor_mat[i][j]
                if not isinstance(tensor, Tensor):
                    raise RuntimeError(
                        "The element of tensor_mat must be PyTorch Tensor")
                if tensor.ndim != 1:
                    raise RuntimeError(
                        "tensor in tensor_mat needs to be 1-dimensional.")
                values_list.append(tensor)
                accum_idx += len(tensor)
                offset_list.append(accum_idx)

        values = torch.cat(values_list)
        offset = torch.LongTensor(offset_list)

        return cls(num_rows, num_cols, values, offset)

    def __setitem__(self, index: Any, values: Any):
        raise RuntimeError(
            "MultiNestedTensor object does not support setting values. "
            "It should be used for read-only. ")

    def __getitem__(
        self,
        index: Any,
    ) -> Union['MultiNestedTensor', Tensor]:

        if isinstance(index, tuple):
            # index[0] for row indexing, index[1] for column indexing
            assert len(index) == 2
            if isinstance(index[0], int) and isinstance(index[1], int):
                # Returns Tensor
                return self.get_value(index[0], index[1])
            else:
                # Returns MultiNestedTensor
                out = self
                for dim, idx in enumerate(index):
                    out = out.select(idx, dim)
                return out
        else:
            # Returns MultiNestedTensor
            return self.select(index, dim=0)

    def narrow(self, dim: int, start: int, length: int) -> 'MultiNestedTensor':
        assert start >= 0
        dim = self._check_dim(dim)
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
            # Return empty MultiNestedTensor if length is 0 or negative
            if dim == 0:
                num_rows = 0
                num_cols = self.num_cols
            else:
                num_rows = self.num_rows
                num_cols = 0
            return MultiNestedTensor(
                num_rows=num_rows, num_cols=num_cols,
                values=torch.zeros(0, device=self.device,
                                   dtype=self.values.dtype),
                offset=torch.zeros(1, device=self.device, dtype=torch.long))

    def slice(self, slice: slice, dim: int) -> 'MultiNestedTensor':
        dim = self._check_dim(dim)

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

    def _row_narrow(self, start: int, length: int) -> 'MultiNestedTensor':
        r"""Helper function called by :obj:`narrow`."""
        assert start >= 0
        assert length > 0
        end = start + length
        assert not (start == 0 and end >= self.num_rows)
        offset = self.offset[start * self.num_cols:end * self.num_cols + 1]
        values = self.values[offset[0]:offset[-1]]
        offset = offset - offset[0]
        return MultiNestedTensor(num_rows=end - start, num_cols=self.num_cols,
                                 values=values, offset=offset)

    def _col_narrow(self, start: int, length: int) -> 'MultiNestedTensor':
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
        return MultiNestedTensor(num_rows=self.num_rows, num_cols=num_cols,
                                 values=values, offset=offset)

    def index_select(self, index: Tensor, dim: int) -> 'MultiNestedTensor':
        dim = self._check_dim(dim)
        index = self._to_positive_index(index, dim=dim)
        if dim == 0:
            return self._row_index_select(index)
        else:
            return self._col_index_select(index)

    def _row_index_select(self, index: Tensor) -> 'MultiNestedTensor':
        r"""Helper function called by :obj:`index_select`."""
        # Calculate values
        if index.numel() == 0:
            return MultiNestedTensor(
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
        return MultiNestedTensor(num_rows=len(index), num_cols=self.num_cols,
                                 values=values, offset=offset)

    def _col_index_select(self, index: Tensor) -> 'MultiNestedTensor':
        r"""Helper function called by :obj:`index_select`."""
        if index.numel() == 0:
            return MultiNestedTensor(
                num_rows=self.num_rows, num_cols=0,
                values=torch.tensor([], device=self.device),
                offset=torch.tensor([0], device=self.device))
        start_idx = (
            index +
            torch.arange(0, self.num_rows * self.num_cols, self.num_cols,
                         device=self.device).view(-1, 1)).flatten()
        offset_start = self.offset[start_idx]
        count = self.offset[start_idx + 1] - self.offset[start_idx]
        offset = count.new_zeros(count.numel() + 1)
        torch.cumsum(count, dim=0, out=offset[1:])
        batch, arange = batched_arange(count)
        values = self.values[offset_start[batch] + arange]
        return MultiNestedTensor(num_rows=self.num_rows, num_cols=len(index),
                                 values=values, offset=offset)

    def single_index_select(self, index: int, dim: int) -> 'MultiNestedTensor':
        r"""Get :obj:`index`-th row (:obj:`dim=0`) or column (:obj:`dim=1`)"""
        dim = self._check_dim(dim)
        index = self._to_positive_index(index, dim=dim)
        if dim == 0:
            start_idx = index * self.num_cols
            end_idx = (index + 1) * self.num_cols + 1
            offset = self.offset[start_idx:end_idx]
            values = self.values[offset[0]:offset[-1]]
            offset = offset - offset[0]
            return MultiNestedTensor(num_rows=1, num_cols=self.num_cols,
                                     values=values, offset=offset)
        elif dim == 1:
            start_idx = torch.arange(index, self.num_rows * self.num_cols,
                                     self.num_cols, device=self.device)
            diff = self.offset[start_idx + 1] - self.offset[start_idx]
            batch, arange = batched_arange(diff)
            # Compute values
            values = self.values[self.offset[start_idx][batch] + arange]
            # Compute offset
            offset = diff.new_zeros(diff.numel() + 1)
            torch.cumsum(diff, dim=0, out=offset[1:])
            return MultiNestedTensor(num_rows=self.num_rows, num_cols=1,
                                     values=values, offset=offset)
        else:
            raise RuntimeError(f"Unsupported dim={dim} for index_select.")

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

    def _check_dim(self, dim: int) -> int:
        r"""Check :obj:`dim` argument and make it 0 or 1."""
        if dim < 0:
            dim = dim + self.ndim
        if dim not in [0, 1]:
            raise ValueError(
                f"Advanced indexing with dim={dim} is unsupported in "
                f"MultiNestedTensor. Please use dim=0 or 1.")
        return dim

    @property
    def device(self) -> torch.device:
        return self.values.device

    def __len__(self):
        return self.num_rows

    # Device Transfer #########################################################

    def to(self, *args, **kwargs):
        return self._apply(lambda x: x.to(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return self._apply(lambda x: x.cpu(*args, **kwargs))

    def cuda(self, *args, **kwargs):
        return self._apply(lambda x: x.cuda(*args, **kwargs))

    # Helper Functions ########################################################

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> 'MultiNestedTensor':
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
        r"""Dimension of the :class:`torch_frame.data.MultiNestedTensor`"""
        # Handle negative dim
        if dim < 0:
            dim = self.ndim - dim

        if dim == 0:
            return self.num_rows
        elif dim == 1:
            return self.num_cols
        elif dim == 2:
            raise ValueError(
                "MultiNestedTensor does not have a fixed length on the third"
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
    def cat(xs: Union[Tuple['MultiNestedTensor', ...],
                      List['MultiNestedTensor']],
            dim: int = 0) -> 'MultiNestedTensor':
        if len(xs) == 0:
            raise RuntimeError("Cannot concatenate a list of length 0.")
        assert isinstance(xs[0], MultiNestedTensor)
        if dim == 0 or dim + xs[0].ndim == 0:
            num_rows = sum(x.num_rows for x in xs)
            num_cols = xs[0].num_cols
            for x in xs[1:]:
                if x.num_cols != num_cols:
                    raise RuntimeError(
                        "num_cols must be the same across a list of input "
                        "multi nested tensors.")
            values = torch.cat([x.values for x in xs], dim=0)

            offset = torch.empty(num_rows * num_cols + 1, dtype=torch.long,
                                 device=values.device)
            accum = 0
            idx = 0
            for x in xs[:-1]:
                offset[idx:idx + len(x.offset[:-1])] = x.offset[:-1]
                offset[idx:idx + len(x.offset[:-1])].add_(accum)
                accum += x.offset[-1]
                idx += len(x.offset[:-1])
            offset[idx:] = xs[-1].offset
            offset[idx:].add_(accum)
            return MultiNestedTensor(num_rows=num_rows, num_cols=num_cols,
                                     values=values, offset=offset)
        elif dim == 1 or dim + xs[0].ndim == 1:
            if len(xs) == 1:
                return xs[0]
            else:
                # TODO Weihua implement this
                raise NotImplementedError
        else:
            raise RuntimeError(f"Unsupported dim={dim} for concat.")

    def select(
        self,
        index: Union[int, Tensor, List, slice],
        dim: int,
    ) -> 'MultiNestedTensor':
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

    def clone(self) -> 'MultiNestedTensor':
        return MultiNestedTensor(self.num_rows, self.num_cols,
                                 self.values.clone(), self.offset.clone())

    def __repr__(self) -> str:
        return ' '.join([
            f"{self.__class__.__name__}(num_rows={self.num_rows},",
            f"num_cols={self.num_cols},",
            f"device='{self.device}')",
        ])


def batched_arange(count: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Fast implementation of batched version of :meth:`torch.arange`.
    It essentially does the following
    >>> batch = torch.cat([torch.full((c,), i) for i, c in enumerate(count)])
    >>> arange = torch.cat([torch.arange(c) for c in count])

    Args:
        counts (Tensor): The count vectors.

    Returns:
        batch (Tensor): batch[i] indicates the batch index of
            batched_arange[i]
        arange (Tensor): batched version of arange
    """
    ptr = count.new_zeros(count.numel() + 1)
    torch.cumsum(count, dim=0, out=ptr[1:])

    batch = torch.arange(count.numel(), device=count.device).repeat_interleave(
        count, output_size=ptr[-1])  # type: ignore

    arange = torch.arange(batch.numel(), device=count.device)
    arange -= ptr[batch]

    return batch, arange
