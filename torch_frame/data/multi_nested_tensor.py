from __future__ import annotations

from typing import Sequence, cast

import torch
from torch import Tensor

from torch_frame.data.multi_tensor import _batched_arange, _MultiTensor


class MultiNestedTensor(_MultiTensor):
    r"""A read-only PyTorch tensor-based data structure that stores
    :obj:`[num_rows, num_cols, *]`, where the size of last dimension can be
    different for different row/column. Internally, we store the object in an
    efficient flattened format: :obj:`(values, offset)`, where the PyTorch
    Tensor at :obj:`(i, j)` is accessed by
    :obj:`values[offset[i*num_cols+j]:offset[i*num_cols+j+1]]`.
    It supports various advanced indexing, including slicing and list indexing
    along both row and column.

    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
        values (torch.Tensor): The values :class:`torch.Tensor` of size
            :obj:`[numel,]`.
        offset (torch.Tensor): The offset :class:`torch.Tensor` of size
            :obj:`[num_rows*num_cols+1,]`.

    Example:
        >>> import torch
        >>> tensor_mat = [
        ...    [torch.tensor([1, 2]), torch.tensor([3])],
        ...    [torch.tensor([4]), torch.tensor([5, 6, 7])],
        ...    [torch.tensor([8, 9]), torch.tensor([10])],
        ... ]
        >>> mnt = MultiNestedTensor.from_tensor_mat(tensor_mat)
        >>> mnt
        MultiNestedTensor(num_rows=3, num_cols=2, device='cpu')
        >>> mnt.values
        tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
        >>> mnt.offset
        tensor([ 0,  2,  3,  4,  7,  9, 10])
        >>> mnt[0, 0]
        torch.tensor([1, 2])
        >>> mnt[1, 1]
        tensor([5, 6, 7])
        >>> mnt[0] # Row integer indexing
        MultiNestedTensor(num_rows=1, num_cols=2, device='cpu')
        >>> mnt[:, 0] # Column integer indexing
        MultiNestedTensor(num_rows=3, num_cols=1, device='cpu')
        >>> mnt[:2] # Row integer slicing
        MultiNestedTensor(num_rows=2, num_cols=2, device='cpu')
        >>> mnt[[2, 1, 2, 0]] # Row list indexing
        MultiNestedTensor(num_rows=4, num_cols=2, device='cpu')
        >>> mnt.to_dense(fill_value = -1) # Map to a dense matrix via padding
        tensor([[[ 1,  2, -1],
                [ 3, -1, -1]],
                [[ 4, -1, -1],
                [ 5,  6,  7]],
                [[ 8,  9, -1],
                [10, -1, -1]]])
    """
    def validate(self):
        assert self.offset[0] == 0
        assert self.offset[-1] == len(self.values)
        assert len(self.offset) == self.num_rows * self.num_cols + 1

    @classmethod
    def from_tensor_mat(
        cls,
        tensor_mat: list[list[Tensor]],
    ) -> MultiNestedTensor:
        r"""Construct :class:`MultiNestedTensor` object from
        :obj:`tensor_mat`.

        Args:
            tensor_mat (List[List[Tensor]]): A matrix of
                :class:`torch.Tensor` objects. :obj:`tensor_mat[i][j]`
                contains 1-dim :class:`torch.Tensor` of :obj:`i`-th row
                and :obj:`j`-th column, varying in size.

        Returns:
            MultiNestedTensor: A :class:`MultiNestedTensor` instance.
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
        offset = torch.tensor(offset_list, device=values.device)

        return cls(num_rows, num_cols, values, offset)

    def _get_value(self, i: int, j: int) -> Tensor:
        r"""Get :obj:`(i, j)`-th :class:`Tensor` object.

        Args:
            i (int): The row integer index.
            j (int): The column integer index.
        """
        i = self._normalize_index(i, dim=0)
        j = self._normalize_index(j, dim=1)
        idx = i * self.num_cols + j
        start_idx = self.offset[idx]
        end_idx = self.offset[idx + 1]
        out = self.values[start_idx:end_idx]
        return out

    def _row_narrow(self, start: int, length: int) -> MultiNestedTensor:
        r"""Helper function called by :meth:`MultiNestedTensor.narrow`."""
        assert start >= 0
        assert length > 0
        end = start + length
        assert not (start == 0 and end >= self.num_rows)
        offset = self.offset[start * self.num_cols:end * self.num_cols + 1]
        values = self.values[offset[0]:offset[-1]]
        offset = offset - offset[0]
        return MultiNestedTensor(
            num_rows=end - start,
            num_cols=self.num_cols,
            values=values,
            offset=offset,
        )

    def _col_narrow(self, start: int, length: int) -> MultiNestedTensor:
        r"""Helper function called by :meth:`MultiNestedTensor.narrow`."""
        assert start >= 0
        assert length > 0
        end = start + length
        if start == 0:
            assert end < self.num_cols
            offset_mat = self.offset[:-1].reshape(self.num_rows, self.num_cols)
            offset_mat = offset_mat[:, start:end + 1]
        else:
            offset_mat = self.offset[1:].reshape(self.num_rows, self.num_cols)
            offset_mat = offset_mat[:, start - 1:end]

        offset_start = offset_mat[:, 0]
        count = offset_mat[:, -1] - offset_start
        batch, arange = _batched_arange(count)
        values = self.values[offset_start[batch] + arange]

        offset_mat_zero_start = offset_mat - offset_start.view(-1, 1)
        accum = torch.cumsum(offset_mat_zero_start[:, -1], dim=0)
        offset_mat_zero_start[1:] += accum[:-1].view(-1, 1)
        num_cols = end - start
        offset = torch.full((self.num_rows * num_cols + 1, ),
                            cast(int, accum[-1]))
        offset[:-1] = offset_mat_zero_start[:, :-1].flatten()
        return MultiNestedTensor(
            num_rows=self.num_rows,
            num_cols=num_cols,
            values=values,
            offset=offset,
        )

    def _row_index_select(self, index: Tensor) -> MultiNestedTensor:
        r"""Helper function called by :obj:`index_select`."""
        # Calculate values
        if index.numel() == 0:
            return self._empty(dim=0)
        index_right = (index + 1) * self.num_cols
        index_left = index * self.num_cols
        diff = self.offset[index_right] - self.offset[index_left]
        batch, arange = _batched_arange(diff)
        idx = self.offset[index_left][batch] + arange
        values = self.values[idx]

        # Calculate offset
        count = torch.full(
            size=(len(index), ),
            fill_value=self.num_cols,
            dtype=torch.long,
            device=self.device,
        )
        count[-1] += 1
        batch, arange = _batched_arange(count)
        idx = index_left[batch] + arange
        offset = self.offset[idx] - self.offset[index_left][batch]
        diff_cumsum = torch.cumsum(diff, dim=0)
        diff_cumsum = torch.roll(diff_cumsum, 1)
        diff_cumsum[0] = 0
        offset = offset + diff_cumsum[batch]
        return MultiNestedTensor(
            num_rows=len(index),
            num_cols=self.num_cols,
            values=values,
            offset=offset,
        )

    def _col_index_select(self, index: Tensor) -> MultiNestedTensor:
        r"""Helper function called by :obj:`index_select`."""
        if index.numel() == 0:
            return self._empty(dim=1)
        start_idx = (index + torch.arange(
            0,
            self.num_rows * self.num_cols,
            self.num_cols,
            device=self.device,
        ).view(-1, 1)).flatten()
        offset_start = self.offset[start_idx]
        count = self.offset[start_idx + 1] - self.offset[start_idx]
        offset = count.new_zeros(count.numel() + 1)
        torch.cumsum(count, dim=0, out=offset[1:])
        batch, arange = _batched_arange(count)
        values = self.values[offset_start[batch] + arange]
        return MultiNestedTensor(
            num_rows=self.num_rows,
            num_cols=len(index),
            values=values,
            offset=offset,
        )

    def _single_index_select(self, index: int, dim: int) -> MultiNestedTensor:
        r"""Get :obj:`index`-th row (:obj:`dim=0`) or column (:obj:`dim=1`)."""
        dim = MultiNestedTensor._normalize_dim(dim)
        index = self._normalize_index(index, dim=dim)
        start_idx: int | Tensor
        if dim == 0:
            start_idx = index * self.num_cols
            end_idx = (index + 1) * self.num_cols + 1
            offset = self.offset[start_idx:end_idx]
            values = self.values[offset[0]:offset[-1]]
            offset = offset - offset[0]
            return MultiNestedTensor(num_rows=1, num_cols=self.num_cols,
                                     values=values, offset=offset)
        elif dim == 1:
            start_idx = torch.arange(
                0,
                self.num_rows * self.num_cols,
                self.num_cols,
                device=self.device,
            ) + index
            diff = self.offset[start_idx + 1] - self.offset[start_idx]
            batch, arange = _batched_arange(diff)
            # Compute values
            values = self.values[self.offset[start_idx][batch] + arange]
            # Compute offset
            offset = diff.new_zeros(diff.numel() + 1)
            torch.cumsum(diff, dim=0, out=offset[1:])
            return MultiNestedTensor(num_rows=self.num_rows, num_cols=1,
                                     values=values, offset=offset)
        else:
            raise RuntimeError(f"Unsupported dim={dim} for index_select.")

    def fillna_col(
        self,
        col_index: int,
        fill_value: int | float | Tensor,
    ) -> None:
        start_idx = torch.arange(
            col_index,
            self.num_rows * self.num_cols,
            self.num_cols,
            device=self.device,
        )
        diff = self.offset[start_idx + 1] - self.offset[start_idx]
        batch, arange = _batched_arange(diff)
        # Compute values
        values_index = self.offset[start_idx][batch] + arange
        values_col = self.values[values_index]
        if self.values.is_floating_point():
            values_col[torch.isnan(values_col)] = fill_value
        else:
            values_col[values_col == -1] = fill_value
        self.values[values_index] = values_col

    def to_dense(self, fill_value: int | float) -> Tensor:
        """Map MultiNestedTensor into dense Tensor representation with padding.

        Args:
            fill_value (Union[int, float]): Fill values.

        Returns:
            Tensor: Padded PyTorch Tensor object with shape
                :obj:`(num_rows, num_cols, max_length)`
        """
        count = self.offset[1:] - self.offset[:-1]
        max_length = cast(int, count.max())
        batch, arange = _batched_arange(count)
        dense = self.values.new_full(
            (self.num_rows, self.num_cols, max_length),
            fill_value=fill_value,
        )
        row = batch // self.num_cols
        col = batch % self.num_cols
        dense[row, col, arange] = self.values
        return dense

    def _empty(self, dim: int) -> MultiNestedTensor:
        r"""Creates an empty :class:`MultiNestedTensor`.

        Args:
            dim (int): The dimension to empty.

        Returns:
            MultiNestedTensor: An empty :class:`MultiNestedTensor`.
        """
        values = torch.tensor([], device=self.device, dtype=self.dtype)
        offset = torch.zeros(1, device=self.device, dtype=torch.long)
        return MultiNestedTensor(
            num_rows=0 if dim == 0 else self.num_rows,
            num_cols=0 if dim == 1 else self.num_cols,
            values=values,
            offset=offset,
        )

    # Static methods ##########################################################
    @staticmethod
    def cat(
        xs: Sequence[MultiNestedTensor],
        dim: int = 0,
    ) -> MultiNestedTensor:
        """Concatenates a sequence of :class:`MultiNestedTensor` along the
        specified dimension.

        Args:
            xs (Sequence[MultiNestedTensor]): A sequence of
                :class:`MultiNestedTensor` to be concatenated.
            dim (int): The dimension to concatenate along.

        Returns:
            MultiNestedTensor: Concatenated multi nested tensor.
        """
        if len(xs) == 0:
            raise RuntimeError("Cannot concatenate a sequence of length 0.")
        assert isinstance(xs[0], MultiNestedTensor)
        dim = MultiNestedTensor._normalize_dim(dim)
        device = xs[0].device
        if dim == 0:
            num_rows = sum(x.num_rows for x in xs)
            num_cols = xs[0].num_cols
            for x in xs[1:]:
                if x.num_cols != num_cols:
                    raise RuntimeError(
                        "num_cols must be the same across a list of input "
                        "multi nested tensors.")
            values = torch.cat([x.values for x in xs], dim=0)

            offset = torch.empty(num_rows * num_cols + 1, dtype=torch.long,
                                 device=device)
            accum = 0
            idx = 0
            for x in xs[:-1]:
                offset[idx:idx + len(x.offset[:-1])] = x.offset[:-1]
                offset[idx:idx + len(x.offset[:-1])].add_(accum)
                accum += cast(int, x.offset[-1])
                idx += len(x.offset[:-1])
            offset[idx:] = xs[-1].offset
            offset[idx:].add_(accum)
            return MultiNestedTensor(
                num_rows=num_rows,
                num_cols=num_cols,
                values=values,
                offset=offset,
            )
        else:
            num_rows = xs[0].num_rows
            num_cols = sum(x.num_cols for x in xs)
            for x in xs[1:]:
                if x.num_rows != num_rows:
                    raise RuntimeError(
                        "num_rows must be the same across a list of input "
                        "multi nested tensors.")

            # (i,j)-th element stores the length of its stored Tensor
            elem_length_mat = torch.empty(num_rows, num_cols, dtype=torch.long,
                                          device=device)
            col_start_idx = 0
            for x in xs:
                elem_count = x.offset[1:] - x.offset[:-1]
                elem_length_mat[:, col_start_idx:col_start_idx +
                                x.num_cols] = elem_count.reshape(
                                    x.num_rows, x.num_cols)
                col_start_idx += x.num_cols

            # Compute offset
            offset = torch.zeros(num_rows * num_cols + 1, dtype=torch.long,
                                 device=device)
            torch.cumsum(elem_length_mat.flatten(), dim=0, out=offset[1:])

            # Compute values
            values = torch.empty(
                sum([x.values.numel() for x in xs]),
                dtype=xs[0].values.dtype,
                device=device,
            )
            col_start_idx = 0
            for x in xs:
                offset_start_idx = col_start_idx + torch.arange(
                    0, num_rows * num_cols, num_cols, device=device)
                offset_start = offset[offset_start_idx]
                offset_end_idx = offset_start_idx + x.num_cols
                offset_end = offset[offset_end_idx]
                count = offset_end - offset_start
                batch, arange = _batched_arange(count)
                values[offset_start[batch] + arange] = x.values
                col_start_idx += x.num_cols

            return MultiNestedTensor(
                num_rows=num_rows,
                num_cols=num_cols,
                values=values,
                offset=offset,
            )
