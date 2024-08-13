from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from torch_frame.data.multi_tensor import _batched_arange, _MultiTensor


class MultiEmbeddingTensor(_MultiTensor):
    r"""A read-only PyTorch tensor-based data structure that stores
    :obj:`[num_rows, num_cols, *]`, where the size of last dimension can be
    different for different column. Note that the last dimension is the same
    within each column across rows while in :class:`MultiNestedTensor`, the
    last dimension can be different across both rows and columns.
    It supports various advanced indexing, including slicing and list indexing
    along both row and column.

    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
        values (torch.Tensor): The values :class:`torch.Tensor` of size
            :obj:`[num_rows, dim1+dim2+...+dimN]`.
        offset (torch.Tensor): The offset :class:`torch.Tensor` of size
            :obj:`[num_cols+1,]`.

    Example:
        >>> tensor_list = [
        ...    torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),  # emb col 0
        ...    torch.tensor([[0.6, 0.7], [0.8, 0.9]]),            # emb col 1
        ...    torch.tensor([[1.], [1.1]]),                       # emb col 2
        ... ]
        >>> met = MultiEmbeddingTensor.from_tensor_list(tensor_list)
        >>> met
        MultiEmbeddingTensor(num_rows=2, num_cols=3, device='cpu')
        >>> met.values
        tensor([[0.0000, 0.1000, 0.2000, 0.6000, 0.7000, 1.0000],
                [0.3000, 0.4000, 0.5000, 0.8000, 0.9000, 1.1000]])
        >>> met.offset
        tensor([0, 3, 5, 6])
        >>> met[0, 0]
        tensor([0.0000, 0.1000, 0.2000])
        >>> met[1, 1]
        tensor([0.8000, 0.9000])
        >>> met[0] # Row integer indexing
        MultiEmbeddingTensor(num_rows=1, num_cols=3, device='cpu')
        >>> met[:, 0] # Column integer indexing
        MultiEmbeddingTensor(num_rows=2, num_cols=1, device='cpu')
        >>> met[:, 0].values # Embedding of column 0
        tensor([[0.0000, 0.1000, 0.2000],
                [0.3000, 0.4000, 0.5000]])
        >>> met[:1] # Row slicing
        MultiEmbeddingTensor(num_rows=1, num_cols=3, device='cpu')
        >>> met[[0, 1, 0, 0]] # Row list indexing
        MultiEmbeddingTensor(num_rows=4, num_cols=3, device='cpu')
    """
    def validate(self) -> None:
        assert self.offset[0] == 0
        assert len(self.offset) == self.num_cols + 1
        assert self.offset.ndim == 1
        assert self.values.ndim == 2 or self.values.numel() == 0

    @classmethod
    def from_tensor_list(
        cls,
        tensor_list: list[Tensor],
    ) -> MultiEmbeddingTensor:
        r"""Creates a :class:`MultiEmbeddingTensor` from a list of
        :class:`torch.Tensor`.

        Args:
            tensor_list (List[Tensor]): A list of tensors, where each tensor
                has the same number of rows and can have a different number of
                columns.

        Returns:
            MultiEmbeddingTensor: A :class:`MultiEmbeddingTensor` instance.
        """
        assert isinstance(tensor_list, list) and len(tensor_list) > 0
        num_rows = tensor_list[0].size(0)
        device = tensor_list[0].device
        for tensor in tensor_list:
            msg = "tensor_list must be a list of tensors."
            assert isinstance(tensor, torch.Tensor), msg
            msg = "tensor_list must be a list of 2D tensors."
            assert tensor.dim() == 2, msg
            msg = "num_rows must be the same across a list of input tensors."
            assert tensor.size(0) == num_rows, msg
            msg = "device must be the same across a list of input tensors."
            assert tensor.device == device, msg

        offset_list = []
        accum_idx = 0
        offset_list.append(accum_idx)
        for tensor in tensor_list:
            accum_idx += tensor.size(1)
            offset_list.append(accum_idx)

        num_cols = len(tensor_list)
        values = torch.cat(tensor_list, dim=1)
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
        return self.values[i, self.offset[j]:self.offset[j + 1]]

    def _row_narrow(self, start: int, length: int) -> MultiEmbeddingTensor:
        r"""Helper function called by :meth:`MultiEmbeddingTensor.narrow`."""
        return MultiEmbeddingTensor(
            num_rows=length,
            num_cols=self.num_cols,
            values=self.values[start:start + length],
            offset=self.offset,
        )

    def _col_narrow(self, start: int, length: int) -> MultiEmbeddingTensor:
        r"""Helper function called by :meth:`MultiEmbeddingTensor.narrow`."""
        offset = self.offset[start:start + length + 1] - self.offset[start]
        col_offset_start = self.offset[start]
        col_offset_end = self.offset[start + length]
        return MultiEmbeddingTensor(
            num_rows=self.num_rows,
            num_cols=length,
            values=self.values[:, col_offset_start:col_offset_end],
            offset=offset,
        )

    def _row_index_select(self, index: Tensor) -> MultiEmbeddingTensor:
        r"""Helper function called by
        :meth:`MultiEmbeddingTensor.index_select`.
        """
        return MultiEmbeddingTensor(
            num_rows=index.size(0),
            num_cols=self.num_cols,
            values=self.values[index],
            offset=self.offset,
        )

    def _col_index_select(self, index: Tensor) -> MultiEmbeddingTensor:
        r"""Helper function called by
        :meth:`MultiEmbeddingTensor.index_select`.
        """
        if index.numel() == 0:
            return self._empty(dim=1)
        offset = torch.zeros(
            index.size(0) + 1,
            dtype=torch.long,
            device=self.device,
        )
        col_dims = self.offset[1:] - self.offset[:-1]
        new_col_dims = col_dims[index]
        torch.cumsum(new_col_dims, dim=0, out=offset[1:])
        batch, arange = _batched_arange(new_col_dims)
        value_index = self.offset[index][batch] + arange
        return MultiEmbeddingTensor(
            num_rows=self.num_rows,
            num_cols=index.size(0),
            values=self.values[:, value_index],
            offset=offset,
        )

    def _single_index_select(
        self,
        index: int,
        dim: int,
    ) -> MultiEmbeddingTensor:
        r"""Helper function called by
        :meth:`MultiEmbeddingTensor.index_select`.
        """
        index = self._normalize_index(index, dim=dim)
        if dim == 0:
            return MultiEmbeddingTensor(
                num_rows=1,
                num_cols=self.num_cols,
                values=self.values[index].view(1, -1),
                offset=self.offset,
            )
        elif dim == 1:
            value_index = slice(self.offset[index], self.offset[index + 1])
            values = self.values[:, value_index]
            offset = self.offset[[0, index + 1]] - self.offset[[0, index]]
            return MultiEmbeddingTensor(
                num_rows=self.num_rows,
                num_cols=1,
                values=values,
                offset=offset,
            )
        assert False, "Should not reach here."

    def fillna_col(
        self,
        col_index: int,
        fill_value: int | float | Tensor,
    ) -> None:
        values_index = slice(self.offset[col_index],
                             self.offset[col_index + 1])
        values_col = self.values[:, values_index]
        if self.values.is_floating_point():
            values_col[torch.isnan(values_col)] = fill_value
        else:
            values_col[values_col == -1] = fill_value

    def _empty(self, dim: int) -> MultiEmbeddingTensor:
        """Creates an empty :class:`MultiEmbeddingTensor`.

        Args:
            dim (int): The dimension to empty.

        Returns:
            MultiEmbeddingTensor: An empty :class:`MultiEmbeddingTensor`.
                Note that if :obj:`dim=0`, it will return with the original
                offset tensor.
        """
        return MultiEmbeddingTensor(
            num_rows=0 if dim == 0 else self.num_rows,
            num_cols=0 if dim == 1 else self.num_cols,
            values=torch.tensor([], device=self.device, dtype=self.dtype),
            offset=torch.tensor([0], device=self.device, dtype=torch.long)
            if dim == 1 else self.offset,
        )

    @staticmethod
    def cat(
        xs: Sequence[MultiEmbeddingTensor],
        dim: int = 0,
    ) -> MultiEmbeddingTensor:
        """Concatenates a sequence of :class:`MultiEmbeddingTensor` along the
        specified dimension.

        Args:
            xs (Sequence[MultiEmbeddingTensor]): A sequence of
                :class:`MultiEmbeddingTensor` to be concatenated.
            dim (int): The dimension to concatenate along.

        Returns:
            MultiEmbeddingTensor: Concatenated multi embedding tensor.
        """
        if len(xs) == 0:
            raise RuntimeError("Cannot concatenate a sequence of length 0.")

        for x in xs:
            msg = "`xs` must be a list of MultiEmbeddingTensor."
            assert isinstance(x, MultiEmbeddingTensor), msg
            msg = ("device must be the same across a sequence of"
                   " MultiEmbeddingTensor.")
            assert x.device == xs[0].device, msg

        dim = MultiEmbeddingTensor._normalize_dim(dim)

        if len(xs) == 1:
            return xs[0]

        if dim == 0:
            num_rows = sum(x.num_rows for x in xs)
            num_cols = xs[0].num_cols
            for x in xs[1:]:
                if x.num_cols != num_cols:
                    raise RuntimeError(
                        "num_cols must be the same across a list of input "
                        "multi embedding tensors.")
            values = torch.cat([x.values for x in xs], dim=0)
            # NOTE: offset shares the same data with the input's offset,
            # which is inconsistent with when dim=1
            offset = xs[0].offset
            return MultiEmbeddingTensor(num_rows, num_cols, values, offset)

        elif dim == 1:
            num_rows = xs[0].num_rows
            for x in xs[1:]:
                if x.num_rows != num_rows:
                    raise RuntimeError(
                        "num_rows must be the same across a list of input "
                        "multi embedding tensors.")
            num_cols = sum(x.num_cols for x in xs)
            values = torch.cat([x.values for x in xs], dim=1)
            offset_list = [0]
            for x in xs:
                offset_list.extend(x.offset[1:] + offset_list[-1])
            # NOTE: offset is a data copy of the input's offset,
            # which is inconsistent with when dim=0
            offset = torch.tensor(offset_list)
            return MultiEmbeddingTensor(num_rows, num_cols, values, offset)

        assert False, "Should not reach here."
