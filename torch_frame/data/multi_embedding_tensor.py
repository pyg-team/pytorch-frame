from typing import Any, List, Sequence, Union

import torch
from torch import Tensor

from torch_frame.data.multi_tensor import _MultiTensor


class MultiEmbeddingTensor(_MultiTensor):
    r"""A PyTorch tensor-based data structure that stores
    :obj:`[num_rows, num_cols, *]`, where the size of last dimension can be
    different for different column.

    Note that the last dimension is the same within each column across rows
    while in :class:`MultiNestedTensor`, the last dimension can be different
    across both rows and columns.

    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
        values (torch.Tensor): The values :class:`torch.Tensor` of size
            :obj:`[num_rows, dim1+dim2+...+dimN]`.
        offset (torch.Tensor): The offset :class:`torch.Tensor` of size
            :obj:`[num_cols+1,]`.

    Example:
        >>> num_rows = 2
        >>> tensor_list = [
        ...    torch.tensor([[0, 1, 2], [3, 4, 5]]),  # col0
        ...    torch.tensor([[6, 7], [8, 9]]),        # col1
        ...    torch.tensor([[10], [11]]),            # col2
        ... ]
        >>> out = MultiEmbeddingTensor.from_tensor_list(tensor_list)
        >>> out
        MultiEmbeddingTensor(num_rows=2, num_cols=3, device='cpu')
        >>> out[0, 2]
        tensor([10])
    """
    def validate(self):
        assert self.offset[0] == 0
        assert len(
            self.offset) == self.num_cols + 1 or self.values.numel() == 0
        assert self.offset.ndim == 1
        assert self.values.ndim == 2 or self.values.numel() == 0

    @classmethod
    def from_tensor_list(
        cls,
        tensor_list: List[Tensor],
    ) -> "MultiEmbeddingTensor":
        r"""Creates a :class:`MultiEmbeddingTensor` from a list of
        :class:`torch.Tensor`.

        Args:
            tensor_list (List[Tensor]): A list of tensors, where each tensor
                has the same number of rows and can have a different number of
                columns.

        Returns:
            MultiEmbeddingTensor: A :class:`MultiEmbeddingTensor` instance.

        Example:
            >>> num_rows = 2
            >>> tensor_list = [
            ...    torch.tensor([[0, 1, 2], [3, 4, 5]]),  # col0
            ...    torch.tensor([[6, 7], [8, 9]]),        # col1
            ...    torch.tensor([[10], [11]]),            # col2
            ... ]
            >>> out = MultiEmbeddingTensor.from_tensor_list(tensor_list)
            >>> out
            MultiEmbeddingTensor(num_rows=2, num_cols=3, device='cpu')
            >>> out[0, 0]
            tensor([0, 1, 2])
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
        offset = torch.LongTensor(offset_list)
        return cls(num_rows, num_cols, values, offset)

    def __getitem__(
        self,
        index: Any,
    ) -> Union["MultiEmbeddingTensor", Tensor]:
        if isinstance(index, tuple):
            assert len(index) == 2
            if isinstance(index[0], int) and isinstance(index[1], int):
                i = self._normalize_index(index[0], dim=0)
                j = self._normalize_index(index[1], dim=1)
                return self.values[i, self.offset[j]:self.offset[j + 1]]
            else:
                out = self
                for dim, idx in enumerate(index):
                    out = out.select(idx, dim=dim)
                return out
        else:
            return self.select(index, dim=0)

    def select(
        self,
        index: Union[int, Tensor, List[int], slice, range],
        dim: int,
    ) -> "MultiEmbeddingTensor":
        """Returns a new :class:`MultiEmbeddingTensor` which indexes the input
        :class:`MultiEmbeddingTensor` along the specified dimension.

        Args:
            index (Union[int, Tensor, List[int], slice]): An index to select.
            dim (int): The dimension to index in.

        Returns:
            MultiEmbeddingTensor: A :class:`MultiEmbeddingTensor` instance.

        Note:
            :obj:`select(index, dim=0)` is equivalent to
            :obj:`multi_embedding_tensor[index]`, and
            :obj:`select(index, dim=1)` is equivalent to
            :obj:`multi_embedding_tensor[:, index]`.

        Example:
            >>> num_rows = 2
            >>> tensor_list = [
            ...    torch.tensor([[0, 1, 2], [3, 4, 5]]),  # col0
            ...    torch.tensor([[6, 7], [8, 9]]),        # col1
            ...    torch.tensor([[10], [11]]),            # col2
            ... ]
            >>> out = MultiEmbeddingTensor.from_tensor_list(tensor_list)
            >>> out
            MultiEmbeddingTensor(num_rows=2, num_cols=3, device='cpu')
            >>> # int can select one row/col
            >>> out.select(0, dim=0)
            MultiEmbeddingTensor(num_rows=1, num_cols=3, device='cpu')
            >>> # list/tensor/range can select multiple rows/cols
            >>> out.select([0, 2], dim=1)
            MultiEmbeddingTensor(num_rows=2, num_cols=2, device='cpu')
        """
        dim = self._normalize_dim(dim)
        if isinstance(index, int):
            return self._single_index_select(index, dim=dim)
        elif isinstance(index, slice):
            return self._slice(index, dim=dim)
        elif isinstance(index, Tensor) and index.ndim == 1:
            return self.index_select(index, dim=dim)
        # TODO: Don't materialize range but instead pass it to PyTorch tensor
        # directly for possible better performance.
        elif isinstance(index, (list, range)):
            return self.index_select(
                torch.tensor(index, dtype=torch.long, device=self.device),
                dim=dim,
            )
        raise NotImplementedError

    def narrow(
        self,
        dim: int,
        start: int,
        length: int,
    ) -> "MultiEmbeddingTensor":
        """Returns a narrowed version of input :class:`MultiEmbeddingTensor`.

        Args:
            dim (int): The dimension along which to narrow.
            start (int): The starting dimension index to narrow.
            length (int): The length of the dimension to narrow.

        Returns:
            MultiEmbeddingTensor: A :class:`MultiEmbeddingTensor` instance.

        Example:
            >>> num_rows = 2
            >>> tensor_list = [
            ...    torch.tensor([[0, 1, 2], [3, 4, 5]]),  # col0
            ...    torch.tensor([[6, 7], [8, 9]]),        # col1
            ...    torch.tensor([[10], [11]]),            # col2
            ... ]
            >>> out = MultiEmbeddingTensor.from_tensor_list(tensor_list)
            >>> out
            MultiEmbeddingTensor(num_rows=2, num_cols=3, device='cpu')
            >>> out.narrow(dim=1, start=1, length=2)
            MultiEmbeddingTensor(num_rows=2, num_cols=2, device='cpu')
        """
        dim = self._normalize_dim(dim)
        num_data = self.num_rows if dim == 0 else self.num_cols
        if start == 0 and start + length >= num_data:
            return self
        elif length <= 0:
            return MultiEmbeddingTensor(
                num_rows=0 if dim == 0 else self.num_rows,
                num_cols=0 if dim == 1 else self.num_cols,
                values=torch.tensor([], device=self.device),
                offset=torch.tensor([0], device=self.device),
            )

        elif dim == 0:
            return MultiEmbeddingTensor(
                num_rows=length,
                num_cols=self.num_cols,
                values=self.values[start:start + length],
                offset=self.offset,
            )
        elif dim == 1:
            offset = self.offset[start:start + length + 1] - self.offset[start]
            return MultiEmbeddingTensor(
                num_rows=self.num_rows,
                num_cols=length,
                values=self.values[:, self.offset[start]:self.offset[start +
                                                                     length]],
                offset=offset,
            )

    def _slice(
        self,
        index: slice,
        dim: int,
    ) -> "MultiEmbeddingTensor":
        dim = self._normalize_dim(dim)
        num_data = self.num_rows if dim == 0 else self.num_cols
        if index.step is not None and index.step > 1:
            idx = torch.tensor(
                range(num_data)[index],
                device=self.device,
                dtype=torch.long,
            )
            return self.index_select(idx, dim=dim)
        else:
            start_idx: int = self._normalize_index(index.start or 0, dim=dim)
            end_idx: int = self._normalize_index(
                index.stop or num_data,
                dim=dim,
                is_slice_end=True,
            )
            return self.narrow(
                dim=dim,
                start=start_idx,
                length=end_idx - start_idx,
            )

    def index_select(
        self,
        index: Tensor,
        dim: int,
    ) -> 'MultiEmbeddingTensor':
        """Returns a :class:`MultiEmbeddingTensor` which indexes the input
        :class:`MultiEmbeddingTensor` along the specified dimension.

        Args:
            index (Tensor): A 1-D tensor of indices to select.
            dim (int): The dimension to index in.

        Returns:
            MultiEmbeddingTensor: A :class:`MultiEmbeddingTensor` instance.
        """
        dim = self._normalize_dim(dim)
        index = self._normalize_index(index, dim=dim)
        if dim == 0:
            return self._row_index_select(index)
        elif dim == 1:
            return self._col_index_select(index)

    def _row_index_select(self, index: Tensor) -> 'MultiEmbeddingTensor':
        return MultiEmbeddingTensor(
            num_rows=index.size(0),
            num_cols=self.num_cols,
            values=self.values[index],
            offset=self.offset,
        )

    def _col_index_select(self, index: Tensor) -> 'MultiEmbeddingTensor':
        if index.numel() == 0:
            return MultiEmbeddingTensor(
                num_rows=self.num_rows,
                num_cols=0,
                values=torch.tensor([], device=self.device),
                offset=torch.tensor([0], device=self.device),
            )
        offset = torch.zeros(
            index.size(0) + 1,
            dtype=torch.long,
            device=self.device,
        )
        col_dims = self.offset[1:] - self.offset[:-1]
        new_col_dims = col_dims[index]
        torch.cumsum(new_col_dims, dim=0, out=offset[1:])
        value_index = [
            torch.arange(
                self.offset[i],
                self.offset[i + 1],
                dtype=torch.long,
                device=self.device,
            ) for i in index
        ]
        value_index = torch.cat(value_index, dim=0)
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
    ) -> "MultiEmbeddingTensor":
        index = self._normalize_index(index, dim=dim)
        if dim == 0:
            return MultiEmbeddingTensor(
                num_rows=1,
                num_cols=self.num_cols,
                values=self.values[index].view(1, -1),
                offset=self.offset,
            )
        elif dim == 1:
            return MultiEmbeddingTensor(
                num_rows=self.num_rows,
                num_cols=1,
                values=self.values[:,
                                   self.offset[index]:self.offset[index + 1]],
                offset=self.offset[[0, index + 1]],
            )

    @staticmethod
    def cat(
        xs: Sequence["MultiEmbeddingTensor"],
        dim: int = 0,
    ) -> "MultiEmbeddingTensor":
        """Concatenates a sequence of :class:`MultiEmbeddingTensor` along the
        specified dimension.

        Args:
            xs (Sequence[MultiEmbeddingTensor]): A sequence of
                :class:`MultiEmbeddingTensor` to be concatenated.
            dim (int): The dimension to concatenate along.

        Returns:
            MultiEmbeddingTensor: Concatenated multi embedding tensor that
                shares the same data as the input multi embedding tensors.

        Example:
            >>> from torch_frame.data import MultiEmbeddingTensor
            >>> tensor_list1 = [
            ...     torch.tensor([[0, 1, 2], [6, 7, 8]]),  # col1
            ...     torch.tensor([[3, 4], [9, 10]]),       # col2
            ...     torch.tensor([[5], [11]]),             # col3
            ... ]
            >>> tenosor_list2 = [
            ...     torch.tensor([[12, 13, 14]]),          # col1
            ...     torch.tensor([[15, 16]]),              # col2
            ...     torch.tensor([[17]]),                  # col3
            ... ]
            >>> met1 = MultiEmbeddingTensor.from_tensor_list(tensor_list1)
            >>> met2 = MultiEmbeddingTensor.from_tensor_list(tensor_list2)
            >>> met1
            MultiEmbeddingTensor(num_rows=2, num_cols=3, device='cpu')
            >>> met2
            MultiEmbeddingTensor(num_rows=1, num_cols=3, device='cpu')
            >>> out = MultiEmbeddingTensor.cat([met1, met2], dim=0)
            >>> out
            MultiEmbeddingTensor(num_rows=3, num_cols=3, device='cpu')
            >>> out.values
            tensor([[ 0,  1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10, 11],
                    [12, 13, 14, 15, 16, 17]])
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
            offset = torch.LongTensor(offset_list)
            return MultiEmbeddingTensor(num_rows, num_cols, values, offset)
