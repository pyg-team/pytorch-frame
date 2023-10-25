import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor


class MultiNestedTensor:
    r"""A PyTorch Tensor-based data structure that stores
    :obj:`[num_rows, num_cols, *]` , where the size of last dimension can be
    different for different data/column. Internally, we store the object in an
    efficient flattened format: :obj:`(value, offset)`, where the element
    at :obj:`(i, j)` is accessed by

    Args:
        num_rows (int): Numnber of rows.
        num_cols (int): Number of columns.
        values (Tensor): The values Tensor.
        offset (Tensor): The offset Tensor.
        device (torch.device): Device to put.
    """
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        values: Dict[str, Tensor],
        offset: Tensor,
        device: torch.device = None,
    ):
        assert offset[0] == 0
        assert offset[-1] == len(values)
        assert len(offset) == num_rows * num_cols + 1
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.device = device
        self.values = values.to(self.device)
        self.offset = offset.to(self.device)

    @classmethod
    def from_tensor_mat(
        cls,
        tensor_mat: List[List[Tensor]],
        device: torch.device = None,
    ) -> 'MultiNestedTensor':
        r"""Construct :class:`MultiNestedTensor` object from
        :obj:`tensor_mat`.

        Args:
            tensor_mat List[List[Tensor]]: A dictionary of
                matrix of PyTorch Tensors. :obj:`tensor_mat[i][j]` contains
                1-dim PyTorch Tensor of :obj:`i`-th row and :obj:`j`-th column.
            device (torch.device): Device to put.

        Returns:
            MultiNestedTensor: Returned class object.
        """
        num_rows = len(tensor_mat)
        num_cols = len(tensor_mat[0])

        offset_list = []
        accum_idx = 0
        offset_list.append(accum_idx)
        values_list = []
        for i in range(num_rows):
            for j in range(num_cols):
                tensor = tensor_mat[i][j]
                if not isinstance(tensor, Tensor):
                    raise RuntimeError(
                        "The element of tensor_mat must be PyTorch Tensor")
                values_list.append(tensor)
                accum_idx += len(tensor)
                offset_list.append(accum_idx)

        values = torch.cat(values_list)
        offset = torch.tensor(offset_list, dtype=torch.long)

        return cls(num_rows, num_cols, values, offset, device)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        name += f"(num_rows={self.num_rows}, num_cols={self.num_cols} "
        name += f"device={self.device})"
        return name

    def __copy__(self) -> 'MultiNestedTensor':
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value

        out.values = copy.copy(out.values)
        out.offset = copy.copy(out.offset)

        return out

    def __getitem__(
        self,
        index: Any,
    ) -> Union['MultiNestedTensor', Tensor]:
        if isinstance(index, Tuple):
            # Get an element of (i, j). Returns Tensor
            assert len(index) == 2
            if index[0] < 0 or index[0] >= self.num_rows:
                raise IndexError(
                    f"Index out-of-bounds. The first element of {index} needs "
                    f"to be [0, {self.num_rows}]")
            if index[1] < 0 or index[1] >= self.num_cols:
                raise IndexError(
                    f"Index out-of-bounds. The second element of {index} "
                    f"needs to be [0, {self.num_cols}]")
            idx = index[0] * self.num_cols + index[1]
            start_idx = self.offset[idx]
            end_idx = self.offset[idx + 1]
            out = self.values[start_idx:end_idx]
            return out
        elif isinstance(index, int):
            if index < 0 or index >= self.num_rows:
                raise IndexError(
                    f"Index out-of-bounds. The {index} needs to be "
                    f"[0, {self.num_rows}]")
            # Get i-th row. Returns MultiNestedTensor
            start_idx = index * self.num_cols
            end_idx = (index + 1) * self.num_cols + 1
            offset = self.offset[start_idx:end_idx]
            values = self.values[offset[0]:offset[-1]]
            offset = offset - offset[0]
            return MultiNestedTensor(num_rows=1, num_cols=self.num_cols,
                                     values=values, offset=offset)
        elif isinstance(index, Tensor) and index.ndim == 1:
            return self._row_index_selet_helper(index)
        elif isinstance(index, List):
            return self._row_index_selet_helper(
                torch.LongTensor(index, device=self.device))
        else:
            raise RuntimeError("Advanced indexing not supported yet.")

    def _row_index_selet_helper(self, index: Tensor) -> Tensor:
        if index.min() < 0 or index.max() >= self.num_rows:
            raise IndexError(f"Index out-of-bounds. The {index} needs to be "
                             f"[0, {self.num_rows}]")
        # Calculate values
        count = self.offset[(index + 1) *
                            self.num_cols] - self.offset[index * self.num_cols]
        batch, arange = batched_arange(count)
        idx = self.offset[index * self.num_cols][batch] + arange
        values = self.values[idx]

        # Calculate offset
        count = torch.full(size=(len(index), ), fill_value=self.num_cols,
                           dtype=torch.long, device=self.device)
        count[-1] += 1
        batch, arange = batched_arange(count)
        idx = (index * self.num_cols)[batch] + arange
        offset = self.offset[idx] - self.offset[index * self.num_cols][batch]
        diff = self.offset[(index + 1) *
                           self.num_cols] - self.offset[index * self.num_cols]
        diff_cumsum = torch.cumsum(diff, dim=0)
        diff_cumsum = torch.roll(diff_cumsum, 1)
        diff_cumsum[0] = 0
        offset = offset + diff_cumsum[batch]
        return MultiNestedTensor(num_rows=len(index), num_cols=self.num_cols,
                                 values=values, offset=offset)

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


def batched_arange(count: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Fast implementation of bached version of torch.arange.
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
