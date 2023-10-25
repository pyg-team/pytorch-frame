import copy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor


class MultiNestedTensor:
    r"""A PyTorch Tensor-based data structure that stores
    :obj:`[num_rows, num_cols, *]` , where the size of last dimension can be
    different for different data/column. Internally, we store the object in an
    efficient flattened format: :obj:`(value, offset)`, where the element
    at :obj:`(i, j)` is accessed by
    :obj:`values[offset[i*num_cols+j]:offset[i*num_cols+j+1]]`.
    We store a dictionary of :obj:`values` so that the element at :obj:`(i, j)`
    can be a dictionary of tensors.

    Args:
        num_rows (int): Numnber of rows.
        num_cols (int): Number of columns.
        values_dict (Dict[str, Tensor]): A dictionary of values Tensor.
        offset (Tensor): The offset Tensor.


    """
    def __init__(self, num_rows: int, num_cols: int,
                 values_dict: Dict[str, Tensor], offset: Tensor):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.values_dict = values_dict
        self.offset = offset

    @classmethod
    def from_tensor_mat_dict(
        cls,
        tensor_mat_dict: Dict[str, List[List[Tensor]]],
    ) -> 'MultiNestedTensor':
        r"""Construct :class:`MultiNestedTensor` object from
        :obj:`tensor_mat_dict`.

        Args:
            tensor_mat_dict Dict[str, List[List[Tensor]]]: A dictionary of
                matrix of PyTorch Tensors. :obj:`tensor_mat[i][j]` contains
                1-dim PyTorch Tensor of :obj:`i`-th row and :obj:`j`-th column.
                If there are multiple keys, we require the shape of
                :obj:`tensor_mat_dict[key1][i][j]` and
                :obj:`tensor_mat_dict[key2][i][j]` to be the same for every
                :obj:`(i, j)`.

        Returns:
            MultiNestedTensor: Returned class object.
        """
        first_name = next(iter(tensor_mat_dict))
        first_tensor_mat = tensor_mat_dict[first_name]
        num_rows = len(first_tensor_mat)
        num_cols = len(first_tensor_mat[0])

        offset_list = []
        accum_idx = 0
        offset_list.append(accum_idx)
        values_list_dict = defaultdict(list)
        for row_i in range(num_rows):
            for col_j in range(num_cols):
                for name, tensor_mat in tensor_mat_dict.items():
                    tensor = tensor_mat[row_i][col_j]
                    if not isinstance(tensor, Tensor):
                        raise RuntimeError(
                            "The element of tensor_mat must be PyTorch Tensor")
                    if tensor.ndim != 1:
                        raise RuntimeError(
                            "Each element of tensor_mat_dict must be "
                            "1-dimensional.")
                    if tensor.shape != first_tensor_mat[row_i][col_j].shape:
                        raise RuntimeError(
                            "The shape of tensor_mat_dict[key1][i][j] must be "
                            "the same for every (i, j).")
                    values_list_dict[name].append(tensor)
                accum_idx += len(tensor)
                offset_list.append(accum_idx)

        values_dict = {}
        for name, value_list in values_list_dict.items():
            values_dict[name] = torch.cat(value_list)
        offset = torch.tensor(offset_list, dtype=torch.long)

        return cls(num_rows, num_cols, values_dict, offset)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        name += f"(num_rows: {self.num_rows}, num_cols: {self.num_cols})"
        return name

    def __copy__(self) -> 'MultiNestedTensor':
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value

        out.values_dict = copy.copy(out.values_dict)
        out.offset = copy.copy(out.offset)

        return out

    def __getitem__(
        self,
        index: Any,
    ) -> Union['MultiNestedTensor', Dict[str, Tensor]]:
        if isinstance(index, Tuple):
            # Get an element of (i, j). Returns Dict[str, Tensor]
            assert len(index) == 2
            if index[0] < 0 or index[0] >= self.num_rows:
                raise IndexError(
                    f"Index out-of-bounds. The first element of {index} needs "
                    f"to be [0, {self.num_rows}]")
            if index[1] < 0 or index[1] >= self.num_cols:
                raise IndexError(
                    f"Index out-of-bounds. The second element of {index} "
                    f"needs to be [0, {self.num_cols}]")
            out = {}
            idx = index[0] * self.num_cols + index[1]
            start_idx = self.offset[idx]
            end_idx = self.offset[idx + 1]
            for name, values in self.values_dict.items():
                out[name] = values[start_idx:end_idx]
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
            values_dict = {}
            for name, values in self.values_dict.items():
                values_dict[name] = values[offset[0]:offset[-1]]
            offset = offset - offset[0]
            return MultiNestedTensor(num_rows=1, num_cols=self.num_cols,
                                     values_dict=values_dict, offset=offset)
        elif isinstance(index, Tensor) and index.ndim == 1:
            return self._row_index_selet_helper(index)
        elif isinstance(index, List):
            return self._row_index_selet_helper(torch.LongTensor(index))
        else:
            raise RuntimeError("Advanced indexing not supported yet.")

    def _row_index_selet_helper(self, index: Tensor) -> Tensor:
        if index.min() < 0 or index.max() >= self.num_rows:
            raise IndexError(f"Index out-of-bounds. The {index} needs to be "
                             f"[0, {self.num_rows}]")
        # Calculate values_dict
        count = self.offset[(index + 1) *
                            self.num_cols] - self.offset[index * self.num_cols]
        batch, arange = batched_arange(count)
        idx = self.offset[index * self.num_cols][batch] + arange
        values_dict = {}
        for name in self.values_dict.keys():
            values_dict[name] = self.values_dict[name][idx]

        # Calculate offset
        count = torch.full(size=(len(index), ), fill_value=self.num_cols,
                           dtype=torch.long)
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
                                 values_dict=values_dict, offset=offset)

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
        out.values_dict = {
            name: fn(values)
            for name, values in out.values_dict.items()
        }
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
