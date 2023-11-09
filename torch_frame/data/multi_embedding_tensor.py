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
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        values: Tensor,
        offset: Tensor,
    ) -> None:
        super().__init__(num_rows, num_cols, values, offset)
        assert offset[0] == 0
        assert len(offset) == num_cols + 1
        assert values.size() == (num_rows, offset[-1])

    def __getitem__(
        self,
        index: Any,
    ) -> Union['MultiEmbeddingTensor', Tensor]:
        if isinstance(index, tuple) and len(index) == 2 and isinstance(
                index[0], int) and isinstance(index[1], int):
            i = index[0]
            j = index[1]
            return self.values[i, self.offset[j]:self.offset[j + 1]]

        # TODO(akihironitta): Support more index types
        raise NotImplementedError

    @classmethod
    def from_tensor_list(
        cls,
        tensor_list: List[Tensor],
    ) -> 'MultiEmbeddingTensor':
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

    @staticmethod
    def cat(
        xs: Sequence['MultiEmbeddingTensor'],
        dim: int = 0,
    ) -> 'MultiEmbeddingTensor':
        if len(xs) == 0:
            raise ValueError('Cannot concatenate a sequence of length 0.')

        device = xs[0].device
        for x in xs:
            msg = "xs must be a list of MultiEmbeddingTensor."
            assert isinstance(x, MultiEmbeddingTensor), msg
            msg = "device must be the same across a sequence of MultiEmbeddingTensor."
            assert x.device == device, msg

        dim = MultiEmbeddingTensor._normalize_dim(dim)

        if dim == 0:
            # values: [num_rows_1, dim1+dim2], [num_rows_2, dim1+dim2]
            # values: [num_rows_1+num_rows_2, dim1+dim2]
            num_rows = sum(x.num_rows for x in xs)
            num_cols = xs[0].num_cols
            for x in xs[1:]:
                if x.num_cols != num_cols:
                    raise RuntimeError(
                        "num_cols must be the same across a list of input "
                        "multi embedding tensors.")
            values = torch.cat([x.values for x in xs], dim=0)
            offset = xs[0].offset  # TODO: clone()
            return MultiEmbeddingTensor(num_rows, num_cols, values, offset)

        elif dim == 1:
            # values: [num_rows, dim1], [num_rows, dim2]
            # values: [num_rows, dim1+dim2]
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
            offset = torch.LongTensor(offset_list)
            return MultiEmbeddingTensor(num_rows, num_cols, values, offset)
