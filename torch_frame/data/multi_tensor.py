from __future__ import annotations

import copy
from typing import Any, Callable, Sequence, TypeVar

import torch
from torch import Tensor

T = TypeVar("T", int, Tensor)


class _MultiTensor:
    ndim = 3

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        values: Tensor,
        offset: Tensor,
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.values = values
        self.offset = offset
        self.validate()

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, int | Tensor]:
        r"""Serialize the object into a dictionary."""
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "values": self.values,
            "offset": self.offset,
        }

    def __setitem__(self, index: Any, values: Any) -> None:
        raise RuntimeError("Setting values is not currently supported by "
                           f"{self.__class__.__name__}.")

    def __repr__(self) -> str:
        return " ".join([
            f"{self.__class__.__name__}(num_rows={self.num_rows},",
            f"num_cols={self.num_cols},",
            f"device='{self.device}')",
        ])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_rows, self.num_cols, -1)

    def size(self, dim: int) -> int:
        dim = self._normalize_dim(dim)
        if dim == 0:
            return self.num_rows
        elif dim == 1:
            return self.num_cols
        assert False, "Should not reach here."

    def dim(self) -> int:
        return self.ndim

    def __len__(self) -> int:
        return self.num_rows

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def is_floating_point(self) -> bool:
        return self.values.is_floating_point()

    def clone(self) -> _MultiTensor:
        return self.__class__(
            self.num_rows,
            self.num_cols,
            self.values.clone(),
            self.offset.clone(),
        )

    # Device Transfer #########################################################

    def to(self, *args, **kwargs):
        return self._apply(lambda x: x.to(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return self._apply(lambda x: x.cpu(*args, **kwargs))

    def cuda(self, *args, **kwargs):
        return self._apply(lambda x: x.cuda(*args, **kwargs))

    # Helper Functions ########################################################

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> _MultiTensor:
        out = copy.copy(self)
        out.values = fn(out.values)
        out.offset = fn(out.offset)
        return out

    @classmethod
    def _normalize_dim(cls, dim: int) -> int:
        r"""Normalize :obj:`dim` argument from range [-2, 1] to range [0, 1].

        Raises:
            IndexError: If :obj:`dim` is out of range [-2, 1].
        """
        if dim < 0:
            dim = dim + cls.ndim
        if dim == 2:
            raise IndexError(
                f"{cls.__name__} does not have a fixed length on the third"
                "dimension.")
        if dim not in [0, 1]:
            raise IndexError(
                "Dimension out of range (expected to be in range of [-2, 1],"
                f" but got {dim}.")
        return dim

    def _normalize_index(
        self,
        index: T,
        dim: int,
        check_out_of_bounds: bool = True,
    ) -> T:
        """Helper function to map negative indices to positive indices and
        raise :obj:`IndexError` when necessary.

        Args:
            index (int or Tensor): Input :obj:`index` with potentially negative
                elements.
            dim (int): Dimension to be indexed.
            check_out_of_bounds (bool, optional): Whether to perform
                out-of-bound checks. (default: :obj:`True`)
        """
        dim = self._normalize_dim(dim)
        max_entries = self.num_rows if dim == 0 else self.num_cols
        if isinstance(index, int):
            if index < 0:
                index = index + max_entries
            if index < 0 or (check_out_of_bounds and index >= max_entries):
                raise IndexError(f"index {index} is out of bounds for "
                                 f"dimension {dim} with size {max_entries}")
        elif isinstance(index, Tensor):
            assert index.dim() == 1
            if index.dtype == torch.bool:  # Convert boolean mask to index.
                assert index.numel() == max_entries
                index = index.nonzero().flatten()
            else:
                neg_mask = index < 0
                if neg_mask.any():
                    index = index.clone()
                    index[neg_mask] += max_entries
                if (index.numel() > 0 and check_out_of_bounds
                        and (index.min() < 0 or index.max() >= max_entries)):
                    raise IndexError(f"index is out of bounds for dimension "
                                     f"{dim} with size {max_entries}")

        return index

    @classmethod
    def allclose(
        cls,
        tensor1: _MultiTensor,
        tensor2: _MultiTensor,
        equal_nan: bool = False,
    ) -> bool:
        r"""Returns whether given two tensors are all close or not.

        Args:
            tensor1 (_MultiTensor): The first tensor.
            tensor2 (_MultiTensor): The second tensor.
            equal_nan (bool): If :obj:`True`, then two :obj:`NaN`s will be
                considered equal (default: :obj:`False`).

        Returns:
            bool: Whether the given two tensors are close or not.
        """
        if tensor1.shape != tensor2.shape:
            return False
        if tensor1.values.shape != tensor2.values.shape:
            return False
        if not torch.allclose(tensor1.values, tensor2.values,
                              equal_nan=equal_nan):
            return False
        if tensor1.offset.shape != tensor2.offset.shape:
            return False
        if not torch.allclose(tensor1.offset, tensor2.offset,
                              equal_nan=equal_nan):
            return False
        return True

    # Indexing Functions ######################################################

    def __getitem__(
        self,
        index: Any,
    ) -> _MultiTensor | Tensor:
        if isinstance(index, tuple):
            # index[0] for row indexing, index[1] for column indexing
            assert len(index) == 2
            if all(isinstance(idx, int) for idx in index):
                # Return type: torch.Tensor
                return self._get_value(index[0], index[1])
            else:
                # Return type: self.__class__
                out = self
                for dim, idx in enumerate(index):
                    out = out.select(idx, dim)
                return out
        else:
            # Return type: self.__class__
            return self.select(index, dim=0)

    def _get_value(self, row: int, col: int) -> Tensor:
        raise NotImplementedError

    def index_select(self, index: Tensor, dim: int) -> _MultiTensor:
        """Returns a :class:`_MultiTensor` which indexes the input
        :class:`_MultiTensor` along the specified dimension.

        Args:
            index (Tensor): A 1-D tensor of indices to select.
            dim (int): The dimension to index in.
        """
        dim = self._normalize_dim(dim)
        idx = self._normalize_index(index, dim=dim)
        if dim == 0:
            return self._row_index_select(idx)
        elif dim == 1:
            return self._col_index_select(idx)
        assert False, "Should not reach here."

    def _row_index_select(self, index: Tensor) -> _MultiTensor:
        raise NotImplementedError

    def _col_index_select(self, index: Tensor) -> _MultiTensor:
        raise NotImplementedError

    def _slice(self, index: slice, dim: int) -> _MultiTensor:
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
            # For narrow, we don't need out-of-bound checks since something
            # like mat[100:110] (with mat.size() < 100) is perfectly valid and
            # should return an empty tensor.
            start_idx: int = self._normalize_index(
                index.start or 0,
                dim=dim,
                check_out_of_bounds=False,
            )
            end_idx: int = self._normalize_index(
                index.stop if index.stop is not None else num_data,
                dim=dim,
                check_out_of_bounds=False,
            )
            return self.narrow(
                dim=dim,
                start=start_idx,
                length=end_idx - start_idx,
            )

    def narrow(self, dim: int, start: int, length: int) -> _MultiTensor:
        """Narrow the tensor along the given dimension.

        Args:
            dim (int): The dimension along which to narrow.
            start (int): The starting index.
            length (int): The length of the slice.
        """
        assert start >= 0
        dim = self._normalize_dim(dim)
        num_data = self.num_rows if dim == 0 else self.num_cols
        if start == 0 and start + length >= num_data:
            return self
        elif length <= 0:
            return self._empty(dim)
        elif dim == 0:
            return self._row_narrow(start, length)
        elif dim == 1:
            return self._col_narrow(start, length)
        assert False, "Should not reach here."

    def _row_narrow(self, start: int, length: int) -> _MultiTensor:
        raise NotImplementedError

    def _col_narrow(self, start: int, length: int) -> _MultiTensor:
        raise NotImplementedError

    def _empty(self, dim: int) -> _MultiTensor:
        raise NotImplementedError

    def select(
        self,
        index: int | Tensor | Sequence[int] | slice | range,
        dim: int,
    ) -> _MultiTensor:
        """Returns a new :class:`MultiEmbeddingTensor` which indexes the input
        :class:`MultiEmbeddingTensor` along the specified dimension.

        Args:
            index (Union[int, Tensor, Sequence[int], slice, range]): A row or
                column index of the tensor to select.
            dim (int): The dimension to index in. If :obj:`dim=0`, it selects
                rows. If :obj:`dim=1`, it selects columns.
        """
        dim = self._normalize_dim(dim)
        if isinstance(index, int):
            return self._single_index_select(index, dim=dim)
        elif isinstance(index, slice):
            return self._slice(index, dim=dim)
        elif isinstance(index, Tensor) and index.ndim == 1:
            return self.index_select(index, dim=dim)
        # TODO: Don't materialize range, and instead pass it to PyTorch tensor
        # as index directly to avoid unnecessary memory usage.
        elif isinstance(index, (list, range)):
            return self.index_select(
                torch.tensor(index, dtype=torch.long, device=self.device),
                dim=dim,
            )
        assert False, "Should not reach here."

    def _single_index_select(self, index: int, dim: int) -> _MultiTensor:
        raise NotImplementedError

    def fillna_col(
        self,
        col_index: int,
        fill_value: int | float | Tensor,
    ):
        """Fill the :obj:`index`-th column in :obj:`MultiTensor` with
        fill_value in-place.

        Args:
            col_index (int): A column index of the tensor to select.
            fill_value (Union[int, float, Tensor]): Scalar values to replace
                NaNs.
        """
        raise NotImplementedError


def _batched_arange(count: Tensor) -> tuple[Tensor, Tensor]:
    r"""Fast implementation of batched version of :meth:`torch.arange`.
    It essentially does the following.

    .. code-block:: python

        batch = torch.cat([torch.full((c,), i) for i, c in enumerate(count)])
        arange = torch.cat([torch.arange(c) for c in count])

    Args:
        count (Tensor): The count vectors.

    Returns:
        batch (Tensor): batch[i] indicates the batch index of
            _batched_arange[i]
        arange (Tensor): batched version of arange

    Example:
        >>> count = torch.tensor([3, 2, 4])
        >>> batch, arange = _batched_arange(count)
        >>> batch
        tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> arange
        tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
    """
    ptr = count.new_zeros(count.numel() + 1)
    torch.cumsum(count, dim=0, out=ptr[1:])

    batch = torch.arange(count.numel(), device=count.device).repeat_interleave(
        count, output_size=ptr[-1])  # type: ignore[call-overload]

    arange = torch.arange(batch.numel(), device=count.device)
    arange -= ptr[batch]

    return batch, arange
