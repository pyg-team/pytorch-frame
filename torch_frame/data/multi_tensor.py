import copy
from typing import Any, Callable, Dict, Tuple, Union

import torch
from torch import Tensor


class _MultiTensor:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        values: Tensor,
        offset: Tensor,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.values = values
        self.offset = offset
        self.validate()

    def validate(self):
        r"""Validates the :class:`_MultiTensor` object."""

    def to_dict(self) -> Dict[str, Any]:
        r"""Serialize the object into a dictionary."""
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "values": self.values,
            "offset": self.offset,
        }

    def __setitem__(self, index: Any, values: Any) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} object does not support setting "
            "values. It should be used for read-only.")

    def __repr__(self) -> str:
        return " ".join([
            f"{self.__class__.__name__}(num_rows={self.num_rows},",
            f"num_cols={self.num_cols},",
            f"device='{self.device}')",
        ])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.num_rows, self.num_cols, -1)

    @property
    def ndim(self) -> int:
        return 3

    def size(self, dim: int) -> int:
        dim = self._normalize_dim(dim)
        if dim == 0:
            return self.num_rows
        elif dim == 1:
            return self.num_cols

    def dim(self) -> int:
        return self.ndim

    def __len__(self):
        return self.num_rows

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def clone(self) -> "_MultiTensor":
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

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> "_MultiTensor":
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
        index: Union[int, Tensor],
        dim: int,
        is_slice_end: bool = False,
    ) -> Union[int, Tensor]:
        """Helper function to map negative indices to positive indices and
        raise :obj:`IndexError` when necessary.

        Args:
            index: Union[int, Tensor]: Input :obj:`index` with potentially
                negative elements.
            dim (int): Dimension to be indexed.
            is_slice_end (bool): Whether a given index (int) is slice end or
                not. If :obj:`True`, we have more lenient :obj:`IndexError`.
                (default: :obj:`False`)
        """
        dim = self._normalize_dim(dim)
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
    def allclose(
        cls,
        tensor1: "_MultiTensor",
        tensor2: "_MultiTensor",
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
