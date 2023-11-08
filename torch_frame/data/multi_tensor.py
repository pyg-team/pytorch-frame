import copy
from typing import Any, Callable, Tuple

import torch
from torch import Tensor


class _MultiTensor:
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

    def __setitem__(self, index: Any, values: Any) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} object does not support setting "
            "values. It should be used for read-only.")

    def __repr__(self) -> str:
        return ' '.join([
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

    def clone(self) -> '_MultiTensor':
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

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> '_MultiTensor':
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
