import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import Tensor

import torch_frame
from torch_frame.typing import IndexSelectType


@dataclass(repr=False)
class TensorFrame:
    r"""TensorFrame holds a :pytorch:`PyTorch` tensor for each table column.
    Table columns are first organized into their semantic types (e.g.,
    categorical, numerical) and then converted into their tensor
    representation, which is stored as :obj:`x_dict`. For instance,
    :obj:`x_dict[stype.numerical]` stores a concatenated :pytorch:`PyTorch`
    tensor for all numerical features, where 0th/1st dim represents the
    row/column in the original DataFrame, respectively.

    :obj:`col_names_dict` stores column names of :obj:`x_dict`. For example,
    :obj:`col_names_dict[stype.numerical][i]` stores the column name of
    :obj:`x_dict[stype.numerical][:,i]`.

    Additionally, TensorFrame can store the target values in :obj:`y`.
    """
    x_dict: Dict[torch_frame.stype, Tensor]
    col_names_dict: Dict[torch_frame.stype, List[str]]
    y: Optional[Tensor] = None

    def __post_init__(self):
        num_rows = self.num_rows
        for stype_name, x in self.x_dict.items():
            if x.dim() < 2:
                raise ValueError(
                    f"x_dict['{stype_name}'] must be at least 2-dimensional")
            num_cols = len(self.col_names_dict[stype_name])
            if num_cols != x.size(1):
                raise ValueError(
                    f"The expected number of columns for {stype_name} feature "
                    f"is {num_cols}, which does not align with the column "
                    f"dimensionality of x_dict[{stype_name}] (got "
                    f"{x.size(1)})")
            if x.size(0) != num_rows:
                raise ValueError(
                    f"The length of elements in x_dict are not aligned, got "
                    f"{x.size(0)} but expected {num_rows}.")
        if self.y is not None:
            if len(self.y) != num_rows:
                raise ValueError(
                    f"The length of y is {len(self.y)}, which is not aligned "
                    f"with the number of rows ({num_rows}).")

    @property
    def num_rows(self) -> int:
        return len(next(iter(self.x_dict.values())))

    def __repr__(self) -> str:
        stype_repr = '\n'.join([
            f'  {stype.value} ({len(col_names)}): {col_names},'
            for stype, col_names in self.col_names_dict.items()
        ])

        return (f'{self.__class__.__name__}(\n'
                f'  num_rows={self.num_rows},\n'
                f'{stype_repr}\n'
                f')')

    def __getitem__(self, index: IndexSelectType) -> 'TensorFrame':
        if isinstance(index, int):
            index = [index]

        out = copy.copy(self)

        out.x_dict = {stype: x[index] for stype, x in out.x_dict.items()}
        out.col_names_dict = copy.copy(out.col_names_dict)
        if out.y is not None:
            out.y = out.y[index]

        return out
