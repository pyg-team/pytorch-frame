from dataclasses import dataclass
from torch_frame import stype
from typing import Dict, List, Optional
from torch import Tensor


@dataclass
class TensorFrame:
    r"""torch_frame.Preprocessor converts input Dataframe into TensorFrame,
    which is input into torch_frame.Encoder.
    """
    # stype to 2-dim tensor mapping
    x_dict: Dict[stype, Tensor]
    # stype to column names mapping
    col_names_dict: Dict[stype, List[str]]
    # target values.
    y: Optional[Tensor] = None

    def __post_init__(self):
        num_rows = self.num_rows
        for stype_name, x in self.x_dict.items():
            if x.ndim != 2:
                raise ValueError(
                    f"x_dict['{stype_name}'] is not 2-dim tensor.")
            num_cols = len(self.col_names_dict[stype_name])
            if num_cols != x.size(1):
                raise ValueError(
                    f"The length of col_names['{stype_name}'] is "
                    f"{num_cols}, which does not match the dimensionality "
                    f"of x_dict['{stype_name}'] ({x.size(1)})")
            if x.size(0) != num_rows:
                raise ValueError(
                    "The length of elements in x_dict are not aligned.")
        if self.y is not None:
            if len(self.y) != num_rows:
                raise ValueError(
                    f"The length of y is {len(self.y)}, which is not aligned "
                    f"with the number of rows ({num_rows}).")

    @property
    def num_rows(self):
        return len(next(iter(self.x_dict.values())))
