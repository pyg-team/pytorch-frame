from typing import List, Union

import pandas as pd
from torch import Tensor

Series = pd.Series
DataFrame = pd.DataFrame

ColumnSelectType = Union[str, List[str]]
IndexSelectType = Union[int, List[int], range, slice, Tensor]
