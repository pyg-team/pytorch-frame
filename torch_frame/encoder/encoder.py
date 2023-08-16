from abc import ABC, abstractmethod
from typing import Dict

from torch_frame import TensorFrame, stype
from torch_frame.typing import DataFrame


class TensorEncoder(ABC):
    r"""Base class for tensor encoder that transforms input DataFrame into
    TensorFrame.

    Args:
        col2stype: (Dict[str, stype]): A dictionary that maps column name in
        DataFrame to its stype.
    """
    def __init__(
        self,
        col2stype: Dict[str, stype],
    ):
        self.col2stype = col2stype

    @abstractmethod
    def to_tensor(self, df: DataFrame) -> TensorFrame:
        r"""Convert DataFrame into TensorFrame"""
        raise NotImplementedError
