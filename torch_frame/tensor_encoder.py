from abc import ABC
from typing import Dict


import torch_frame
from torch_frame import TensorFrame
from torch_frame.typing import DataFrame

class TensorEncoder(ABC):
    r"""Base class for tensor encoder that transforms input DataFrame into
    TensorFrame

    Args:
        stypes: (Dict[str, torch_frame.stype]): A dictionary that maps each
            column in the data frame to a semantic type.    
    """
    def __init__(
        self,
        stypes: Dict[str, torch_frame.stype],
    ):
        self.stypes = stypes
    
    def encode(self, df: DataFrame) -> TensorFrame:
        raise NotImplementedError()
