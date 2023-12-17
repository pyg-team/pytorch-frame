import torch
from torch import Tensor

from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.typing import TensorData


def eq(td1: TensorData, td2: TensorData):
    r"""Function to compare two :obj:`TensorData`. Note that if the
    two input :obj:`TensorData` has to be the same class.

    Args:
        td1 (TensorData): Input :obj:`TensorData` to be compared
        td2 (TensorData): Input :obj:`TensorData` to be compared
    """
    if isinstance(td1, Tensor):
        return isinstance(td2, Tensor) and torch.all(
            torch.eq(td1, td2) | torch.isnan(td1) & torch.isnan(td2))
    elif isinstance(td1, MultiNestedTensor):
        return isinstance(td2, MultiNestedTensor) and td1 == td2
    elif isinstance(td1, MultiEmbeddingTensor):
        return isinstance(td2, MultiEmbeddingTensor) and td1 == td2
    elif isinstance(td1, dict):
        if not isinstance(td2, dict):
            return False
        for key in td1:
            if not (key in td2 and td1[key] == td2[key]):
                return False
        return True
