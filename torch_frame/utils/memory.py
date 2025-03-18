from torch import Tensor

from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.typing import TensorData


def num_bytes(data: TensorData) -> int:
    r"""Returns the number of bytes the tensor data consumes.

    Args:
        data (TensorData): The tensor data.
    """
    if isinstance(data, Tensor):
        return data.element_size() * data.numel()
    if isinstance(data, (MultiNestedTensor, MultiEmbeddingTensor)):
        return (data.values.element_size() * data.values.numel() +
                data.offset.element_size() * data.offset.numel())
    if isinstance(data, dict):
        return sum([num_bytes(value) for value in data.values()])

    raise NotImplementedError(f"'num_bytes' not implemented for "
                              f"'{type(data)}'")
