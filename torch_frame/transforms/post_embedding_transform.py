from torch import Tensor
from torch.nn import Module

from torch_frame.transforms import BaseTransform


class PostEmbeddingTransform(BaseTransform, Module):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
