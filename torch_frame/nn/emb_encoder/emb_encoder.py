from abc import ABC

from torch import Module
from torch_frame import TensorFrame, EmbeddingFrame


class EmbeddingEncoder(Module, ABC):
    r"""Base class for embedding encoder that transforms TensorFrame into
    EmbeddingFrame"""
    def forward(self, tf: TensorFrame) -> EmbeddingFrame:
        raise NotImplementedError()
