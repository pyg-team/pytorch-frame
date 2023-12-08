from dataclasses import dataclass
from typing import Callable, Dict

from torch import Tensor

from torch_frame.data import MultiNestedTensor


@dataclass
class TextModelConfig:
    r"""Text model that maps a dictionary of :class:`MultiNestedTensor`
    into embeddings of each row.

    Args:
        model (callable): A callable text model that takes a dictionary
            of :obj:`MultiNestedTensor` as input and outputs embeddings
            for each row in the :class:`MultiNestedTensor`. Input
            :obj:`MultiNestedTensor` has the shape `[batch_size, 1, *]`
            and output embeddings should have shape
            `[batch_size, 1, text_model_out_channels]`.
        out_channels (int): Text model output channels.

    """
    model: Callable[[Dict[str, MultiNestedTensor]], Tensor]
    out_channels: int
