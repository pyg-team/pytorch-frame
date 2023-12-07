from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor


@dataclass
class TextEmbedderConfig:
    r"""Text embedder model that maps a list of strings/sentences into PyTorch
    Tensor embeddings.

    Args:
        text_embedder (callable): A callable text embedder that takes a list
            of strings as input and outputs the PyTorch Tensor embeddings for
            that list of strings.
        batch_size (int, optional): Batch size to use when encoding the
            sentences. If set to :obj:`None`, the text embeddings will
            be obtained in a full-batch manner. (default: :obj:`None`)

    """
    text_embedder: Callable[[list[str]], Tensor]
    # Batch size to use when encoding the sentences. It is recommended to set
    # it to a reasonable value when one uses a heavy text embedding model
    # (e.g., Transformer) on GPU. If set to :obj:`None`, the text embeddings
    # will be obtained in a full-batch manner.
    batch_size: int | None = None
