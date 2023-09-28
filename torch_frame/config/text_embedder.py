from dataclasses import dataclass
from typing import Callable, List, Optional

from torch import Tensor


@dataclass
class TextEmbedderConfig:
    # Text embedder model that maps a list of strings/sentences into PyTorch
    # Tensor embeddings.
    text_embedder: Callable[[List[str]], Tensor]
    # Batch size to use when encoding the sentences. It is recommended to set
    # it to a reasonable value when one uses a heavy text embedding model
    # (e.g., Transformer) on GPU. If set to :obj:`None`, the text embeddings
    # will be obtained in a full-batch manner.
    batch_size: Optional[int] = None
