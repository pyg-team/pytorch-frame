from dataclasses import dataclass
from typing import Callable, List, Optional

from torch import Tensor


@dataclass
class TextEmbedderConfig:
    text_embedder: Callable[[List[str]], Tensor]
    batch_size: Optional[int]
