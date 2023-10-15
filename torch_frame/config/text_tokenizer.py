from dataclasses import dataclass
from typing import Callable, List, Optional

from torch.nested import nested_tensor


@dataclass
class TextTokenizerConfig:
    text_tokenizer: Callable[[List[str]], nested_tensor]
    batch_size: Optional[int] = None
