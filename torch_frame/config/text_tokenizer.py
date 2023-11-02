from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from torch import Tensor


@dataclass
class TextTokenizerConfig:
    r"""Text tokenizer that maps a list of strings/sentences into a
    dictionary of :class:`MultiNestedTensor`.

    Args:
        text_tokenizer (callable): A callable text tokenizer that takes a list
            of strings as input and output the dictionary of
            :class:`MultiNestedTensor`.
        batch_size (int, optional): Batch size to use when encoding the
            sentences. If set to :obj:`None`, the text embeddings will
            be obtained in a full-batch manner. (default: :obj:`None`)

    """
    text_tokenizer: Callable[[List[str]], List[Dict[str, Tensor]]]
    batch_size: Optional[int] = None
