from dataclasses import dataclass
from typing import Callable, List, Optional

from torch_frame.typing import TextTokenizationOutput


@dataclass
class TextTokenizerConfig:
    r"""Text tokenizer that maps a list of strings/sentences into a
    dictionary of :class:`MultiNestedTensor`.

    Args:
        text_tokenizer (callable): A callable text tokenizer that takes a
            list of strings as input and outputs a list of dictionaries.
            Each dictionary contains keys that are arguments to the text
            encoder model and values are corresponding tensors such as
            tokens and attention masks.
        batch_size (int, optional): Batch size to use when tokenizing the
            sentences. If set to :obj:`None`, the text embeddings will
            be obtained in a full-batch manner. (default: :obj:`None`)

    """
    text_tokenizer: Callable[[List[str]], TextTokenizationOutput]
    batch_size: Optional[int] = None
