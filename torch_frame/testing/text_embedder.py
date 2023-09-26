from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Embedding


class HashTextEmbedder:
    r"""A fake light-weight has-based text embedder for testing purpose.
    It hashes each sentence into an index modulo :obj:`num_hash` and then
    uses :class:`torch.nn.Embedding` to look up the index to obtain the
    sentence embedding.

    Args:
        out_channels (int): The output dimensionality
        num_hash (int): Number of hashes to use.
        device (torch.device, optional): The device to put Embedding module.
    """
    def __init__(self, out_channels: int, num_hash: int = 64,
                 device: Optional[torch.device] = None):
        self.out_channels = out_channels
        self.num_hash = num_hash
        self.device = device
        self.embedding = Embedding(num_hash, out_channels).to(device)

    def __call__(self, sentences: List[str]) -> Tensor:
        idx = torch.LongTensor([hash(s) % self.num_hash for s in sentences],
                               device=self.device)
        return self.embedding(idx)
