from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Embedding


class HashTextEmbedder:
    r"""A hash-based light-weight text embedder for testing purposes.
    It hashes each sentence into an index modulo :obj:`num_hash` and then
    uses :class:`torch.nn.Embedding` to look up the index to obtain the
    sentence embedding.

    Args:
        out_channels (int): The output dimensionality
        num_hash_bins (int): Number of hash bins to use.
            (default: :obj:`64`)
        device (torch.device, optional): The device to put :class:`Embedding`
            module. (default: :obj:`None`)
    """
    def __init__(
        self,
        out_channels: int,
        num_hash_bins: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.out_channels = out_channels
        self.num_hash_bins = num_hash_bins
        self.device = device
        self.embedding = Embedding(num_hash_bins, out_channels).to(device)

    def __call__(self, sentences: list[str]) -> Tensor:
        idx = torch.tensor([hash(s) % self.num_hash_bins for s in sentences],
                           device=self.device)
        return self.embedding(idx).detach()
