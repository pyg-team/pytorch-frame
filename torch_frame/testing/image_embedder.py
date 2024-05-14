from __future__ import annotations

import torch
from PIL import Image
from torch import Tensor

from torch_frame.config.image_embedder import ImageEmbedder


class RandomImageEmbedder(ImageEmbedder):
    r"""A random-based light-weight image embedder for testing
    purposes. It opens each image and generates a random embedding
    with :obj:`out_channels` embedding size.

    Args:
        out_channels (int): The output dimensionality
    """
    def __init__(
        self,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward_embed(self, images: list[Image.Image]) -> Tensor:
        return torch.rand(len(images), self.out_channels)
