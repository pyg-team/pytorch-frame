from __future__ import annotations

import torch
from PIL import Image
from torch import Tensor


class RandomImageEmbedder:
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
        self.out_channels = out_channels

    def forward_retrieve(self, path_to_images: list[str]) -> list[Image]:
        images: list[Image] = []
        for path_to_image in path_to_images:
            image = Image.open(path_to_image)
            images.append(image.copy())
            image.close()
        images = [image.convert('RGB') for image in images]
        return images

    def forward_embed(self, images: list[Image]) -> Tensor:
        embeddings = []
        for _ in images:
            embeddings.append(torch.rand(1, self.out_channels))
        return torch.cat(embeddings, dim=0)

    def __call__(self, path_to_images: list[str]) -> Tensor:
        images = self.forward_retrieve(path_to_images)
        return self.forward_embed(images)
