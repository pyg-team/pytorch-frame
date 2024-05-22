from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from PIL import Image
from torch import Tensor


class ImageEmbedder(ABC):
    r"""Parent class for the :obj:`image_embedder` of
    :class:`ImageEmbedderConfig`. This class first retrieves images based
    on given paths stored in the data frame and then embeds retrieved images
    into tensor. Users are responsible for implementing :meth:`forward_embed`
    which takes a list of images and returns embeddings tensor. User can also
    override :meth:`forward_retrieve` which takes the paths to images and
    return a list of :obj:`PIL.Image.Image`.
    """
    def __init__(self, *args, **kwargs):
        pass

    def forward_retrieve(self, path_to_images: list[str]) -> list[Image.Image]:
        r"""Retrieval function that reads a list of images from
        a list of file paths with the :obj:`RGB` mode.
        """
        images: list[Image.Image] = []
        for path_to_image in path_to_images:
            image = Image.open(path_to_image)
            images.append(image.copy())
            image.close()
        images = [image.convert('RGB') for image in images]
        return images

    @abstractmethod
    def forward_embed(self, images: list[Image.Image]) -> Tensor:
        r"""Embedding function that takes a list of images and returns
        an embedding tensor.
        """
        raise NotImplementedError

    def __call__(self, path_to_images: list[str]) -> Tensor:
        images = self.forward_retrieve(path_to_images)
        return self.forward_embed(images)


@dataclass
class ImageEmbedderConfig:
    r"""Image embedder model that maps a list of images into PyTorch
    Tensor embeddings.

    Args:
        image_embedder (callable): A callable image embedder that takes a
            list of path to images as input and outputs the PyTorch Tensor
            embeddings for that list of images. Usually it contains a retriever
            to load image files and then a embedder converting images to
            embeddings.
        batch_size (int, optional): Batch size to use when encoding the
            images. If set to :obj:`None`, the image embeddings will
            be obtained in a full-batch manner. (default: :obj:`None`)

    """
    image_embedder: Callable[[list[str]], Tensor]
    # Batch size to use when encoding the images. It is recommended to set
    # it to a reasonable value when one uses a heavy image embedding model
    # (e.g., ViT) on GPU. If set to :obj:`None`, the image embeddings
    # will be obtained in a full-batch manner.
    batch_size: int | None = None
