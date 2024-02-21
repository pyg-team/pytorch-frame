from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor


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
