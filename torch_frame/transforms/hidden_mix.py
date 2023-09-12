import numpy as np
import torch
from torch import Tensor

from torch_frame.transforms import PostEmbeddingTransform


class HiddenMix(PostEmbeddingTransform):
    r""" Implementation of Hidden-Mix, which is a variation of mixup,
        introduced in https://arxiv.org/abs/1710.09412.

        Hidden-Mix is applied to the representation after the embedding
        layer and the labels. It exchanges some representation elements
        of two samples. The scala coefficient shuffle_rates is sampled
        from the Beta distribution, Beta(beta, beta).

        Args:
            beta (float): The shape parameter of Beta distribution used
                to sample shuffle rates of Hidden-Mix. (default 0.5)
    """
    def __init__(self, beta: float = 0.5):
        self.beta = 0.5
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        B, _, channels = x.shape
        beta = self.beta
        shuffle_rates = np.random.beta(beta, beta, size=(B, 1))
        dim_masks = np.random.random(size=(B, channels)) < shuffle_rates
        dim_masks = torch.from_numpy(dim_masks).to(x.device)

        shuffled_sample_ids = np.random.permutation(B)

        x_shuffled = x[shuffled_sample_ids]
        dim_masks = dim_masks.unsqueeze(1)
        x_mixup = dim_masks * x + ~dim_masks * x_shuffled
        return x_mixup
