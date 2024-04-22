import torch
from torch import Tensor


def cross_entropy_with_logits(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    r"""Cross Entropy Loss for Soft Multi-classification Labels."""
    probs = torch.softmax(logits, dim=1)
    return -(targets * probs.log()).sum(1).mean()
