from dataclasses import dataclass

from torch.nn import Module


@dataclass
class ModelConfig:
    r"""Learnable model that maps a single-column :class:`TensorData` object
    into row embeddings.

    Args:
        model (Module): A :class:`~torch.nn.Module` that takes a
            :obj:`TensorData` of shape :obj:`[batch_size, 1, *]` as input and
            outputs embeddings of shape :obj:`[batch_size, 1, out_channels]`.
        out_channels (int): Model output channels.
    """
    model: Module
    out_channels: int
