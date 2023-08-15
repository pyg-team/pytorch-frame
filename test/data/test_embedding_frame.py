from torch_frame import EmbeddingFrame
import torch


def test_embedding_frame_basics():
    x = torch.randn(size=(10, 6, 8))
    col_names = ['a', 'b', 'c', 'prompt']
    col_index = torch.Tensor([0, 1, 2, 3, 6])
    ef = EmbeddingFrame(x=x, col_names=col_names, col_index=col_index)
    assert len(ef) == 10
    assert ef.num_cols == 4
    assert ef.dim == 8
