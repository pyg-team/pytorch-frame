import torch

from torch_frame.nn import TabNet


def test_tabnet_encoder():
    x = torch.randn(size=(10, 8))
    tabnet_encoder = TabNet(8)
    out, regularization = tabnet_encoder(x)
    assert out.shape == (10, 8)
