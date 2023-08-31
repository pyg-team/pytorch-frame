import torch

from torch_frame.nn import FTTransformerConvs


def test_ft_transformer_convs():
    x = torch.randn(size=(10, 3, 8))
    conv = FTTransformerConvs(channels=8, num_layers=3)
    x, x_cls = conv(x)
    # The first added column corresponds to CLS token.
    assert x.shape == (10, 3, 8)
    assert x_cls.shape == (10, 8)
