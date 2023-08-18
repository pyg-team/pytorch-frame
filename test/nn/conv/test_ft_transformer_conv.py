import torch

from torch_frame.nn import CLSDecoder, FTTransformerConv


def test_ft_transformer_conv():
    x = torch.randn(size=(10, 3, 8))
    conv = FTTransformerConv(channels=8, num_layers=3)
    cls_decoder = CLSDecoder()
    x = conv(x)
    # The first added column corresponds to CLS token.
    assert x.shape == (10, 4, 8)
    x = cls_decoder(x)
    assert x.shape == (10, 8)
