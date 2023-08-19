import torch_frame


def test_stype():
    assert len(torch_frame.Stype) == 2
    assert torch_frame.numerical == torch_frame.Stype('numerical')
    assert torch_frame.categorical == torch_frame.Stype('categorical')
