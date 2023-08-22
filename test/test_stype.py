import torch_frame


def test_stype():
    assert len(torch_frame.stype) == 2
    assert torch_frame.numerical == torch_frame.stype('numerical')
    assert torch_frame.categorical == torch_frame.stype('categorical')
