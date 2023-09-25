import torch_frame


def test_stype():
    assert len(torch_frame.stype) == 3
    assert torch_frame.numerical == torch_frame.stype('numerical')
    assert torch_frame.categorical == torch_frame.stype('categorical')
    assert torch_frame.text_embedded == torch_frame.stype('text_embedded')
