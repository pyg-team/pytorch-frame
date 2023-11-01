import torch_frame


def test_stype():
    assert len(torch_frame.stype) == 6
    assert torch_frame.numerical == torch_frame.stype('numerical')
    assert not torch_frame.numerical.is_text_stype
    assert torch_frame.categorical == torch_frame.stype('categorical')
    assert not torch_frame.categorical.is_text_stype
    assert torch_frame.text_embedded == torch_frame.stype('text_embedded')
    assert torch_frame.text_embedded.is_text_stype
    assert torch_frame.text_tokenized == torch_frame.stype('text_tokenized')
    assert torch_frame.text_tokenized.is_text_stype
