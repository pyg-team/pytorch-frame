import torch_frame


def test_stype():
    assert len(torch_frame.stype) == 5
    assert torch_frame.numerical == torch_frame.stype('numerical')
    assert not torch_frame.numerical.is_text_stype
    assert torch_frame.categorical == torch_frame.stype('categorical')
    assert not torch_frame.categorical.is_text_stype
    assert torch_frame.multicategorical == torch_frame.stype(
        'multicategorical')
    assert not torch_frame.multicategorical.is_text_stype
    assert torch_frame.sequence_numerical == torch_frame.stype(
        'sequence_numerical')
    assert not torch_frame.sequence_numerical.is_text_stype
    assert torch_frame.text_embedded == torch_frame.stype('text_embedded')
    assert torch_frame.text_embedded.is_text_stype
