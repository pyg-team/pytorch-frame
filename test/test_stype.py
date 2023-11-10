import pytest

import torch_frame


def test_stype():
    assert len(torch_frame.stype) == 7
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
    assert torch_frame.text_tokenized == torch_frame.stype('text_tokenized')
    assert torch_frame.text_tokenized.is_text_stype
    assert torch_frame.embedding == torch_frame.stype('embedding')
    assert torch_frame.embedding.use_multi_embedding_tensor
