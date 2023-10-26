import pandas as pd
import torch

from torch_frame.data.mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    TextEmbeddingTensorMapper,
)
from torch_frame.testing.text_embedder import HashTextEmbedder


def test_numerical_tensor_mapper():
    ser = pd.Series([0.0, 10.0, float('NaN'), 30.0])
    expected = torch.tensor([0.0, 10.0, float('NaN'), 30.0])

    mapper = NumericalTensorMapper()

    out = mapper.forward(ser)
    assert out.dtype == torch.float
    assert torch.equal(out.isnan(), expected.isnan())
    assert torch.equal(out.nan_to_num(), expected.nan_to_num())

    out = mapper.backward(out)
    pd.testing.assert_series_equal(out, ser, check_dtype=False)


def test_categorical_tensor_mapper():
    ser = pd.Series(['A', 'B', None, 'C', 'B'])
    expected = torch.tensor([1, 0, -1, -1, 0])

    mapper = CategoricalTensorMapper(['B', 'A'])

    out = mapper.forward(ser)
    assert out.dtype == torch.long
    assert torch.equal(out, expected)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(out, pd.Series(['A', 'B', None, None, 'B']))


def test_multicategorical_tensor_mapper():
    ser = pd.Series(['A,B', 'B', '', 'C', 'B,C'])
    expected_values = torch.tensor([1, 0, 0, -1, 0, -1])
    expected_boundaries = torch.tensor([0, 2, 3, 3, 4, 6])
    mapper = MultiCategoricalTensorMapper(['B', 'A'], sep=",")

    tensor = mapper.forward(ser)
    values = tensor.values
    offset = tensor.offset
    assert values.dtype == torch.long
    assert torch.equal(values, expected_values)
    assert torch.equal(offset, expected_boundaries)

    out = mapper.backward(tensor)
    pd.testing.assert_series_equal(out, pd.Series(['A,B', 'B', '', '', 'B']))


def test_text_embedding_tensor_mapper():
    out_channels = 10
    num_sentences = 20
    ser = pd.Series(["Hello world!"] * (num_sentences // 2) +
                    ["I love torch-frame"] * (num_sentences // 2))
    mapper = TextEmbeddingTensorMapper(HashTextEmbedder(out_channels),
                                       batch_size=8)
    emb = mapper.forward(ser)
    assert emb.shape == (num_sentences, out_channels)
    mapper.batch_size = None
    emb2 = mapper.forward(ser)
    assert torch.allclose(emb, emb2)
