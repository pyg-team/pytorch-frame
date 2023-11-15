import numpy as np
import pandas as pd
import torch

from torch_frame.data.mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalSequenceTensorMapper,
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


def test_timestamp_tensor_mapper():
    ser = pd.Series(['A', 'B', None, 'C', 'B'])
    expected = torch.tensor([1, 0, -1, -1, 0])

    mapper = CategoricalTensorMapper(['B', 'A'])

    out = mapper.forward(ser)
    assert out.dtype == torch.long
    assert torch.equal(out, expected)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(out, pd.Series(['A', 'B', None, None, 'B']))


def test_multicategorical_tensor_mapper():
    ser = pd.Series(['A,B', 'B', '', 'C', 'B,C', None])
    expected_values = torch.tensor([1, 0, 0, 0, -1])
    expected_boundaries = torch.tensor([0, 2, 3, 3, 3, 4, 5])
    mapper = MultiCategoricalTensorMapper(['B', 'A'], sep=",")

    tensor = mapper.forward(ser)
    values = tensor.values
    offset = tensor.offset
    assert values.dtype == torch.long
    assert torch.equal(
        values[expected_boundaries[0]:expected_boundaries[1]].sort().values,
        torch.tensor([0, 1]))
    assert torch.equal(values[expected_boundaries[1]:],
                       expected_values[expected_boundaries[1]:])
    assert torch.equal(offset, expected_boundaries)

    out = mapper.backward(tensor)
    assert out.values[0] == 'A,B' or out.values[0] == 'B,A'
    assert out.values[1] == 'B'
    assert out.values[2] == ''
    assert out.values[3] == ''
    assert out.values[4] == 'B'
    assert out.values[5] == ''


def test_numerical_sequence_tensor_mapper():
    ser = pd.Series([[0.1, 0.5], [0.3], [], [0.2, np.nan], None, np.nan])
    expected_values = torch.tensor([0.1, 0.5, 0.3, 0.2, torch.nan],
                                   dtype=torch.float32)
    expected_offset = torch.tensor([0, 2, 3, 3, 5, 5, 5])
    mapper = NumericalSequenceTensorMapper()

    tensor = mapper.forward(ser)
    values = tensor.values
    offset = tensor.offset
    assert values.dtype == torch.float32
    assert ((values == expected_values) |
            (torch.isnan(values) & torch.isnan(expected_values))).all()
    assert torch.equal(offset, expected_offset)

    out = mapper.backward(tensor)
    pd.testing.assert_series_equal(
        out, pd.Series([[0.1, 0.5], [0.3], None, [0.2, np.nan], None, None]))


def test_text_embedding_tensor_mapper():
    out_channels = 10
    num_sentences = 20
    ser = pd.Series(["Hello world!"] * (num_sentences // 2) +
                    ["I love torch-frame"] * (num_sentences // 2) + [0.1])
    mapper = TextEmbeddingTensorMapper(HashTextEmbedder(out_channels),
                                       batch_size=8)
    emb = mapper.forward(ser)
    assert emb.shape == (num_sentences + 1, out_channels)
    mapper.batch_size = None
    emb2 = mapper.forward(ser)
    assert torch.allclose(emb, emb2)
