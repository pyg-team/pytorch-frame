import pytest

from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import FTTransformer


@pytest.mark.parametrize('batch_size', [0, 5])
def test_ft_transformer(batch_size):
    channels = 8
    out_channels = 1
    num_layers = 3
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame[:batch_size]
    # Feature-based embeddings
    model = FTTransformer(
        channels=channels,
        out_channels=out_channels,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    model.reset_parameters()
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)


def test_ft_transformer_custom_hyperparameters():
    batch_size = 5
    channels = 8
    out_channels = 1
    num_layers = 3

    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame[:batch_size]
    feedforward_channels = 16
    nhead = 4
    dropout = 0.1
    activation = 'gelu'
    model = FTTransformer(
        channels=channels,
        out_channels=out_channels,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        feedforward_channels=feedforward_channels,
        nhead=nhead,
        dropout=dropout,
        activation=activation,
    )
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)
    layer = model.backbone.transformer.layers[0]
    assert layer.linear1.out_features == feedforward_channels
    assert layer.dropout.p == dropout
    assert layer.self_attn.num_heads == nhead
