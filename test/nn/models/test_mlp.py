import pytest

from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import MLP


@pytest.mark.parametrize('batch_size', [0, 5])
@pytest.mark.parametrize('normalization', ["layer_norm", "batch_norm"])
def test_mlp(batch_size, normalization):
    channels = 8
    out_channels = 1
    num_layers = 3
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame[:batch_size]
    # Feature-based embeddings
    model = MLP(
        channels=channels,
        out_channels=out_channels,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        normalization=normalization,
    )
    model.reset_parameters()
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)
