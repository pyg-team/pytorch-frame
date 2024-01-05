import pytest

from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import EmbeddingEncoder, LinearEncoder, Trompt


@pytest.mark.parametrize('batch_size', [0, 5])
@pytest.mark.parametrize('stype_encoder_dicts', [
    [
        {
            stype.numerical: LinearEncoder(),
            stype.categorical: EmbeddingEncoder(),
        },
        {
            stype.numerical: LinearEncoder(),
            stype.categorical: EmbeddingEncoder(),
        },
        {
            stype.numerical: LinearEncoder(),
            stype.categorical: EmbeddingEncoder(),
        },
    ],
    None,
])
def test_trompt(batch_size, stype_encoder_dicts):
    batch_size = 10
    channels = 8
    out_channels = 1
    num_prompts = 2
    num_layers = 3
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame[:batch_size]
    model = Trompt(
        channels=channels,
        out_channels=out_channels,
        num_prompts=num_prompts,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        stype_encoder_dicts=stype_encoder_dicts,
    )
    model.reset_parameters()
    out = model.forward_stacked(tensor_frame)
    assert out.shape == (batch_size, num_layers, out_channels)
    pred = model(tensor_frame)
    assert pred.shape == (batch_size, out_channels)
