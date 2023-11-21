import pytest

from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import TabTransformer


@pytest.mark.parametrize('stypes', [[stype.categorical, stype.numerical],
                                    [stype.categorical], [stype.numerical]])
@pytest.mark.parametrize('batch_size', [0, 5])
def test_tab_transformer(stypes, batch_size):
    channels = 8
    out_channels = 1
    num_layers = 3
    num_heads = 2
    encoder_pad_size = 2
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False, stypes=stypes)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame[:batch_size]
    model = TabTransformer(
        channels=channels,
        out_channels=out_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        encoder_pad_size=encoder_pad_size,
        attn_dropout=0.,
        ffn_dropout=0.,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    model.reset_parameters()
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)
