from torch_frame import Stype
from torch_frame.data.dataset import Dataset
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)


def test_stypewise_encoder(get_fake_dataset):
    dataset: Dataset = get_fake_dataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame

    encoder = StypeWiseFeatureEncoder(
        out_channels=8,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    x, col_names = encoder(tensor_frame)
    assert x.shape == (10, 5, 8)
    assert col_names == ['a', 'b', 'c', 'x', 'y']


def test_stype_encoder(get_fake_dataset):
    dataset: Dataset = get_fake_dataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[Stype.categorical]
    ]
    encoder = EmbeddingEncoder(8, stats_list=stats_list)
    x = encoder(tensor_frame.x_dict[Stype.categorical])
    assert x.shape == (10, 2, 8)

    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[Stype.numerical]
    ]
    encoder = LinearEncoder(8, stats_list=stats_list)
    x = encoder(tensor_frame.x_dict[Stype.numerical])
    assert x.shape == (10, 3, 8)
