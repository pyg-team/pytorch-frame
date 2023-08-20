from torch_frame import Stype
from torch_frame.data.dataset import Dataset
from torch_frame.nn.encoder import EmbeddingEncoder, LinearEncoder


def test_stype_feature_encoder(get_fake_dataset):
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
