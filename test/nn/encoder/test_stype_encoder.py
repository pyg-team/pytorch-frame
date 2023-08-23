import torch
from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn.encoder import EmbeddingEncoder, LinearEncoder, LinearBucketEncoder


def test_stype_feature_encoder():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.categorical]
    ]
    encoder = EmbeddingEncoder(8, stats_list=stats_list)
    x = encoder(tensor_frame.x_dict[stype.categorical])
    assert x.shape == (10, 2, 8)

    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.numerical]
    ]
    encoder = LinearEncoder(8, stats_list=stats_list)
    x = encoder(tensor_frame.x_dict[stype.numerical])
    assert x.shape == (10, 3, 8)

    encoder = LinearBucketEncoder(out_channels=8, stats_list=stats_list)

    # Apply the encoder to the numerical columns of the tensor frame
    x_numerical = tensor_frame.x_dict[stype.numerical]
    encoded_x = encoder(x_numerical)

    # Expected shape: [batch_size, num_numerical_cols, 8]
    assert encoded_x.shape == (
        10, len(tensor_frame.col_names_dict[stype.numerical]), 8)
