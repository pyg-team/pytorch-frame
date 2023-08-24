from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearBucketEncoder,
    LinearEncoder,
)


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
    assert encoded_x.shape == (10, 3, 8)


def test_stype_feature_encoder_with_nan():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=True)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.categorical]
    ]
    encoder = EmbeddingEncoder(8, stats_list=stats_list)
    x_cat = tensor_frame.x_dict[stype.categorical]
    isnan_mask = x_cat == -1
    x = encoder(x_cat)
    assert x.shape == (10, 2, 8)
    assert (x[isnan_mask, :] == 0).all()
    # Make sure original data is not modified
    assert (x_cat[isnan_mask] == -1).all()

    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.numerical]
    ]

    encoder = LinearEncoder(8, stats_list=stats_list)
    x_num = tensor_frame.x_dict[stype.numerical]
    isnan_mask = x_num.isnan()
    x = encoder(x_num)
    assert x.shape == (10, 3, 8)
    assert (x[isnan_mask, :] == 0).all()
    # Make sure original data is not modified
    assert x_num[isnan_mask].isnan().all()

    # TODO check for LinearBucketEncoder.
    encoder = LinearBucketEncoder(8, stats_list=stats_list)
    x_num = tensor_frame.x_dict[stype.numerical]
    isnan_mask = x_num.isnan()
    x = encoder(x_num)
    assert x.shape == (10, 3, 8)
    assert (x[isnan_mask, :] == 0).all()
    # Make sure original data is not modified
    assert x_num[isnan_mask].isnan().all()
