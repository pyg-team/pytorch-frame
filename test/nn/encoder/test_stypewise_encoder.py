from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearBucketEncoder,
    LinearEncoder,
    PeriodicEncoder,
    StypeWiseFeatureEncoder,
)


def test_stypewise_feature_encoder():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame

    encoder = StypeWiseFeatureEncoder(
        out_channels=8, col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict, stype_encoder_dict={
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearEncoder(),
        })
    x, col_names = encoder(tensor_frame)
    assert x.shape == (10, 5, 8)
    assert col_names == ['a', 'b', 'c', 'x', 'y']

    encoder = StypeWiseFeatureEncoder(
        out_channels=8, col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict, stype_encoder_dict={
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearBucketEncoder(),
        })
    x, col_names = encoder(tensor_frame)
    assert x.shape == (10, 5, 8)
    assert col_names == ['a', 'b', 'c', 'x', 'y']

    encoder = StypeWiseFeatureEncoder(
        out_channels=8, col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict, stype_encoder_dict={
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: PeriodicEncoder(),
        })
    x, col_names = encoder(tensor_frame)
    assert x.shape == (10, 5, 8)
    assert col_names == ['a', 'b', 'c', 'x', 'y']
