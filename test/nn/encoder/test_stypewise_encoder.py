from torch_frame import Stype
from torch_frame.data.dataset import Dataset
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)


def test_stypewise_feature_encoder(get_fake_dataset):
    dataset: Dataset = get_fake_dataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame

    encoder = StypeWiseFeatureEncoder(
        out_channels=8, col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict, stype_encoder_dict={
            Stype.categorical: EmbeddingEncoder(),
            Stype.numerical: LinearEncoder(),
        })
    x, col_names = encoder(tensor_frame)
    assert x.shape == (10, 5, 8)
    assert col_names == ['a', 'b', 'c', 'x', 'y']
