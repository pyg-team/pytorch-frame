import pytest

import torch_frame.nn.encoder as Encoder
from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn.encoder import StypeWiseFeatureEncoder


@pytest.mark.parametrize('encoder_cat_cls_kwargs', [('EmbeddingEncoder', {})])
@pytest.mark.parametrize('encoder_num_cls_kwargs', [
    ('LinearEncoder', {}),
    ('LinearBucketEncoder', {}),
    ('LinearPeriodicEncoder', {
        'n_bins': 4
    }),
])
def test_stypewise_feature_encoder(
    encoder_cat_cls_kwargs,
    encoder_num_cls_kwargs,
):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame

    encoder = StypeWiseFeatureEncoder(
        out_channels=8,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        stype_encoder_dict={
            stype.categorical:
            getattr(Encoder,
                    encoder_cat_cls_kwargs[0])(**encoder_cat_cls_kwargs[1]),
            stype.numerical:
            getattr(Encoder,
                    encoder_num_cls_kwargs[0])(**encoder_num_cls_kwargs[1]),
        },
    )
    x, col_names = encoder(tensor_frame)
    assert x.shape == (10, tensor_frame.num_cols, 8)
    assert col_names == ['a', 'b', 'c', 'x', 'y']
