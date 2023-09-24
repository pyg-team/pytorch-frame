import pytest
import torch
from torch.nn import ReLU

import torch_frame
from torch_frame import NAStrategy, stype
from torch_frame.data.dataset import Dataset
from torch_frame.data.stats import StatType
from torch_frame.datasets import FakeDataset
from torch_frame.nn import (
    EmbeddingEncoder,
    ExcelFormerEncoder,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    TextEmbeddingEncoder,
)


@pytest.mark.parametrize('encoder_cls_kwargs', [(EmbeddingEncoder, {})])
def test_categorical_feature_encoder(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.categorical]
    ]
    encoder = encoder_cls_kwargs[0](8, stats_list=stats_list,
                                    stype=stype.categorical,
                                    **encoder_cls_kwargs[1])
    x_cat = tensor_frame.x_dict[stype.categorical]
    x = encoder(x_cat)
    assert x.shape == (x_cat.size(0), x_cat.size(1), 8)

    # Perturb the first column
    num_categories = len(stats_list[0][StatType.COUNT])
    x_cat[:, 0] = (x_cat[:, 0] + 1) % num_categories
    x_perturbed = encoder(x_cat)
    # Make sure other column embeddings are unchanged
    assert (x_perturbed[:, 1:, :] == x[:, 1:, :]).all()


@pytest.mark.parametrize('encoder_cls_kwargs', [
    (LinearEncoder, {}),
    (LinearEncoder, {
        'post_module': ReLU(),
    }),
    (LinearBucketEncoder, {}),
    (LinearBucketEncoder, {
        'post_module': ReLU()
    }),
    (LinearEncoder, {
        'post_module': ReLU()
    }),
    (LinearPeriodicEncoder, {
        'n_bins': 4
    }),
    (LinearPeriodicEncoder, {
        'n_bins': 4,
        'post_module': ReLU(),
    }),
    (ExcelFormerEncoder, {
        'post_module': ReLU(),
    }),
])
def test_numerical_feature_encoder(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame

    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.numerical]
    ]
    encoder = encoder_cls_kwargs[0](8, stats_list=stats_list,
                                    stype=stype.numerical,
                                    **encoder_cls_kwargs[1])
    x_num = tensor_frame.x_dict[stype.numerical]
    x = encoder(x_num)
    assert x.shape == (x_num.size(0), x_num.size(1), 8)
    if 'post_module' in encoder_cls_kwargs[1]:
        assert encoder.post_module is not None
    else:
        assert encoder.post_module is None

    # Perturb the first column
    x_num[:, 0] = x_num[:, 0] + 10.
    x_perturbed = encoder(x_num)
    # Make sure other column embeddings are unchanged
    assert (x_perturbed[:, 1:, :] == x[:, 1:, :]).all()


@pytest.mark.parametrize('encoder_cls_kwargs', [
    (EmbeddingEncoder, {
        'na_strategy': NAStrategy.MOST_FREQUENT,
    }),
])
def test_categorical_feature_encoder_with_nan(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=True)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.categorical]
    ]

    encoder = encoder_cls_kwargs[0](8, stats_list=stats_list,
                                    stype=stype.categorical,
                                    **encoder_cls_kwargs[1])
    x_cat = tensor_frame.x_dict[stype.categorical]
    isnan_mask = x_cat == -1
    x = encoder(x_cat)
    assert x.shape == (x_cat.size(0), x_cat.size(1), 8)
    # Make sure there's no NaNs in x
    assert (~torch.isnan(x)).all()
    # Make sure original data is not modified
    assert (x_cat[isnan_mask] == -1).all()


@pytest.mark.parametrize('encoder_cls_kwargs', [
    (LinearEncoder, {
        'na_strategy': NAStrategy.ZEROS,
    }),
    (LinearEncoder, {
        'na_strategy': NAStrategy.MEAN,
    }),
])
def test_numerical_feature_encoder_with_nan(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=True)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.numerical]
    ]
    encoder = encoder_cls_kwargs[0](8, stats_list=stats_list,
                                    stype=stype.numerical,
                                    **encoder_cls_kwargs[1])
    x_num = tensor_frame.x_dict[stype.numerical]
    isnan_mask = x_num.isnan()
    x = encoder(x_num)
    assert x.shape == (x_num.size(0), x_num.size(1), 8)
    # Make sure there's no NaNs in x
    assert (~torch.isnan(x)).all()
    # Make sure original data is not modified
    assert x_num[isnan_mask].isnan().all()


def test_text_embedding_encoder(get_fake_text_embedding):
    num_rows = 10
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=[
            torch_frame.numerical,
            torch_frame.categorical,
            torch_frame.text_encoded,
        ],
        text_encoder=get_fake_text_embedding,
    )
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.text_encoded]
    ]
    encoder = TextEmbeddingEncoder(out_channels=5, stats_list=stats_list,
                                   stype=stype.text_encoded, in_channels=10)
    x_text = tensor_frame.x_dict[stype.text_encoded]
    x = encoder(x_text)
    assert x.shape == (10, 2, 5)
