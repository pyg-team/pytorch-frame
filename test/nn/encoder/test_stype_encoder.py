import copy

import pytest
import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential

import torch_frame
from torch_frame import NAStrategy, stype
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data.dataset import Dataset
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.data.stats import StatType
from torch_frame.datasets import FakeDataset
from torch_frame.nn import (
    EmbeddingEncoder,
    ExcelFormerEncoder,
    LinearBucketEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
    LinearPeriodicEncoder,
    MultiCategoricalEmbeddingEncoder,
    StackEncoder,
    TimestampEncoder,
)
from torch_frame.nn.encoding import CyclicEncoding
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.testing.text_tokenizer import (
    RandomTextModel,
    WhiteSpaceHashTokenizer,
)


@pytest.mark.parametrize("encoder_cls_kwargs", [(EmbeddingEncoder, {})])
def test_categorical_feature_encoder(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.categorical]
    ]
    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.categorical,
        **encoder_cls_kwargs[1],
    )
    feat_cat = tensor_frame.feat_dict[stype.categorical].clone()
    col_names = tensor_frame.col_names_dict[stype.categorical]
    x = encoder(feat_cat, col_names)
    assert x.shape == (feat_cat.size(0), feat_cat.size(1), 8)
    # Make sure no in-place modification
    assert torch.allclose(feat_cat, tensor_frame.feat_dict[stype.categorical])

    # Perturb the first column
    num_categories = len(stats_list[0][StatType.COUNT])
    feat_cat[:, 0] = (feat_cat[:, 0] + 1) % num_categories
    x_perturbed = encoder(feat_cat, col_names)
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
    (StackEncoder, {}),
])
def test_numerical_feature_encoder(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame

    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.numerical]
    ]
    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.numerical,
        **encoder_cls_kwargs[1],
    )
    feat_num = tensor_frame.feat_dict[stype.numerical].clone()
    col_names = tensor_frame.col_names_dict[stype.numerical]
    x = encoder(feat_num, col_names)
    assert x.shape == (feat_num.size(0), feat_num.size(1), 8)
    # Make sure no in-place modification
    assert torch.allclose(feat_num, tensor_frame.feat_dict[stype.numerical])
    if "post_module" in encoder_cls_kwargs[1]:
        assert encoder.post_module is not None
    else:
        assert encoder.post_module is None

    # Perturb the first column
    feat_num[:, 0] = feat_num[:, 0] + 10.0
    x_perturbed = encoder(feat_num, col_names)
    # Make sure other column embeddings are unchanged
    assert (x_perturbed[:, 1:, :] == x[:, 1:, :]).all()


@pytest.mark.parametrize(
    "encoder_cls_kwargs",
    [
        (MultiCategoricalEmbeddingEncoder, {
            "mode": "mean"
        }),
        (MultiCategoricalEmbeddingEncoder, {
            "mode": "sum"
        }),
        (MultiCategoricalEmbeddingEncoder, {
            "mode": "max"
        }),
    ],
)
def test_multicategorical_feature_encoder(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10,
                                   stypes=[stype.multicategorical],
                                   with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.multicategorical]
    ]
    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.multicategorical,
        **encoder_cls_kwargs[1],
    )
    feat_multicat = tensor_frame.feat_dict[stype.multicategorical].clone()
    col_names = tensor_frame.col_names_dict[stype.multicategorical]
    x = encoder(feat_multicat, col_names)
    # Make sure no in-place modification
    assert torch.allclose(
        feat_multicat.values,
        tensor_frame.feat_dict[stype.multicategorical].values)
    assert torch.allclose(
        feat_multicat.offset,
        tensor_frame.feat_dict[stype.multicategorical].offset)
    assert x.shape == (feat_multicat.size(0), feat_multicat.size(1), 8)

    # Perturb the first column
    num_categories = len(stats_list[0][StatType.MULTI_COUNT])
    feat_multicat[:,
                  0].values = (feat_multicat[:, 0].values + 1) % num_categories
    x_perturbed = encoder(feat_multicat, col_names)
    # Make sure other column embeddings are unchanged
    assert (x_perturbed[:, 1:, :] == x[:, 1:, :]).all()


@pytest.mark.parametrize(
    "encoder_cls_kwargs",
    [(
        TimestampEncoder,
        {},
    )],
)
def test_timestamp_feature_encoder(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, stypes=[stype.timestamp])
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.timestamp]
    ]

    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.timestamp,
        **encoder_cls_kwargs[1],
    )
    feat_timestamp = tensor_frame.feat_dict[stype.timestamp].clone()
    col_names = tensor_frame.col_names_dict[stype.timestamp]
    x = encoder(feat_timestamp, col_names)
    # Make sure no in-place modification
    assert torch.allclose(feat_timestamp,
                          tensor_frame.feat_dict[stype.timestamp])
    assert x.shape == (feat_timestamp.size(0), feat_timestamp.size(1), 8)


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

    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.categorical,
        **encoder_cls_kwargs[1],
    )
    feat_cat = tensor_frame.feat_dict[stype.categorical]
    col_names = tensor_frame.col_names_dict[stype.categorical]
    isnan_mask = feat_cat == -1
    x = encoder(feat_cat, col_names)
    assert x.shape == (feat_cat.size(0), feat_cat.size(1), 8)
    # Make sure there's no NaNs in x
    assert (~torch.isnan(x)).all()
    # Make sure original data is not modified
    assert (feat_cat[isnan_mask] == -1).all()


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
    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.numerical,
        **encoder_cls_kwargs[1],
    )
    feat_num = tensor_frame.feat_dict[stype.numerical]
    col_names = tensor_frame.col_names_dict[stype.numerical]
    isnan_mask = feat_num.isnan()
    x = encoder(feat_num, col_names)
    assert x.shape == (feat_num.size(0), feat_num.size(1), 8)
    # Make sure there's no NaNs in x
    assert (~torch.isnan(x)).all()
    # Make sure original data is not modified
    assert feat_num[isnan_mask].isnan().all()


@pytest.mark.parametrize(
    "encoder_cls_kwargs",
    [(
        MultiCategoricalEmbeddingEncoder,
        {
            "mode": "mean",
            "na_strategy": NAStrategy.ZEROS
        },
    )],
)
def test_multicategorical_feature_encoder_with_nan(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10,
                                   stypes=[stype.multicategorical],
                                   with_nan=True)
    dataset.materialize()
    assert len(dataset.df) != len(dataset.df.dropna())
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.multicategorical]
    ]

    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.multicategorical,
        **encoder_cls_kwargs[1],
    )
    feat_multicat = tensor_frame.feat_dict[stype.multicategorical]
    col_names = tensor_frame.col_names_dict[stype.multicategorical]
    isnan_mask = feat_multicat.values == -1
    x = encoder(feat_multicat, col_names)
    assert x.shape == (feat_multicat.size(0), feat_multicat.size(1), 8)
    # Make sure there's no NaNs in x
    assert (~torch.isnan(x)).all()
    # Make sure original data is not modified
    assert (feat_multicat.values[isnan_mask] == -1).all()


@pytest.mark.parametrize(
    "encoder_cls_kwargs",
    [
        (TimestampEncoder, {
            "na_strategy": NAStrategy.NEWEST_TIMESTAMP
        }),
        (TimestampEncoder, {
            "na_strategy": NAStrategy.MEDIAN_TIMESTAMP
        }),
        (TimestampEncoder, {
            "na_strategy": NAStrategy.OLDEST_TIMESTAMP
        }),
    ],
)
def test_timestamp_feature_encoder_with_nan(encoder_cls_kwargs):
    dataset: Dataset = FakeDataset(num_rows=10, stypes=[stype.timestamp],
                                   with_nan=True)
    dataset.materialize()
    assert len(dataset.df) != len(dataset.df.dropna())
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.timestamp]
    ]
    encoder = encoder_cls_kwargs[0](
        8,
        stats_list=stats_list,
        stype=stype.timestamp,
        **encoder_cls_kwargs[1],
    )
    feat_timestamp = tensor_frame.feat_dict[stype.timestamp]
    col_names = tensor_frame.col_names_dict[stype.timestamp]
    x = encoder(feat_timestamp, col_names)
    assert x.shape == (feat_timestamp.size(0), feat_timestamp.size(1), 8)
    assert (~torch.isnan(x)).all()


def test_embedding_encoder():
    num_rows = 20
    out_channels = 5
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=[
            torch_frame.embedding,
        ],
    )
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.embedding]
    ]
    encoder = LinearEmbeddingEncoder(
        out_channels=out_channels,
        stats_list=stats_list,
        stype=stype.embedding,
    )
    feat_emb = tensor_frame.feat_dict[stype.embedding].clone()
    col_names = tensor_frame.col_names_dict[stype.embedding]
    x = encoder(feat_emb, col_names)
    # Make sure no in-place modification
    assert torch.allclose(feat_emb.values,
                          tensor_frame.feat_dict[stype.embedding].values)
    assert torch.allclose(feat_emb.offset,
                          tensor_frame.feat_dict[stype.embedding].offset)
    assert x.shape == (
        num_rows,
        len(tensor_frame.col_names_dict[stype.embedding]),
        out_channels,
    )


def test_text_tokenized_encoder():
    num_rows = 20
    num_hash_bins = 10
    out_channels = 5
    text_emb_channels = 15
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=[
            torch_frame.text_tokenized,
        ],
        col_to_text_tokenizer_cfg=TextTokenizerConfig(
            text_tokenizer=WhiteSpaceHashTokenizer(
                num_hash_bins=num_hash_bins),
            batch_size=None,
        ),
    )
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.text_tokenized]
    ]
    model = RandomTextModel(text_emb_channels=text_emb_channels)
    col_to_model_cfg = {
        col_name: ModelConfig(model=model, out_channels=text_emb_channels)
        for col_name in tensor_frame.col_names_dict[stype.text_tokenized]
    }
    encoder = LinearModelEncoder(
        out_channels=out_channels,
        stats_list=stats_list,
        stype=stype.text_tokenized,
        col_to_model_cfg=col_to_model_cfg,
    )
    feat_text = copy.deepcopy(tensor_frame.feat_dict[stype.text_tokenized])
    col_names = tensor_frame.col_names_dict[stype.text_tokenized]
    x = encoder(feat_text, col_names)
    assert x.shape == (
        num_rows,
        len(tensor_frame.col_names_dict[stype.text_tokenized]),
        out_channels,
    )
    # Make sure no in-place modification
    assert isinstance(feat_text, dict) and isinstance(
        tensor_frame.feat_dict[stype.text_tokenized], dict)
    assert feat_text.keys() == tensor_frame.feat_dict[
        stype.text_tokenized].keys()
    for key in feat_text.keys():
        assert torch.allclose(
            feat_text[key].values,
            tensor_frame.feat_dict[stype.text_tokenized][key].values)
        assert torch.allclose(
            feat_text[key].offset,
            tensor_frame.feat_dict[stype.text_tokenized][key].offset)


def test_linear_model_encoder():
    num_rows = 20
    out_channels = 8
    data_stypes = [
        torch_frame.numerical,
        torch_frame.text_embedded,
        torch_frame.timestamp,
        torch_frame.categorical,
        torch_frame.multicategorical,
        torch_frame.sequence_numerical,
        torch_frame.embedding,
    ]
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=data_stypes,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(out_channels=out_channels),
            batch_size=None,
        ),
    )
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = []
    col_to_model_cfg = {}
    encoder_dict = {}
    for data_stype in data_stypes:
        data_stype = data_stype.parent
        stats_list.extend(
            dataset.col_stats[col_name]
            for col_name in tensor_frame.col_names_dict[data_stype])
        for col_name in tensor_frame.col_names_dict[data_stype]:
            if data_stype == torch_frame.embedding:
                in_channels = dataset.col_stats[col_name][StatType.EMB_DIM]
                model = EmbeddingModel(in_channels, out_channels)
            elif data_stype == torch_frame.numerical:
                in_channels = 1
                model = NumericalModel(in_channels, out_channels)
            elif data_stype == torch_frame.timestamp:
                in_channels = 7
                model = TimestampModel(in_channels, out_channels)
            elif data_stype == torch_frame.categorical:
                count_index, _ = dataset.col_stats[col_name][StatType.COUNT]
                model = CategoricalModel(len(count_index), out_channels)
            elif data_stype == torch_frame.multicategorical:
                count_index, _ = dataset.col_stats[col_name][
                    StatType.MULTI_COUNT]
                model = MultiCategoricalModel(len(count_index), out_channels)
            elif data_stype == torch_frame.sequence_numerical:
                in_channels = dataset.col_stats[col_name][StatType.MAX_LENGTH]
                model = SequenceNumericalModel(in_channels, out_channels)
            else:
                raise ValueError(f"Stype {data_stype} not supported")
            col_to_model_cfg[col_name] = ModelConfig(model=model,
                                                     out_channels=out_channels)

        encoder_dict[data_stype] = LinearModelEncoder(
            out_channels=out_channels,
            stats_list=stats_list,
            stype=data_stype,
            col_to_model_cfg=col_to_model_cfg,
        )

    for data_stype in data_stypes:
        data_stype = data_stype.parent
        col_names = tensor_frame.col_names_dict[data_stype]
        x = encoder_dict[data_stype](tensor_frame.feat_dict[data_stype],
                                     col_names)
        assert x.shape == (
            num_rows,
            len(tensor_frame.col_names_dict[data_stype]),
            out_channels,
        )


class EmbeddingModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = Sequential(Linear(in_channels, out_channels), ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, x: MultiEmbeddingTensor) -> Tensor:
        # [batch_size, 1, embedding_size]
        return self.mlp(x.values.unsqueeze(dim=1))


class NumericalModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = Sequential(Linear(in_channels, out_channels), ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, 1, 1] -> [batch_size, 1, out_channels]
        return self.mlp(x)


class TimestampModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(in_channels, out_channels, out_channels))
        self.cyclic_encoding = CyclicEncoding(out_size=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, 1, num_time_feats]
        x = x.to(torch.float32)
        # [batch_size, 1, num_time_feats, out_channels]
        x_cyclic = self.cyclic_encoding(x / x.max())
        # [batch_size, 1, out_channels]
        return torch.einsum('ijk,jkl->il', x_cyclic.squeeze(1),
                            self.weight).unsqueeze(dim=1)


class CategoricalModel(torch.nn.Module):
    def __init__(self, num_categories: int, out_channels: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_categories, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, 1, 1] -> [batch_size, 1]
        x = x.squeeze(dim=1)
        # [batch_size, 1] -> [batch_size, 1, out_channels]
        return self.emb(x)


class MultiCategoricalModel(torch.nn.Module):
    def __init__(self, num_categories: int, out_channels: int):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(num_categories, out_channels)

    def forward(self, x: MultiNestedTensor) -> Tensor:
        return self.emb(x.values, x.offset[:-1]).unsqueeze(dim=1)


class SequenceNumericalModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = Sequential(Linear(in_channels, out_channels), ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, x: MultiNestedTensor) -> Tensor:
        # [batch_size, 1, max_length]
        return self.mlp(x.to_dense(fill_value=0.0))
