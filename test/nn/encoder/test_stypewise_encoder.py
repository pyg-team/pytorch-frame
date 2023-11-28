import pytest

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearBucketEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
    LinearPeriodicEncoder,
    MultiCategoricalEmbeddingEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.testing.text_tokenizer import (
    RandomTextModel,
    WhiteSpaceHashTokenizer,
)


@pytest.mark.parametrize('encoder_cat_cls_kwargs', [(EmbeddingEncoder, {})])
@pytest.mark.parametrize('encoder_num_cls_kwargs', [
    (LinearEncoder, {}),
    (LinearBucketEncoder, {}),
    (LinearPeriodicEncoder, {
        'n_bins': 4
    }),
])
@pytest.mark.parametrize('encoder_multicategorical_cls_kwargs', [
    (MultiCategoricalEmbeddingEncoder, {}),
])
@pytest.mark.parametrize('encoder_text_embedded_cls_kwargs', [
    (LinearEmbeddingEncoder, {}),
])
@pytest.mark.parametrize('encoder_text_tokenized_cls_kwargs', [
    (LinearModelEncoder, {
        'model': RandomTextModel(12, 2),
        'in_channels': 12,
    }),
])
@pytest.mark.parametrize('encoder_embedding_cls_kwargs', [
    (LinearEmbeddingEncoder, {}),
])
def test_stypewise_feature_encoder(
    encoder_cat_cls_kwargs,
    encoder_num_cls_kwargs,
    encoder_multicategorical_cls_kwargs,
    encoder_text_embedded_cls_kwargs,
    encoder_text_tokenized_cls_kwargs,
    encoder_embedding_cls_kwargs,
):
    num_rows = 10
    dataset: Dataset = FakeDataset(
        num_rows=num_rows,
        with_nan=False,
        stypes=[
            stype.categorical,
            stype.numerical,
            stype.multicategorical,
            stype.text_embedded,
            stype.text_tokenized,
            stype.embedding,
        ],
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(out_channels=16, ),
            batch_size=None,
        ),
        text_tokenizer_cfg=TextTokenizerConfig(
            text_tokenizer=WhiteSpaceHashTokenizer(),
            batch_size=None,
        ),
    )
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    out_channels = 8

    encoder = StypeWiseFeatureEncoder(
        out_channels=out_channels,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        stype_encoder_dict={
            stype.categorical:
            encoder_cat_cls_kwargs[0](**encoder_cat_cls_kwargs[1]),
            stype.numerical:
            encoder_num_cls_kwargs[0](**encoder_num_cls_kwargs[1]),
            stype.multicategorical:
            encoder_multicategorical_cls_kwargs[0](
                **encoder_multicategorical_cls_kwargs[1]),
            stype.text_embedded:
            encoder_text_embedded_cls_kwargs[0](
                **encoder_text_embedded_cls_kwargs[1]),
            stype.text_tokenized:
            encoder_text_tokenized_cls_kwargs[0](
                **encoder_text_tokenized_cls_kwargs[1]),
            stype.embedding:
            encoder_embedding_cls_kwargs[0](**encoder_embedding_cls_kwargs[1]),
        },
    )
    x, col_names = encoder(tensor_frame)
    assert x.shape == (num_rows, tensor_frame.num_cols, out_channels)
    assert col_names == [
        "num_1",
        "num_2",
        "num_3",
        "cat_1",
        "cat_2",
        "text_embedded_1",
        "text_embedded_2",
        "text_tokenized_1",
        "text_tokenized_2",
        "multicat_1",
        "multicat_2",
        "multicat_3",
        "multicat_4",
        "emb_1",
        "emb_2",
    ]
