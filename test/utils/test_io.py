import os
import os.path as osp
import shutil
import tempfile

import pytest

import torch_frame
from torch_frame import TensorFrame, load, save
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.testing.text_tokenizer import WhiteSpaceHashTokenizer

TEST_DIR = tempfile.TemporaryDirectory()
TEST_DATASET_NAME = 'test_dataset_tf.pt'
TEST_SAVE_LOAD_NAME = 'tf.pt'


def teardown_module():
    if osp.exists(TEST_DIR.name):
        shutil.rmtree(TEST_DIR.name, ignore_errors=True)


def get_fake_dataset(
    num_rows: int,
    col_to_text_embedder_cfg: TextEmbedderConfig,
    col_to_text_tokenizer_cfg: TextTokenizerConfig,
) -> FakeDataset:
    stypes = [
        torch_frame.numerical,
        torch_frame.categorical,
        torch_frame.multicategorical,
        torch_frame.text_embedded,
        torch_frame.text_tokenized,
        torch_frame.sequence_numerical,
        torch_frame.embedding,
    ]
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=stypes,
        col_to_text_embedder_cfg=col_to_text_embedder_cfg,
        col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
    )
    return dataset


def test_dataset_cache():
    num_rows = 10
    out_channels = 8

    col_to_text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(out_channels))
    col_to_text_tokenizer_cfg = TextTokenizerConfig(
        text_tokenizer=WhiteSpaceHashTokenizer())
    dataset = get_fake_dataset(
        num_rows,
        col_to_text_embedder_cfg,
        col_to_text_tokenizer_cfg,
    )

    path = osp.join(TEST_DIR.name, TEST_DATASET_NAME)
    dataset.materialize(path=path)

    new_dataset = get_fake_dataset(
        num_rows,
        col_to_text_embedder_cfg,
        col_to_text_tokenizer_cfg,
    )
    new_dataset.df = dataset.df

    # Test materialize via caching
    new_dataset.materialize(path=path)
    assert new_dataset.is_materialized
    assert dataset.col_stats == new_dataset.col_stats
    assert dataset.tensor_frame == new_dataset.tensor_frame

    # Test `tensor_frame` converter
    tf = new_dataset._to_tensor_frame_converter(dataset.df)
    assert dataset.tensor_frame == tf

    # Remove saved tensor frame object
    os.remove(path)

    new_dataset = get_fake_dataset(
        num_rows,
        col_to_text_embedder_cfg,
        col_to_text_tokenizer_cfg,
    )
    new_dataset.df = new_dataset.df

    # Test materialize again with specified path
    new_dataset.materialize()
    new_dataset.materialize(path=path)

    assert new_dataset.is_materialized


def test_save_load_tensor_frame():
    num_rows = 10
    out_channels = 8
    col_to_text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(out_channels))
    col_to_text_tokenizer_cfg = TextTokenizerConfig(
        text_tokenizer=WhiteSpaceHashTokenizer(),
        batch_size=None,
    )
    dataset = get_fake_dataset(num_rows, col_to_text_embedder_cfg,
                               col_to_text_tokenizer_cfg)
    dataset.materialize()

    path = osp.join(TEST_DIR.name, TEST_SAVE_LOAD_NAME)
    save(dataset.tensor_frame, dataset.col_stats, path)

    tf, col_stats = load(path)
    assert dataset.col_stats == col_stats
    assert dataset.tensor_frame == tf


class UntrustedClass:
    pass


@pytest.mark.skipif(
    not torch_frame.typing.WITH_PT24,
    reason='Requres PyTorch 2.4',
)
def test_load_weights_only_gracefully(tmpdir):
    save(
        tensor_frame=TensorFrame({}, {}),
        col_stats={'a': UntrustedClass()},
        path=tmpdir.join('tf.pt'),
    )
    with pytest.warns(UserWarning, match='Weights only load failed'):
        load(tmpdir.join('tf.pt'))
