import os.path as osp
import shutil
import tempfile

import torch

import torch_frame
from torch_frame import load_tf, save_tf
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import TensorFrame
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder

TEST_DIR = tempfile.TemporaryDirectory()
TEST_DATASET_NAME = 'test_dataset_tf.pt'
TEST_SAVE_LOAD_NAME = 'test_save_load_tf.pt'


def teardown_module():
    if osp.exists(TEST_DIR.name):
        shutil.rmtree(TEST_DIR.name, ignore_errors=True)


def compare_tfs(tf_a: TensorFrame, tf_b: TensorFrame):
    assert torch.equal(tf_a.y, tf_b.y)
    assert len(tf_a.feat_dict) == len(tf_b.feat_dict)
    for stype in tf_a.feat_dict:
        assert torch.equal(tf_a.feat_dict[stype], tf_b.feat_dict[stype])
    assert len(tf_a.col_names_dict) == len(tf_b.col_names_dict)
    for stype in tf_a.col_names_dict:
        assert tf_a.col_names_dict[stype] == tf_b.col_names_dict[stype]


def get_fake_dataset(num_rows: int,
                     text_embedder_cfg: TextEmbedderConfig) -> FakeDataset:
    stypes = [
        torch_frame.numerical, torch_frame.categorical,
        torch_frame.text_embedded
    ]
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=stypes,
        text_embedder_cfg=text_embedder_cfg,
    )
    return dataset


def test_dataset_cache():
    num_rows = 10
    out_channels = 8

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(out_channels))
    dataset = get_fake_dataset(num_rows, text_embedder_cfg)

    path = osp.join(TEST_DIR.name, TEST_DATASET_NAME)
    dataset.materialize(path=path)
    new_dataset = get_fake_dataset(num_rows, text_embedder_cfg)
    new_dataset.df = dataset.df

    # Test materialize via caching
    new_dataset.materialize(path=path)
    assert new_dataset.is_materialized
    assert dataset.col_stats == new_dataset.col_stats
    compare_tfs(dataset.tensor_frame, new_dataset.tensor_frame)

    # Test `tensor_frame` converter
    tf = new_dataset._to_tensor_frame_converter(dataset.df)
    compare_tfs(dataset.tensor_frame, tf)


def test_save_load_tf():
    num_rows = 10
    out_channels = 8
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(out_channels))
    dataset = get_fake_dataset(num_rows, text_embedder_cfg)
    dataset.materialize()

    path = osp.join(TEST_DIR.name, TEST_SAVE_LOAD_NAME)
    save_tf(dataset.tensor_frame, dataset.col_stats, path)

    tf, col_stats = load_tf(path)
    assert dataset.col_stats == col_stats
    compare_tfs(dataset.tensor_frame, tf)
