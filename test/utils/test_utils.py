import os.path as osp
import tempfile

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder


def test_dataset_save_load_tf():
    num_rows = 10
    out_channels = 8
    stypes = [
        torch_frame.numerical, torch_frame.categorical,
        torch_frame.text_embedded
    ]
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(out_channels))
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=stypes,
        text_embedder_cfg=text_embedder_cfg,
    )

    dataset.materialize()

    with tempfile.TemporaryDirectory() as tempdir:
        path = osp.join(tempdir, 'tf.pt')
        dataset.materialize(path=path)
        new_dataset = FakeDataset(
            num_rows=num_rows,
            stypes=stypes,
            text_embedder_cfg=text_embedder_cfg,
        )
        new_dataset.materialize(path=path)
        assert len(new_dataset.col_stats) == 8
        assert new_dataset.tensor_frame.y is not None
        assert len(new_dataset.tensor_frame.x_dict) == 3
        assert len(new_dataset.tensor_frame.col_names_dict) == 3
