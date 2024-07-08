import tempfile

import torch

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_frame.datasets import Movielens1M
from torch_frame.testing.text_embedder import HashTextEmbedder


def test_movielens_1m():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = Movielens1M(
            temp_dir,
            col_to_text_embedder_cfg=TextEmbedderConfig(
                text_embedder=HashTextEmbedder(10)),
        )
    assert str(dataset) == 'Movielens1M()'
    assert len(dataset) == 1000209
    assert dataset.feat_cols == [
        'user_id', 'gender', 'age', 'occupation', 'zip', 'movie_id', 'title',
        'genres', 'timestamp'
    ]

    dataset = dataset.materialize()

    tensor_frame = dataset.tensor_frame
    assert len(tensor_frame.feat_dict) == 4
    assert tensor_frame.feat_dict[torch_frame.categorical].dtype == torch.int64
    assert tensor_frame.feat_dict[torch_frame.categorical].size() == (1000209,
                                                                      6)
    assert tensor_frame.feat_dict[
        torch_frame.multicategorical].dtype == torch.int64
    assert tensor_frame.feat_dict[torch_frame.embedding].dtype == torch.float32
    assert tensor_frame.col_names_dict == {
        torch_frame.categorical:
        ['age', 'gender', 'movie_id', 'occupation', 'user_id', 'zip'],
        torch_frame.multicategorical: ['genres'],
        torch_frame.timestamp: ['timestamp'],
        torch_frame.embedding: ['title'],
    }
    assert tensor_frame.y.size() == (1000209, )
    assert tensor_frame.y.min() == 1 and tensor_frame.y.max() == 5

    col_stats = dataset.col_stats
    assert len(col_stats) == 10
    assert StatType.COUNT in col_stats['user_id']
    assert StatType.MULTI_COUNT in col_stats['genres']
    assert StatType.YEAR_RANGE in col_stats['timestamp']
    assert StatType.EMB_DIM in col_stats['title']
