import pandas as pd
import pytest

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.utils import infer_df_stype


def get_fake_dataset(
    num_rows: int,
    col_to_text_embedder_cfg: TextEmbedderConfig,
    with_nan: bool,
) -> FakeDataset:
    stypes = [
        torch_frame.numerical,
        torch_frame.categorical,
        torch_frame.multicategorical,
        torch_frame.text_embedded,
        torch_frame.sequence_numerical,
        torch_frame.timestamp,
        torch_frame.embedding,
    ]
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=stypes,
        col_to_text_embedder_cfg=col_to_text_embedder_cfg,
        with_nan=with_nan,
    )
    return dataset


@pytest.mark.parametrize("with_nan", [True, False])
def test_infer_df_stype(with_nan):
    num_rows = 200
    col_to_text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(8))
    dataset = get_fake_dataset(num_rows, col_to_text_embedder_cfg, with_nan)
    col_to_stype_inferred = infer_df_stype(dataset.df)
    assert col_to_stype_inferred == dataset.col_to_stype


def test_infer_stypes():
    # Test when multicategoricals are lists
    df = pd.DataFrame({
        'category': [['Books', 'Mystery, Thriller'],
                     ['Books', "Children's Books", 'Geography'],
                     ['Books', 'Health', 'Fitness & Dieting'],
                     ['Books', 'Teen & oung Adult']] * 50,
        'id': [i for i in range(200)]
    })
    col_to_stype_inferred = infer_df_stype(df)
    assert col_to_stype_inferred['category'] == torch_frame.multicategorical

    df = pd.DataFrame({'bool': [True] * 50 + [False] * 50})

    col_to_stype_inferred = infer_df_stype(df)
    assert col_to_stype_inferred['bool'] == torch_frame.categorical
