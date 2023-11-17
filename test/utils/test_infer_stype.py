import pytest

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.utils import infer_df_stype


def get_fake_dataset(
    num_rows: int,
    text_embedder_cfg: TextEmbedderConfig,
    with_nan: bool,
) -> FakeDataset:
    stypes = [
        torch_frame.numerical,
        torch_frame.categorical,
        torch_frame.multicategorical,
        torch_frame.text_embedded,
        torch_frame.sequence_numerical,
        torch_frame.timestamp,
    ]
    dataset = FakeDataset(
        num_rows=num_rows,
        stypes=stypes,
        text_embedder_cfg=text_embedder_cfg,
        with_nan=with_nan,
    )
    return dataset


@pytest.mark.parametrize("with_nan", [True, False])
def test_infer_df_stype(with_nan):
    num_rows = 50
    text_embedder_cfg = TextEmbedderConfig(text_embedder=HashTextEmbedder(8))
    dataset = get_fake_dataset(num_rows, text_embedder_cfg, with_nan)
    col_to_stype_inferred = infer_df_stype(dataset.df)
    assert col_to_stype_inferred.keys() == dataset.col_to_stype.keys()
    for col_name, true_stype in dataset.col_to_stype.items():
        if true_stype in [
                torch_frame.numerical, torch_frame.categorical,
                torch_frame.timestamp
        ]:
            assert col_to_stype_inferred[col_name] == true_stype
