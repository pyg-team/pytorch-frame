import pytest
import torch

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.testing.text_tokenizer import WhiteSpaceHashTokenizer


@pytest.mark.parametrize('with_nan', [True, False])
def test_fake_dataset(with_nan):
    num_rows = 20
    out_channels = 10
    dataset = FakeDataset(
        num_rows=num_rows,
        with_nan=with_nan,
        stypes=[
            torch_frame.numerical,
            torch_frame.categorical,
            torch_frame.text_embedded,
            torch_frame.text_tokenized,
        ],
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(out_channels), batch_size=None),
        text_tokenizer_cfg=TextTokenizerConfig(
            text_tokenizer=WhiteSpaceHashTokenizer(num_hash_bins=12),
            batch_size=5),
    )
    assert str(dataset) == 'FakeDataset()'
    assert len(dataset) == num_rows
    assert dataset.feat_cols == [
        'a', 'b', 'c', 'x', 'y', 'text_1', 'text_2', 'text_3', 'text_4'
    ]
    assert dataset.target_col == 'target'

    dataset = dataset.materialize()
    tensor_frame = dataset.tensor_frame
    feat_num = tensor_frame.feat_dict[torch_frame.numerical]
    assert feat_num.dtype == torch.float
    assert feat_num.size() == (num_rows, 3)
    if with_nan:
        assert torch.isnan(feat_num).any()
    else:
        assert (~torch.isnan(feat_num)).all()

    feat_cat = tensor_frame.feat_dict[torch_frame.categorical]
    assert feat_cat.dtype == torch.long
    assert feat_cat.size() == (num_rows, 2)
    if with_nan:
        assert (feat_cat == -1).any()
    else:
        assert (feat_cat >= 0).all()

    feat_text_embedded = tensor_frame.feat_dict[torch_frame.text_embedded]
    assert feat_text_embedded.dtype == torch.float
    assert feat_text_embedded.shape == (
        num_rows, len(tensor_frame.col_names_dict[torch_frame.text_embedded]),
        out_channels)

    feat_text_tokenized = tensor_frame.feat_dict[torch_frame.text_tokenized]
    assert feat_text_tokenized['input_ids'].dtype == torch.int64
    assert feat_text_tokenized['input_ids'].shape == (num_rows, 2, -1)
