import pytest
import torch

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.datasets import FakeDataset
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.testing.text_tokenizer import WhiteSpaceHashTokenizer


@pytest.mark.parametrize('with_nan', [True, False])
@pytest.mark.parametrize('tokenize_with_batch', [True, False])
@pytest.mark.parametrize('text_batch_size', [None, 5])
@pytest.mark.parametrize('use_different_embedder_cfg', [False, True])
def test_fake_dataset(with_nan, tokenize_with_batch, text_batch_size,
                      use_different_embedder_cfg):
    num_rows = 20
    out_channels = 10

    if use_different_embedder_cfg:
        text_embedded_cols = ['text_embedded_1', 'text_embedded_2']
        text_embedder_cfg = {
            col:
            TextEmbedderConfig(
                text_embedder=HashTextEmbedder(out_channels * (i + 1)),
                batch_size=3)
            for i, col in enumerate(text_embedded_cols)
        }
    else:
        text_embedder_cfg = TextEmbedderConfig(
            text_embedder=HashTextEmbedder(out_channels),
            batch_size=text_batch_size)

    dataset = FakeDataset(
        num_rows=num_rows,
        with_nan=with_nan,
        stypes=[
            torch_frame.numerical,
            torch_frame.categorical,
            torch_frame.multicategorical,
            torch_frame.sequence_numerical,
            torch_frame.text_embedded,
            torch_frame.text_tokenized,
            torch_frame.embedding,
        ],
        create_split=True,
        text_embedder_cfg=text_embedder_cfg,
        text_tokenizer_cfg=TextTokenizerConfig(
            text_tokenizer=WhiteSpaceHashTokenizer(
                num_hash_bins=12, batched=tokenize_with_batch),
            batch_size=text_batch_size),
    )
    assert str(dataset) == 'FakeDataset()'
    assert len(dataset) == num_rows
    assert dataset.feat_cols == [
        'num_1',
        'num_2',
        'num_3',
        'cat_1',
        'cat_2',
        'multicat_1',
        'multicat_2',
        'multicat_3',
        'multicat_4',
        'seq_num_1',
        'seq_num_2',
        'text_embedded_1',
        'text_embedded_2',
        'text_tokenized_1',
        'text_tokenized_2',
        'emb_1',
        'emb_2',
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

    feat_multicat = tensor_frame.feat_dict[torch_frame.multicategorical]
    assert isinstance(feat_multicat, MultiNestedTensor)
    assert feat_multicat.size(0) == num_rows
    assert feat_multicat.size(1) == 4

    feat_sequence_numerical = tensor_frame.feat_dict[
        torch_frame.sequence_numerical]
    assert isinstance(feat_sequence_numerical, MultiNestedTensor)
    assert feat_sequence_numerical.size(0) == num_rows
    assert feat_sequence_numerical.size(1) == 2

    feat_text_embedded = tensor_frame.feat_dict[torch_frame.text_embedded]
    assert feat_text_embedded.dtype == torch.float
    text_embedded_cols = tensor_frame.col_names_dict[torch_frame.text_embedded]
    assert feat_text_embedded.shape == (num_rows, len(text_embedded_cols), -1)
    if use_different_embedder_cfg:
        assert feat_text_embedded.offset.max() == out_channels * sum(
            i + 1 for i in range(len(text_embedded_cols)))

    feat_text_tokenized = tensor_frame.feat_dict[torch_frame.text_tokenized]
    assert isinstance(feat_text_tokenized['input_ids'], MultiNestedTensor)
    assert feat_text_tokenized['input_ids'].dtype == torch.int64
    assert feat_text_tokenized['input_ids'].shape == (num_rows, 2, -1)
    assert feat_text_tokenized['attention_mask'].dtype == torch.bool
    assert feat_text_tokenized['attention_mask'].shape == (num_rows, 2, -1)
    assert feat_text_tokenized['input_ids'].to_dense(
        fill_value=0).shape == (num_rows, 2, 2)
    assert feat_text_tokenized['attention_mask'].to_dense(
        fill_value=False).shape == (num_rows, 2, 2)

    feat_emb = tensor_frame.feat_dict[torch_frame.embedding]
    assert isinstance(feat_emb, MultiEmbeddingTensor)

    # Test dataset split
    train_dataset, val_dataset, test_dataset = dataset.split()
    assert train_dataset.num_rows == num_rows - 2
    assert val_dataset.num_rows == 1
    assert test_dataset.num_rows == 1
