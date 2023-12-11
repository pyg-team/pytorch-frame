import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.datasets.huggingface_dataset import HuggingFaceDatasetDict
from torch_frame.testing import withPackage
from torch_frame.testing.text_embedder import HashTextEmbedder


@withPackage("datasets")
def test_huggingface_dataset_dict():
    path = "emotion"
    col_to_stype = {
        "text": torch_frame.text_embedded,
        "label": torch_frame.categorical
    }
    target_col = "label"
    col_to_text_embedder_cfg = TextEmbedderConfig(
        text_embedder=HashTextEmbedder(10))

    dataset = HuggingFaceDatasetDict(
        path=path,
        col_to_stype=col_to_stype,
        target_col=target_col,
        col_to_text_embedder_cfg=col_to_text_embedder_cfg,
    )
    dataset.materialize()
    assert dataset.tensor_frame.num_cols == 1
    assert dataset.tensor_frame.num_rows == 20000
    assert dataset.tensor_frame.y is not None
    assert len(dataset.tensor_frame.feat_dict) == 1
    assert torch_frame.text_embedded in dataset.tensor_frame.feat_dict
