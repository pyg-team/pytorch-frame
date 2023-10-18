import pytest

from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import FTTransformer


def test_ft_transformer():
    batch_size = 10
    channels = 8
    out_channels = 1
    num_layers = 3
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    # Feature-based embeddings
    model = FTTransformer(
        channels=channels,
        out_channels=out_channels,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)


@pytest.mark.xfail
def test_no_graph_breaks():
    import torch._dynamo as dynamo

    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tf = dataset.tensor_frame
    model = FTTransformer(
        channels=8,
        out_channels=1,
        num_layers=3,
        col_stats=dataset.col_stats,
        col_names_dict=tf.col_names_dict,
    )
    explanation = dynamo.explain(model)(tf)
    assert explanation.graph_break_count == 0
