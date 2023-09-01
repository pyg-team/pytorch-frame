from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import Trompt


def test_trompt():
    batch_size = 10
    channels = 8
    out_channels = 1
    num_cols = 5
    num_prompts = 2
    num_layers = 6
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    model = Trompt(
        channels=channels,
        out_channels=out_channels,
        num_cols=num_cols,
        num_prompts=num_prompts,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    out = model(tensor_frame)
    assert out.shape == (batch_size, num_layers, out_channels)
