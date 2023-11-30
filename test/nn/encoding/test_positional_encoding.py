import pytest

from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import PositionalEncoding


@pytest.mark.parametrize('encoding_size', [8, 7])
def test_positional_encoding_shape(encoding_size):
    num_rows = 10
    dataset: Dataset = FakeDataset(num_rows=num_rows, stypes=[stype.timestamp],
                                   with_nan=False)
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.timestamp]
    ]
    positional_encoding = PositionalEncoding(encoding_size)

    feat_timestamp = tensor_frame.feat_dict[stype.timestamp]

    # test tensor with input shape batch_size, num_cols, 1, 7
    out = positional_encoding(feat_timestamp)
    assert out.shape == (num_rows, len(stats_list), 1, 7, encoding_size)

    # test tensor with input shape batch_size, num_cols, 7
    out = positional_encoding(feat_timestamp[:, :, 0, :])
    assert out.shape == (num_rows, len(stats_list), 7, encoding_size)

    # test tensor with input shape batch_size, 7
    feat_timestamp = feat_timestamp[:, 0, 0, :]
    out = positional_encoding(feat_timestamp)
    assert out.shape == (num_rows, 7, encoding_size)
