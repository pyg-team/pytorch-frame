import copy

import pytest
import torch

import torch_frame
from torch_frame import TensorFrame


def test_tensor_frame_basics(get_fake_tensor_frame):
    tf = get_fake_tensor_frame(num_rows=10)
    assert tf.num_rows == len(tf) == 10
    assert str(tf) == (
        "TensorFrame(\n"
        "  num_cols=14,\n"
        "  num_rows=10,\n"
        "  categorical (3): ['cat_1', 'cat_2', 'cat_3'],\n"
        "  numerical (2): ['num_1', 'num_2'],\n"
        "  multicategorical (2): ['multicat_1', 'multicat_2'],\n"
        "  text_embedded (3): ['text_embedded_1', 'text_embedded_2',"
        " 'text_embedded_3'],\n"
        "  text_tokenized (2): ['text_tokenized_1', 'text_tokenized_2'],\n"
        "  sequence_numerical (2): ['seq_num_1', 'seq_num_2'],\n"
        "  has_target=True,\n"
        "  device='cpu',\n"
        ")")


def test_tensor_frame_error():
    feat_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(10, 3)),
        torch_frame.numerical: torch.randn(size=(10, 2)),
    }
    col_names_dict = {
        torch_frame.categorical: ['cat_1', 'cat_2', 'cat_3'],
        torch_frame.numerical: ['num_1', 'num_2'],
    }
    y = torch.randn(10)

    # Wrong number of channels
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, ))
    with pytest.raises(ValueError, match='at least 2-dimensional'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment of the col_names and the number of columns in feat_dict
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 4))
    with pytest.raises(ValueError, match='not align with'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment of the lengths within feat_dict
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(11, 3))
    with pytest.raises(ValueError, match='not aligned'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment between the lengths of feat_dict and y
    y = torch.randn(11)
    with pytest.raises(ValueError, match='not aligned'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)


@pytest.mark.parametrize('index', [
    4,
    [4, 8],
    range(2, 6),
    torch.tensor([4, 8]),
])
def test_tensor_frame_index_select(get_fake_tensor_frame, index):
    tf = get_fake_tensor_frame(num_rows=10)

    out = tf[index]

    if isinstance(index, int):
        assert out.num_rows == 1
    else:
        assert out.num_rows == len(index)

    assert out.col_names_dict == tf.col_names_dict


def test_empty_tensor_frame():
    # Categorical feature is empty
    feat_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(10, 0)),
        torch_frame.numerical: torch.randn(size=(10, 2)),
    }
    col_names_dict = {
        torch_frame.categorical: [],
        torch_frame.numerical: ['num_1', 'num_2'],
    }
    with pytest.raises(RuntimeError, match='Empty columns'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)

    col_names_dict = {
        torch_frame.numerical: ['num_1', 'num_2'],
    }
    with pytest.raises(ValueError, match='The keys of feat_dict'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)


def test_equal_tensor_frame(get_fake_tensor_frame):
    tf1 = get_fake_tensor_frame(num_rows=10)

    # Test equal
    tf2 = copy.copy(tf1)
    assert tf1 == tf2
    assert tf2 == tf1

    # Test difference in col_names_dict
    tf2.col_names_dict[torch_frame.numerical] = [
        name + '_' for name in tf1.col_names_dict[torch_frame.numerical]
    ]
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in y (Tensor versus None)
    tf2 = copy.copy(tf1)
    tf2.y = None
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in y (Tensor versus Tensor)
    tf2.y = torch.randn(tf1.y.shape)
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in feat_dict
    tf2 = copy.copy(tf1)
    tf2.feat_dict[torch_frame.numerical] = torch.randn(
        tf2.feat_dict[torch_frame.numerical].shape)
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in length
    tf2 = get_fake_tensor_frame(num_rows=11)
    assert tf1 != tf2
    assert tf2 != tf1
