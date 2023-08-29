import pytest
import torch

import torch_frame
from torch_frame import TensorFrame


def test_tensor_frame_basics(get_fake_tensor_frame):
    tf = get_fake_tensor_frame(num_rows=10)
    assert tf.num_rows == len(tf) == 10

    assert str(tf) == ("TensorFrame(\n"
                       "  num_cols=5,\n"
                       "  num_rows=10,\n"
                       "  categorical (3): ['a', 'b', 'c'],\n"
                       "  numerical (2): ['x', 'y'],\n"
                       "  has_target=True,\n"
                       ")")


def test_tensor_frame_error():
    x_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(10, 3)),
        torch_frame.numerical: torch.randn(size=(10, 2)),
    }
    col_names_dict = {
        torch_frame.categorical: ['a', 'b', 'c'],
        torch_frame.numerical: ['x', 'y'],
    }
    y = torch.randn(10)

    # Wrong number of channels
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, ))
    with pytest.raises(ValueError, match='at least 2-dimensional'):
        TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict, y=y)
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment of the col_names and the number of columns in x_dict
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 4))
    with pytest.raises(ValueError, match='not align with'):
        TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict, y=y)
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment of the lengths within x_dict
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(11, 3))
    with pytest.raises(ValueError, match='not aligned'):
        TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict, y=y)
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment between the lengths of x_dict and y
    y = torch.randn(11)
    with pytest.raises(ValueError, match='not aligned'):
        TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict, y=y)


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
