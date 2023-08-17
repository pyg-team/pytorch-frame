import pytest
import torch

import torch_frame
from torch_frame import TensorFrame


def test_tensor_frame_basics():
    x_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(10, 3)),
        torch_frame.numerical: torch.randn(size=(10, 2)),
    }
    col_names_dict = {
        torch_frame.categorical: ['a', 'b', 'c'],
        torch_frame.numerical: ['x', 'y'],
    }
    y = torch.randn(10)

    tf = TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict, y=y)

    assert tf.num_rows == 10


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
