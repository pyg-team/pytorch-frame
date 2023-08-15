from torch_frame import TensorFrame
import torch_frame
import torch


def test_tensor_frame_basics():
    x_dict = {}
    x_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))
    x_dict[torch_frame.numerical] = torch.randn(size=(10, 2))
    col_names_dict = {}
    col_names_dict[torch_frame.categorical] = ['a', 'b', 'c']
    col_names_dict[torch_frame.numerical] = ['x', 'y']
    y = torch.randn(10)
    tf = TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict, y=y)
    assert tf.num_rows == 10
