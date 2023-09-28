from typing import Callable

import pytest
import torch

import torch_frame
from torch_frame import TensorFrame


@pytest.fixture()
def get_fake_tensor_frame() -> Callable:
    def _get_fake_tensor_frame(num_rows: int) -> TensorFrame:
        feat_dict = {
            torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
            torch_frame.numerical: torch.randn(size=(num_rows, 2)),
        }
        col_names_dict = {
            torch_frame.categorical: ['a', 'b', 'c'],
            torch_frame.numerical: ['x', 'y'],
        }
        y = torch.randn(num_rows)

        return TensorFrame(
            feat_dict=feat_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    return _get_fake_tensor_frame
