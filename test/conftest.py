from typing import Callable, List

import pytest
import torch
from torch import Tensor

import torch_frame
from torch_frame import TensorFrame


@pytest.fixture()
def get_fake_tensor_frame() -> Callable:
    def _get_fake_tensor_frame(num_rows: int) -> TensorFrame:
        x_dict = {
            torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
            torch_frame.numerical: torch.randn(size=(num_rows, 2)),
        }
        col_names_dict = {
            torch_frame.categorical: ['a', 'b', 'c'],
            torch_frame.numerical: ['x', 'y'],
        }
        y = torch.randn(num_rows)

        return TensorFrame(
            x_dict=x_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    return _get_fake_tensor_frame


@pytest.fixture()
def get_fake_text_embedder() -> Callable:
    def _get_fake_text_embedder(sentences: List[str]) -> Tensor:
        return torch.rand(len(sentences), 10)

    return _get_fake_text_embedder
