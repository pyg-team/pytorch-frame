from typing import Callable

import pytest
import torch

import torch_frame
from torch_frame import TensorFrame
from torch_frame.data import MultiNestedTensor


@pytest.fixture()
def get_fake_tensor_frame() -> Callable:
    def _get_fake_tensor_frame(
            num_rows: int, with_text_embedded: bool = False,
            with_text_tokenized: bool = False) -> TensorFrame:

        feat_dict = {
            torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
            torch_frame.numerical: torch.randn(size=(num_rows, 2)),
        }

        col_names_dict = {
            torch_frame.categorical: ['a', 'b', 'c'],
            torch_frame.numerical: ['x', 'y'],
        }

        if with_text_embedded:
            feat_dict[torch_frame.text_embedded] = torch.randn(size=(num_rows,
                                                                     2, 10))
            col_names_dict[torch_frame.text_embedded] = [
                'text_embedded_1', 'text_embedded_2'
            ]

        if with_text_tokenized:
            tokenized_list = []
            for _ in range(num_rows):
                num_tokens = int(torch.randint(3, 5, size=()))
                tokenized_list.append([
                    torch.randint(0, 3, size=(num_tokens, )),
                    torch.randint(0, 3, size=(num_tokens, ))
                ])
            input_ids = MultiNestedTensor.from_tensor_mat(tokenized_list)
            feat_dict[torch_frame.text_tokenized] = {}
            feat_dict[torch_frame.text_tokenized]['input_ids'] = input_ids
            feat_dict[
                torch_frame.text_tokenized]['input_ids_2'] = input_ids.clone()
            col_names_dict[torch_frame.text_tokenized] = [
                'text_tokenized_1', 'text_tokenized_2'
            ]

        y = torch.randn(num_rows)

        return TensorFrame(
            feat_dict=feat_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    return _get_fake_tensor_frame
