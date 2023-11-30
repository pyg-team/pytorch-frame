import random
from typing import Callable

import pytest
import torch

import torch_frame
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


@pytest.fixture()
def get_fake_tensor_frame() -> Callable:
    def _get_fake_tensor_frame(num_rows: int) -> TensorFrame:
        col_names_dict = {
            torch_frame.categorical: ['cat_1', 'cat_2', 'cat_3'],
            torch_frame.numerical: ['num_1', 'num_2'],
            torch_frame.multicategorical: ['multicat_1', 'multicat_2'],
            torch_frame.text_embedded:
            ['text_embedded_1', 'text_embedded_2', 'text_embedded_3'],
            torch_frame.text_tokenized:
            ['text_tokenized_1', 'text_tokenized_2'],
            torch_frame.sequence_numerical: ['seq_num_1', 'seq_num_2'],
            torch_frame.embedding: ['emb_1', 'emb_2'],
        }
        feat_dict = {
            torch_frame.categorical:
            torch.randint(
                0, 3,
                size=(num_rows, len(col_names_dict[torch_frame.categorical]))),
            torch_frame.numerical:
            torch.randn(size=(num_rows,
                              len(col_names_dict[torch_frame.numerical]))),
            torch_frame.multicategorical:
            MultiNestedTensor.from_tensor_mat([[
                torch.arange(random.randint(0, 10)) for _ in range(
                    len(col_names_dict[torch_frame.multicategorical]))
            ] for _ in range(num_rows)]),
            torch_frame.text_embedded:
            MultiEmbeddingTensor.from_tensor_list([
                torch.randn(num_rows, random.randint(1, 5))
                for _ in range(len(col_names_dict[torch_frame.text_embedded]))
            ]),
            torch_frame.text_tokenized: {
                'input_id':
                MultiNestedTensor.from_tensor_mat([[
                    torch.randint(0, 5, size=(random.randint(0, 10), ))
                    for _ in range(
                        len(col_names_dict[torch_frame.text_tokenized]))
                ] for _ in range(num_rows)]),
                'mask':
                MultiNestedTensor.from_tensor_mat([[
                    torch.randint(0, 5, size=(random.randint(0, 10), ))
                    for _ in range(
                        len(col_names_dict[torch_frame.text_tokenized]))
                ] for _ in range(num_rows)]),
            },
            torch_frame.sequence_numerical:
            MultiNestedTensor.from_tensor_mat([[
                torch.randn(random.randint(0, 10)) for _ in range(
                    len(col_names_dict[torch_frame.sequence_numerical]))
            ] for _ in range(num_rows)]),
            torch_frame.embedding:
            MultiEmbeddingTensor.from_tensor_list([
                torch.randn(num_rows, random.randint(1, 5))
                for _ in range(len(col_names_dict[torch_frame.embedding]))
            ])
        }

        y = torch.randn(num_rows)

        return TensorFrame(
            feat_dict=feat_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    return _get_fake_tensor_frame
