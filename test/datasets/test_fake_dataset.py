import numpy as np
import pytest
import torch
from sklearn.preprocessing import OneHotEncoder

import torch_frame
from torch_frame.datasets import FakeDataset
from torch_frame.typing import Series


class SimpleTextEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def encode(self, ser: Series) -> torch.Tensor:
        list_tokens = []
        for row in ser:
            if isinstance(row, str):
                list_tokens.append(row.split(' '))
            else:
                list_tokens.append([''])
        max_len = max(len(tokens) for tokens in list_tokens)
        data = [
            tokens + [' '] * (max_len - len(tokens)) for tokens in list_tokens
        ]
        emb = self.encoder.fit_transform(data).toarray().astype(np.float32)
        return torch.from_numpy(emb)


@pytest.mark.parametrize('with_nan', [True, False])
def test_fake_dataset(with_nan):
    text_encoder = SimpleTextEncoder()
    num_rows = 20
    dataset = FakeDataset(
        num_rows=num_rows,
        with_nan=with_nan,
        stypes=[
            torch_frame.numerical,
            torch_frame.categorical,
            torch_frame.text_encoded,
        ],
        text_encoder=text_encoder.encode,
    )
    assert str(dataset) == 'FakeDataset()'
    assert len(dataset) == num_rows
    assert dataset.feat_cols == ['a', 'b', 'c', 'x', 'y', 'text_1', 'text_2']
    assert dataset.target_col == 'target'

    dataset = dataset.materialize()
    tensor_frame = dataset.tensor_frame
    x_num = tensor_frame.x_dict[torch_frame.numerical]
    assert x_num.dtype == torch.float
    assert x_num.size() == (num_rows, 3)
    if with_nan:
        assert torch.isnan(x_num).any()
    else:
        assert (~torch.isnan(x_num)).all()

    x_cat = tensor_frame.x_dict[torch_frame.categorical]
    assert x_cat.dtype == torch.long
    assert x_cat.size() == (num_rows, 2)
    if with_nan:
        assert (x_cat == -1).any()
    else:
        assert (x_cat >= 0).all()

    x_text_encoded = tensor_frame.x_dict[torch_frame.text_encoded]
    assert x_text_encoded.dtype == torch.float
    if with_nan:
        assert x_text_encoded.size() == (num_rows, 2, 4)
    else:
        assert x_text_encoded.size() == (num_rows, 2, 2)
