from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from torch_frame.nn import FeatureEncoder, TableConv, Decoder
from torch_frame.encoder import TensorEncoder
from torch_frame import TensorFrame, stype
from torch_frame.typing import DataFrame


def test_simple_basecls():
    # Instantiate each base class with a simple class and test e2e pipeline.
    class SimpleTensorEncoder(TensorEncoder):
        def to_tensor(self, df: DataFrame) -> TensorFrame:
            x_list_dict: Dict[stype, List[Tensor]] = defaultdict(lambda: [])
            col_names_dict: Dict[stype, List[str]] = defaultdict(lambda: [])

            for col_name in df.keys():
                stype_name = self.col2stype[col_name]
                tensor = torch.from_numpy(df[col_name].values).view(-1, 1)
                if stype_name == stype.categorical:
                    x_list_dict[stype_name].append(tensor.to(torch.long))
                else:
                    x_list_dict[stype_name].append(tensor.to(torch.float))
                col_names_dict[stype_name].append(col_name)

            x_dict: Dict[stype, Tensor] = {
                stype_name: torch.cat(x_list, dim=1)
                for stype_name, x_list in x_list_dict.items()
            }
            return TensorFrame(x_dict=x_dict, col_names_dict=col_names_dict)

    class SimpleFeatureEncoder(FeatureEncoder):
        def __init__(
            self,
            out_channels: int,
            num_numerical: int,
            num_categories: List[int],
        ):
            super().__init__()
            self.lin_numerical = torch.nn.Linear(num_numerical, out_channels)
            self.emb_categorical = torch.nn.ModuleList([
                torch.nn.Embedding(num_category, out_channels)
                for num_category in num_categories
            ])

        def forward(self, tf: TensorFrame) -> Tuple[Tensor, List[str]]:
            x_num = self.lin_numerical(
                tf.x_dict[stype.numerical].unsqueeze(dim=2))
            num_cat_cols = tf.x_dict[stype.categorical].size(1)
            x_cat_list = []
            for i in range(num_cat_cols):
                x_cat: Tensor = self.emb_categorical[i](
                    tf.x_dict[stype.categorical][:, i])
                x_cat_list.append(x_cat.unsqueeze(dim=1))
            x_cat = torch.cat(x_cat_list, dim=1)
            x = torch.cat([x_num, x_cat], dim=1)
            col_names = tf.col_names_dict[stype.numerical] + tf.col_names_dict[
                stype.categorical]
            return x, col_names

    class SimpleTableConv(TableConv):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x: Tensor) -> Tensor:
            B, C, H = x.shape
            x = x.view(-1, H)
            return self.lin(x).view(B, C, -1)

    class SimpleDecoder(Decoder):
        def forward(self, x: Tensor) -> Tensor:
            # Pool along the column axis
            return torch.mean(x, dim=1)

    df = DataFrame({
        'num1': np.random.randn(10),
        'num2': np.random.randn(10),
        'cat1': np.random.randint(0, 3, 10),
        'cat2': np.random.randint(0, 5, 10),
    })
    tensor_encoder = SimpleTensorEncoder(
        col2stype={
            'num1': stype.numerical,
            'num2': stype.numerical,
            'cat1': stype.categorical,
            'cat2': stype.categorical,
        })
    feat_encoder = SimpleFeatureEncoder(out_channels=8, num_numerical=1,
                                        num_categories=[3, 5])
    table_conv1 = SimpleTableConv(in_channels=8, out_channels=16)
    table_conv2 = SimpleTableConv(in_channels=16, out_channels=8)
    decoder = SimpleDecoder()

    tf = tensor_encoder.to_tensor(df)
    x, col_names = feat_encoder(tf)
    # [batch_size, num_cols, hidden_channels]
    assert x.shape == (10, 4, 8)
    assert col_names == list(df.keys())
    x = table_conv1(x)
    assert x.shape == (10, 4, 16)
    x = table_conv2(x)
    assert x.shape == (10, 4, 8)
    x = decoder(x)
    assert x.shape == (10, 8)
