from typing import List, Tuple

import torch
from torch import Tensor

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.nn import Decoder, FeatureEncoder, TableConv


def test_simple_basecls():
    # Instantiate each base class with a simple class and test e2e pipeline.
    class SimpleFeatureEncoder(FeatureEncoder):
        def __init__(
            self,
            out_channels: int,
            num_numerical: int,
            num_categories: List[int],
        ):
            super().__init__()

            self.out_channels = out_channels
            self.num_numerical = num_numerical
            self.num_categories = num_categories

            self.lins = torch.nn.ModuleList([
                torch.nn.Linear(1, out_channels) for _ in range(num_numerical)
            ])
            self.embs = torch.nn.ModuleList([
                torch.nn.Embedding(num_category, out_channels)
                for num_category in num_categories
            ])

        def forward(self, tf: TensorFrame) -> Tuple[Tensor, List[str]]:
            xs = []
            for i, lin in enumerate(self.lins):
                xs.append(lin(tf.feat_dict[torch_frame.numerical][:, i:i + 1]))
            for i, emb in enumerate(self.embs):
                xs.append(emb(tf.feat_dict[torch_frame.categorical][:, i]))

            x = torch.stack(xs, dim=1)
            col_names = (tf.col_names_dict[stype.numerical] +
                         tf.col_names_dict[stype.categorical])

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

    tf = TensorFrame(
        feat_dict={
            torch_frame.numerical: torch.randn(10, 2),
            torch_frame.categorical: torch.randint(0, 5, (10, 2)),
        },
        col_names_dict={
            torch_frame.numerical: ['num1', 'num2'],
            torch_frame.categorical: ['cat1', 'cat2'],
        },
    )

    feat_encoder = SimpleFeatureEncoder(
        out_channels=8,
        num_numerical=2,
        num_categories=[5, 5],
    )
    table_conv1 = SimpleTableConv(in_channels=8, out_channels=16)
    table_conv2 = SimpleTableConv(in_channels=16, out_channels=8)
    decoder = SimpleDecoder()

    x, col_names = feat_encoder(tf)
    # [batch_size, num_cols, hidden_channels]
    assert x.shape == (10, 4, 8)
    assert col_names == ['num1', 'num2', 'cat1', 'cat2']
    x = table_conv1(x)
    assert x.shape == (10, 4, 16)
    x = table_conv2(x)
    assert x.shape == (10, 4, 8)
    x = decoder(x)
    assert x.shape == (10, 8)
