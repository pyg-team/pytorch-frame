import pytest
import torch

from torch_frame import stype
from torch_frame.datasets import FakeDataset
from torch_frame.nn.models import (
    ExcelFormer,
    FTTransformer,
    ResNet,
    TabNet,
    TabTransformer,
    Trompt,
)
from torch_frame.testing import withPackage


@withPackage("torch>=2.1.0")
@pytest.mark.parametrize(
    "model_cls, model_kwargs, stypes, expected_graph_breaks",
    [
        pytest.param(
            FTTransformer,
            dict(channels=8),
            None,
            2,
            id="FTTransformer",
        ),
        pytest.param(ResNet, dict(channels=8), None, 2, id="ResNet"),
        pytest.param(
            TabNet,
            dict(
                split_feat_channels=2,
                split_attn_channels=2,
                gamma=0.1,
            ),
            None,
            4,
            id="TabNet",
        ),
        pytest.param(
            TabTransformer,
            dict(
                channels=8,
                num_heads=2,
                encoder_pad_size=2,
                attn_dropout=0.5,
                ffn_dropout=0.5,
            ),
            None,
            4,
            id="TabTransformer",
        ),
        pytest.param(
            Trompt,
            dict(channels=8, num_prompts=2),
            None,
            11,
            id="Trompt",
        ),
        pytest.param(
            ExcelFormer,
            dict(in_channels=8, num_cols=3, num_heads=1),
            [stype.numerical],
            4,
            id="ExcelFormer",
        ),
    ],
)
def test_compile_graph_break(
    model_cls,
    model_kwargs,
    stypes,
    expected_graph_breaks,
):
    torch._dynamo.config.suppress_errors = True

    dataset = FakeDataset(
        num_rows=10,
        with_nan=False,
        stypes=stypes or [stype.categorical, stype.numerical],
    )
    dataset.materialize()
    tf = dataset.tensor_frame
    model = model_cls(
        out_channels=1,
        num_layers=2,
        col_stats=dataset.col_stats,
        col_names_dict=tf.col_names_dict,
        **model_kwargs,
    )
    explanation = torch._dynamo.explain(model)(tf)
    assert explanation.graph_break_count <= expected_graph_breaks
