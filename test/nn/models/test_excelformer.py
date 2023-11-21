import copy

import pytest
import torch

from torch_frame import TaskType
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.nn import ExcelFormer
import torch_frame


@pytest.mark.parametrize('task_type', [
    TaskType.REGRESSION,
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
])
@pytest.mark.parametrize('batch_size', [0, 5])
def test_excelformer(task_type, batch_size):
    in_channels = 8
    num_heads = 2
    num_layers = 6
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[torch_frame.numerical],
                                   task_type=task_type)
    dataset.materialize()
    if task_type.is_classification:
        out_channels = dataset.num_classes
    else:
        out_channels = 1
    num_cols = len(dataset.col_stats) - 1
    tensor_frame = dataset.tensor_frame[:batch_size]
    model = ExcelFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        num_cols=num_cols,
        num_layers=num_layers,
        num_heads=num_heads,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    model.reset_parameters()

    # Test the original forward pass
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)

    # Test the mixup forward pass
    feat_num = copy.copy(tensor_frame.feat_dict[torch_frame.numerical])
    out_mixedup, y_mixedup = model.forward_mixup(tensor_frame)
    assert out_mixedup.shape == (batch_size, out_channels)
    # Make sure the numerical feature is not modified.
    assert torch.allclose(feat_num, tensor_frame.feat_dict[torch_frame.numerical])

    if task_type.is_classification:
        assert y_mixedup.shape == (batch_size, out_channels)
    else:
        assert y_mixedup.shape == tensor_frame.y.shape
