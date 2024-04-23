import copy

import pytest
import torch

from torch_frame import TaskType, stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.nn import ExcelFormer


@pytest.mark.parametrize('task_type', [
    TaskType.REGRESSION,
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
])
@pytest.mark.parametrize('batch_size', [0, 5])
@pytest.mark.parametrize('mixup', ['none', 'feature', 'hidden'])
def test_excelformer(task_type, batch_size, mixup):
    in_channels = 8
    num_heads = 2
    num_layers = 6
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical],
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
        mixup=mixup,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
    )
    model.reset_parameters()

    # Test the original forward pass
    out = model(tensor_frame)
    assert out.shape == (batch_size, out_channels)

    # Test the mixup forward pass
    feat_num = copy.copy(tensor_frame.feat_dict[stype.numerical])
    # Set lazy mutual information scores for `feature` mixup
    tensor_frame.mi_scores = torch.rand(torch.Size((feat_num.shape[1],)))
    out_mixedup, y_mixedup = model(tensor_frame, mixup_encoded=True)
    assert out_mixedup.shape == (batch_size, out_channels)
    # Make sure the numerical feature is not modified.
    assert torch.allclose(feat_num, tensor_frame.feat_dict[stype.numerical])

    if task_type.is_classification:
        assert y_mixedup.shape == (batch_size, out_channels)
    else:
        assert y_mixedup.shape == tensor_frame.y.shape
