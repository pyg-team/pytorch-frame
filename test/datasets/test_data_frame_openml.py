import pytest

from torch_frame.datasets import OpenMLDataset
from torch_frame.typing import TaskType


@pytest.mark.parametrize("dataset_id", [8, 31, 455])
def test_data_frame_openml(dataset_id):
    dataset = OpenMLDataset(dataset_id)
    if dataset_id == 8:
        assert dataset.task_type == TaskType.REGRESSION
        assert dataset.target_col == "drinks"
    if dataset_id == 31:
        assert dataset.task_type == TaskType.BINARY_CLASSIFICATION
        assert dataset.num_classes == 2
        assert dataset.target_col == "class"
    if dataset_id == 455:
        assert dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION
        assert dataset.num_classes == 3
        assert dataset.target_col == "origin"
