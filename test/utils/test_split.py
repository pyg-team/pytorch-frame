import numpy as np

from torch_frame.datasets import FakeDataset
from torch_frame.typing import TrainingStage
from torch_frame.utils.split import SPLIT_TO_NUM, generate_random_split


def test_generate_random_split():
    num_data = 20
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    split = generate_random_split(num_data, seed=42,
                                  ratios=[train_ratio, val_ratio])
    assert (split == SPLIT_TO_NUM.get(TrainingStage.TRAIN)).sum() == int(
        num_data * train_ratio)
    assert (split == SPLIT_TO_NUM.get(TrainingStage.VAL)).sum() == int(
        num_data * val_ratio)
    assert (split == SPLIT_TO_NUM.get(TrainingStage.TEST)).sum() == int(
        num_data * test_ratio)
    assert np.allclose(
        split,
        np.array([0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]),
    )


def test_split_e2e_basic():
    # TODO: Add several more test cases using @pytest.mark.parametrize
    num_rows = 10
    dataset = FakeDataset(num_rows=num_rows).materialize()
    dataset.random_split([0.5, 0.2])
    train_set, val_set, test_set = dataset.split()
    train_set.num_rows, val_set.num_rows == (int(10 * 0.5), int(10 * 0.2))
    if test_set is not None:
        test_set.num_rows == 10 - int(10 * 0.5) - int(10 * 0.2)
