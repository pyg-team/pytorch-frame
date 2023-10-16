import numpy as np

from torch_frame.utils.split import SPLIT_TO_NUM, generate_random_split


def test_generate_random_split():
    num_data = 20
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    split = generate_random_split(num_data, seed=42, train_ratio=train_ratio,
                                  val_ratio=val_ratio)
    assert (split == SPLIT_TO_NUM['train']).sum() == int(num_data *
                                                         train_ratio)
    assert (split == SPLIT_TO_NUM['val']).sum() == int(num_data * val_ratio)
    assert (split == SPLIT_TO_NUM['test']).sum() == int(num_data * test_ratio)
    assert np.allclose(
        split,
        np.array([0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]),
    )
