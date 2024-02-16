import math

import numpy as np

from torch_frame.typing import TrainingStage

# Mapping split name to integer.
SPLIT_TO_NUM = {
    TrainingStage.TRAIN: 0,
    TrainingStage.VAL: 1,
    TrainingStage.TEST: 2
}


def generate_random_split(
    length: int,
    ratios: list[float],
    seed: int = 0,
) -> np.ndarray:
    r"""Generate a list of random split assignments of the specified length.
    The elements are either :obj:`0`, :obj:`1`, or :obj:`2`, representing
    train, val, test, respectively. Note that this function relies on the fact
    that numpy's shuffle is consistent across versions, which has been
    historically the case.

    Args:
        length (int): The length of the dataset.
        ratios (list[float]): Ratios for split assignment. When ratios
            contains 2 variables, we will generate train/val/test set
            respectively based on the split ratios (the 1st variable in
            the list will be the ratio for train set, the 2nd will be
            the ratio for val set and the remaining data will be used
            for test set). When ratios contains 1 variable, we will only
            generate train/val set. (the variable in)
        seed (int, optional): The seed for the randomness generator.

    Returns:
        A np.ndarra object representing the split.
    """
    validate_split_ratios(ratios)
    ratios_length = len(ratios)
    if length < ratios_length + 1:
        raise ValueError(
            f"We want to split data into {ratios_length + 1} disjoint set. "
            f"However data contains {length} data point. Consider "
            f"increase your data size.")

    # train_num = int(length * train_ratio)
    # val_num = int(length * val_ratio)
    # test_num = length - train_num - val_num
    train_num = math.floor(length * ratios[0])
    val_num = math.floor(
        length * ratios[1]) if ratios_length == 2 else length - train_num
    test_num = None
    if ratios_length == 2:
        test_num = length - train_num - val_num

    arr = np.concatenate([
        np.full(train_num, SPLIT_TO_NUM.get(TrainingStage.TRAIN)),
        np.full(val_num, SPLIT_TO_NUM.get(TrainingStage.VAL)),
    ])

    if ratios_length == 2:
        arr = np.concatenate(
            [arr, np.full(test_num, SPLIT_TO_NUM.get(TrainingStage.TEST))])

    np.random.seed(seed)
    np.random.shuffle(arr)

    return arr


def validate_split_ratios(ratio: list[float]):
    if len(ratio) > 2:
        raise ValueError("No more than three training splits is supported")
    if len(ratio) < 1:
        raise ValueError("At least two training splits are required")

    for val in ratio:
        if val < 0:
            raise ValueError("'ratio' can not contain negative values")

    if sum(ratio) - 1 > 1e-2:
        raise ValueError("'ratio' exceeds more than 100% of the data")
