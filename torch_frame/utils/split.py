import numpy as np

# Mapping split name to integer.
SPLIT_TO_NUM = {'train': 0, 'val': 1, 'test': 2}


def generate_random_split(length: int, seed: int, train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          include_test: bool = True) -> np.ndarray:
    r"""Generate a list of random split assignments of the specified length.
    The elements are either :obj:`0`, :obj:`1`, or :obj:`2`, representing
    train, val, test, respectively. Note that this function relies on the fact
    that numpy's shuffle is consistent across versions, which has been
    historically the case.
    """
    assert train_ratio > 0
    assert val_ratio > 0

    if include_test:
        assert train_ratio + val_ratio < 1
        train_num = int(length * train_ratio)
        val_num = int(length * val_ratio)
        test_num = length - train_num - val_num
        arr = np.concatenate([
            np.full(train_num, SPLIT_TO_NUM['train']),
            np.full(val_num, SPLIT_TO_NUM['val']),
            np.full(test_num, SPLIT_TO_NUM['test'])
        ])
    else:
        assert train_ratio + val_ratio == 1
        train_num = int(length * train_ratio)
        val_num = length - train_num
        arr = np.concatenate([
            np.full(train_num, SPLIT_TO_NUM['train']),
            np.full(val_num, SPLIT_TO_NUM['val']),
        ])
    np.random.seed(seed)
    np.random.shuffle(arr)

    return arr
