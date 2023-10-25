import numpy as np
import pandas as pd

from torch_frame.data import Rusty


def test_rusty():
    data = {
        'multi_cat1': ['A,B', 'B', None, 'C', 'B,C'],
        'multi_cat2': ['M', 'M,N', None, None, 'N']
    }
    expected_values = np.array([1, 0, 0, 0, 0, 1, 0, 1])
    expected_boundaries = np.array([2, 3, 4, 6, 6, 6, 6, 6, 7, 8])
    df = pd.DataFrame(data)
    mapper = Rusty({
        'multi_cat1': ['B', 'A'],
        'multi_cat2': ['M', 'N']
    }, sep=",")

    values, boundaries = mapper.forward(df)
    assert values.dtype == np.int64
    assert np.all(values == expected_values)
    assert np.all(boundaries == expected_boundaries)

    out = mapper.backward(values, boundaries)
    data = {
        'multi_cat1': ['A,B', 'B', None, None, 'B'],
        'multi_cat2': ['M', 'M,N', None, None, 'N']
    }
    pd.testing.assert_frame_equal(out, pd.DataFrame(data))
