import pandas as pd
import torch

from torch_frame.data import Rusty


def test_multicategorical_tensor_mapper():
    data = {
        'multi_cat1': ['A,B', 'B', None, 'C', 'B,C'],
        'multi_cat2': ['M', 'M,N', None, None, 'N']
    }
    expected_values = torch.tensor([1, 0, 0, 0, 0, 1, 0, 1])
    expected_boundaries = torch.tensor([0, 2, 3, 4, 6, 6, 6, 6, 6, 7, 8])
    df = pd.DataFrame(data)
    mapper = Rusty({
        'multi_cat1': ['B', 'A'],
        'multi_cat2': ['M', 'N']
    }, delimiter=",")

    values, boundaries = mapper.forward(df)
    print(values, boundaries)
    assert values.dtype == torch.long
    print(values)
    assert torch.equal(values, expected_values)
    assert torch.equal(boundaries, expected_boundaries)

    out = mapper.backward(values, boundaries)
    data = {
        'multi_cat1': ['A,B', 'B', None, 'None', 'B'],
        'multi_cat2': ['M', 'M,N', None, None, 'N']
    }
    pd.testing.assert_frame_equal(out, pd.DataFrame(data))
