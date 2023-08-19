import torch

import torch_frame
from torch_frame.datasets import AdultCensusIncome


def test_adult_census_income(tmp_path):
    dataset = AdultCensusIncome(tmp_path)
    assert str(dataset) == 'AdultCensusIncome()'
    assert len(dataset) == 32561
    assert dataset.feat_cols == [
        'age',
        'workclass',
        'education',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
        'native.country',
    ]

    tensor_frame = dataset.to_tensor_frame()

    assert len(tensor_frame.x_dict) == 2
    assert tensor_frame.x_dict[torch_frame.numerical].dtype == torch.float
    assert tensor_frame.x_dict[torch_frame.numerical].size() == (32561, 4)
    assert tensor_frame.x_dict[torch_frame.categorical].dtype == torch.long
    assert tensor_frame.x_dict[torch_frame.categorical].size() == (32561, 8)
    assert tensor_frame.col_names_dict == {
        torch_frame.categorical: [
            'workclass', 'education', 'marital.status', 'occupation',
            'relationship', 'race', 'sex', 'native.country'
        ],
        torch_frame.numerical:
        ['age', 'capital.gain', 'capital.loss', 'hours.per.week'],
    }
    assert tensor_frame.y.size() == (32561, )
    assert tensor_frame.y.min() == 0 and tensor_frame.y.max() == 1
