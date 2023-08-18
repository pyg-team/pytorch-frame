import torch

import torch_frame
from torch_frame.datasets import Titanic


def test_titanic(tmp_path):
    dataset = Titanic(tmp_path)
    assert str(dataset) == 'Titanic()'
    assert len(dataset) == 891
    assert dataset.feat_cols == [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
    ]

    tensor_frame = dataset.to_tensor_frame()

    assert len(tensor_frame.x_dict) == 2
    assert tensor_frame.x_dict[torch_frame.numerical].dtype == torch.float
    assert tensor_frame.x_dict[torch_frame.numerical].size() == (891, 4)
    assert tensor_frame.x_dict[torch_frame.categorical].dtype == torch.long
    assert tensor_frame.x_dict[torch_frame.categorical].size() == (891, 3)
    assert tensor_frame.col_names_dict == {
        torch_frame.categorical: ['Pclass', 'Sex', 'Embarked'],
        torch_frame.numerical: ['Age', 'SibSp', 'Parch', 'Fare'],
    }
    assert tensor_frame.y.size() == (891, )
    assert tensor_frame.y.min() == 0 and tensor_frame.y.max() == 1
