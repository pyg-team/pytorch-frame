import torch

import torch_frame
from torch_frame.data.stats import StatType
from torch_frame.datasets import Titanic


def test_titanic(tmp_path):
    dataset = Titanic(tmp_path)
    assert str(dataset) == 'Titanic()'
    assert len(dataset) == 891
    assert dataset.feat_cols == [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
    ]

    dataset = dataset.materialize()

    tensor_frame = dataset.tensor_frame
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

    col_stats = dataset.col_stats
    assert len(col_stats) == 8
    assert StatType.CATEGORY_COUNTS in col_stats['Survived']
    assert StatType.CATEGORY_COUNTS in col_stats['Pclass']
    assert StatType.CATEGORY_COUNTS in col_stats['Sex']
    assert StatType.CATEGORY_COUNTS in col_stats['Embarked']
    assert StatType.MEAN and StatType.STD in col_stats['Age']
    assert StatType.MEAN and StatType.STD in col_stats['SibSp']
    assert StatType.MEAN and StatType.STD in col_stats['Parch']
    assert StatType.MEAN and StatType.STD in col_stats['Fare']
