from torch_frame.datasets import Titanic


def test_titanic(tmp_path):
    dataset = Titanic(tmp_path)
    assert str(dataset) == 'Titanic()'
    assert len(dataset) == 891
    assert dataset.columns == [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
    ]
