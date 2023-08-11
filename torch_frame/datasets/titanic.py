import pandas as pd

import torch_frame


class Titanic(torch_frame.data.Dataset):
    url = 'https://github.com/datasciencedojo/datasets/raw/master/titanic.csv'

    def __init__(self, root: str):
        path = self.download(self.url, root)
        df = pd.read_csv(path, index_col=['PassengerId'])

        stypes = {  # TODO Use 'Name', 'Ticket' and 'Cabin'.
            'Survived': torch_frame.categorical,
            'Pclass': torch_frame.categorical,
            'Sex': torch_frame.categorical,
            'Age': torch_frame.numerical,
            'SibSp': torch_frame.numerical,
            'Parch': torch_frame.numerical,
            'Fare': torch_frame.numerical,
            'Embarked': torch_frame.categorical,
        }

        super().__init__(df, stypes, target_col='Survived')
