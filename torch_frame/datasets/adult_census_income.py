import pandas as pd

import torch_frame


class AdultCensusIncome(torch_frame.data.Dataset):
    r"""The `Adult Census Income
    <https://www.kaggle.com/datasets/uciml/adult-census-income>`_
    dataset. It's extracted from census bureau database and the task
    is to predict whether a person's income exceeds $50K/yr."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'  # noqa

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        names = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education.num',
            'marital.status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital.gain',
            'capital.loss',
            'hours.per.week',
            'native.country',
            'income',
        ]
        df = pd.read_csv(path, names=names)

        stypes = {
            'age': torch_frame.numerical,
            'workclass': torch_frame.categorical,
            'education': torch_frame.categorical,
            'marital.status': torch_frame.categorical,
            'occupation': torch_frame.categorical,
            'relationship': torch_frame.categorical,
            'race': torch_frame.categorical,
            'sex': torch_frame.categorical,
            'capital.gain': torch_frame.numerical,
            'capital.loss': torch_frame.numerical,
            'hours.per.week': torch_frame.numerical,
            'native.country': torch_frame.categorical,
            'income': torch_frame.categorical,
        }

        super().__init__(df, stypes, target_col='income')
