import pandas as pd

import torch_frame


class AdultCensusIncome(torch_frame.data.Dataset):
    r"""The `Adult Census Income
    <https://www.kaggle.com/datasets/uciml/adult-census-income>`_
    dataset from Kaggle. It's extracted from census bureau database and the
    task is to predict whether a person's income exceeds $50K/year.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Task
          - Missing value ratio
        * - 32,561
          - 4
          - 8
          - 2
          - binary_classification
          - 0.0%
    """

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

        col_to_stype = {
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

        super().__init__(df, col_to_stype, target_col='income')
