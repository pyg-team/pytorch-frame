import os.path as osp

import pandas as pd

import torch_frame


class Mercari(torch_frame.data.Dataset):
    r"""The `Mercari Price Suggestion Challenge
    <https://www.kaggle.com/c/mercari-price-suggestion-challenge/>`_
    dataset from Kaggle.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #cols (text_embedded)
          - Task
          - Missing value ratio
        * - 1,482,535
          - 1
          - 4
          - 2
          - regression
          - 0.0%
    """
    base_url = 'https://data.pyg.org/datasets/tables/mercari_price_suggestion/'
    files = ['train']

    def __init__(self, root: str):
        self.dfs = dict()
        col_to_stype = {
            'name': torch_frame.text_embedded,
            'item_condition_id': torch_frame.categorical,
            'category_name': torch_frame.categorical,
            'brand_name': torch_frame.categorical,
            'price': torch_frame.numerical,
            'shipping': torch_frame.categorical,
            'item_description': torch_frame.text_embedded,
        }
        path = osp.join(self.base_url, 'train.csv')
        self.download_url(path, root)
        df = pd.read_csv(path)
        df.drop(['train_id'], axis=1, inplace=True)
        super().__init__(df, col_to_stype, target_col='price')
