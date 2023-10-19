import os.path as osp

import pandas as pd

import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM


class Mercari(torch_frame.data.Dataset):
    base_url = 'https://data.pyg.org/datasets/tables/mercari_price_suggestion/'
    files = ['train', 'test', 'test_stg2']

    def __init__(self, root: str):
        self.dfs = dict()
        col_to_stype = {
            'name': torch_frame.text_embedded,
            'item_condition_id': torch_frame.categorical,
            'category_name': torch_frame.categorical,
            'brand_name': torch_frame.categorical,
            'price': torch_frame.numerical,
            'shipping': torch_frame.categorical,
            'item_description': torch_frame.text_embedded
        }
        for file in self.files:
            if file == 'test':
                split = 'val'
            elif file == 'test_stg2':
                split = 'test'
            else:
                split = 'train'
            self.dfs[split] = pd.read_csv(
                self.download_url(osp.join(self.base_url, file + '.csv'),
                                  root))
        df = pd.concat(self.dfs.values(), keys=self.dfs.keys(),
                       names=['split']).reset_index(level=0)
        df['split'] = df['split'].map(SPLIT_TO_NUM)
        df.drop(['train_id', 'test_id'], axis=1, inplace=True)
        super().__init__(df, col_to_stype, target_col='price',
                         split_col='split')
