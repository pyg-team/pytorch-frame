import os.path as osp
import zipfile

import pandas as pd

import torch_frame


class Amphibians(torch_frame.data.Dataset):
    r"""The Amphibians
    <https://archive.ics.uci.edu/dataset/528/amphibians>`_
    dataset. The task is to predict which of the 7 frogs types appeared
    in the habitat.

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
        * - 189
          - 3
          - 20
          - 0
          - multilabel classification
          - 0.0%
    """
    url = 'https://archive.ics.uci.edu/static/public/528/amphibians.zip'

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        folder_path = osp.dirname(path)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        data_path = osp.join(folder_path, 'dataset.csv')
        names = [
            'ID', 'MV', 'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR',
            'FR', 'OR', 'RR', 'BR', 'MR', 'CR', 't1', 't2', 't3', 't4', 't5',
            't6', 't7'
        ]
        df = pd.read_csv(data_path, names=names, sep=';')
        # Drop the first 2 rows containing metadata
        df = df.iloc[2:].reset_index(drop=True)
        target_cols = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
        df['t'] = df.apply(
            lambda row: [col for col in target_cols if row[col] == '1'],
            axis=1)
        df = df.drop(target_cols, axis=1)

        # Infer the pandas dataframe automatically
        path = osp.join(root, 'amphibians_posprocess.csv')
        df.to_csv(path, index=False)
        df = pd.read_csv(path)

        col_to_stype = {
            'ID': torch_frame.numerical,
            'MV': torch_frame.categorical,
            'SR': torch_frame.numerical,
            'NR': torch_frame.numerical,
            'TR': torch_frame.categorical,
            'VR': torch_frame.categorical,
            'SUR1': torch_frame.categorical,
            'SUR2': torch_frame.categorical,
            'SUR3': torch_frame.categorical,
            'UR': torch_frame.categorical,
            'FR': torch_frame.categorical,
            'OR': torch_frame.numerical,
            'RR': torch_frame.categorical,  # Support Ordinal Encoding
            'BR': torch_frame.categorical,  # Support Ordinal Encoding
            'MR': torch_frame.categorical,
            'CR': torch_frame.categorical,
            't': torch_frame.multicategorical,
        }
        super().__init__(df, col_to_stype, target_col='t', col_to_sep=None)
