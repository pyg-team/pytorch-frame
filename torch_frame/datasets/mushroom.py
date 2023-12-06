import os.path as osp
import zipfile

import pandas as pd

import torch_frame


class Mushroom(torch_frame.data.Dataset):
    r"""The `Mushroom classification Kaggle competition
    <https://www.kaggle.com/datasets/uciml/mushroom-classification>`_
    dataset. It's a task to predict whether a mushroom is edible
    or poisonous.

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
        * - 8,124
          - 0
          - 22
          - 2
          - binary_classification
          - 0.0%
    """

    url = 'http://archive.ics.uci.edu/static/public/73/mushroom.zip'

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        folder_path = osp.dirname(path)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        data_path = osp.join(folder_path, 'agaricus-lepiota.data')

        names = [
            'class',
            'cap-shape',
            'cap-surface',
            'cap-color',
            'bruises',
            'odor',
            'gill-attachment',
            'gill-spacing',
            'gill-size',
            'gill-color',
            'stalk-shape',
            'stalk-root',
            'stalk-surface-above-ring',
            'stalk-surface-below-ring',
            'stalk-color-above-ring',
            'stalk-color-below-ring',
            'veil-type',
            'veil-color',
            'ring-number',
            'ring-type',
            'spore-print-color',
            'population',
            'habitat',
        ]
        df = pd.read_csv(data_path, names=names)

        col_to_stype = {
            'class': torch_frame.categorical,
            'cap-shape': torch_frame.categorical,
            'cap-surface': torch_frame.categorical,
            'cap-color': torch_frame.categorical,
            'bruises': torch_frame.categorical,
            'odor': torch_frame.categorical,
            'gill-attachment': torch_frame.categorical,
            'gill-spacing': torch_frame.categorical,
            'gill-size': torch_frame.categorical,
            'gill-color': torch_frame.categorical,
            'stalk-shape': torch_frame.categorical,
            'stalk-root': torch_frame.categorical,
            'stalk-surface-above-ring': torch_frame.categorical,
            'stalk-surface-below-ring': torch_frame.categorical,
            'stalk-color-above-ring': torch_frame.categorical,
            'stalk-color-below-ring': torch_frame.categorical,
            'veil-type': torch_frame.categorical,
            'veil-color': torch_frame.categorical,
            'ring-number': torch_frame.categorical,
            'ring-type': torch_frame.categorical,
            'spore-print-color': torch_frame.categorical,
            'population': torch_frame.categorical,
            'habitat': torch_frame.categorical,
        }

        super().__init__(df, col_to_stype, target_col='class')
