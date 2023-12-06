import os.path as osp
import zipfile

import pandas as pd

import torch_frame


class PokerHand(torch_frame.data.Dataset):
    r"""The `Poker Hand
    <https://archive.ics.uci.edu/dataset/158/poker+hand>`_
    dataset. It's a task to predict 5-card poker hand.

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
        * - 1,025,010
          - 5
          - 5
          - 10
          - multiclass_classification
          - 0.0%
    """

    url = 'https://archive.ics.uci.edu/static/public/158/poker+hand.zip'

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        folder_path = osp.dirname(path)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        train_path = osp.join(folder_path, 'poker-hand-training-true.data')
        test_path = osp.join(folder_path, 'poker-hand-testing.data')

        names = [
            'Suit of card #1',
            'Rank of card #1',
            'Suit of card #2',
            'Rank of card #2',
            'Suit of card #3',
            'Rank of card #3',
            'Suit of card #4',
            'Rank of card #4',
            'Suit of card #5',
            'Rank of card #5',
            'Poker Hand',
        ]
        train_df = pd.read_csv(train_path, names=names)
        test_df = pd.read_csv(test_path, names=names)
        df = pd.concat([train_df, test_df], ignore_index=True)

        col_to_stype = {
            'Suit of card #1': torch_frame.categorical,
            'Rank of card #1': torch_frame.numerical,
            'Suit of card #2': torch_frame.categorical,
            'Rank of card #2': torch_frame.numerical,
            'Suit of card #3': torch_frame.categorical,
            'Rank of card #3': torch_frame.numerical,
            'Suit of card #4': torch_frame.categorical,
            'Rank of card #4': torch_frame.numerical,
            'Suit of card #5': torch_frame.categorical,
            'Rank of card #5': torch_frame.numerical,
            'Poker Hand': torch_frame.categorical,
        }

        super().__init__(df, col_to_stype, target_col='Poker Hand')
