import os.path as osp
import pandas as pd
import zipfile

import torch_frame


class Dota2(torch_frame.data.Dataset):
    r"""The `Dota2 Game Results
    <https://archive.ics.uci.edu/dataset/367/dota2+games+results>`_
    dataset. Dota2 is a popula moba game with two teams of 5 players.
    At start of the game, each palyer choose a unique hero with
    different strengths and weakness. The dataset is reasonably sparse
    as only 10 of 113 possible heros are chosen in a given game. All
    games were played in a space of 2 hours on the 13th of August, 2016.
    The classification goal is to predict the winning team.
    """

    url = 'https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip'  # noqa

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        names = [
            'Team won the game',
            'Cluster ID',
            'Game mode',
            'Game type',
        ]
        num_heros = 113
        names += [f'hero_{i}' for i in range(num_heros)]
        folder_path = osp.dirname(path)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        df = pd.read_csv(osp.join(folder_path, 'dota2Train.csv'), names=names)

        col_to_stype = {
            'Team won the game': torch_frame.categorical,
            'Cluster ID': torch_frame.categorical,
            'Game mode': torch_frame.categorical,
            'Game type': torch_frame.categorical,
        }
        for i in range(num_heros):
            col_to_stype[f'hero_{i}'] = torch_frame.categorical

        super().__init__(df, col_to_stype, target_col='Team won the game')
