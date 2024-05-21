from __future__ import annotations

import os.path as osp
import zipfile

import pandas as pd

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig


class Movielens1M(torch_frame.data.Dataset):
    r"""The MovieLens 1M rating dataset, assembled by GroupLens Research
    from the MovieLens web site, consisting of movies (3,883 nodes) and
    users (6,040 nodes) with approximately 1 million ratings between them.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 20
        :header-rows: 1

        * - #Users
          - #Items
          - #User Field
          - #Item Field
          - #Samples
        * - 6040
          - 3952
          - 5
          - 3
          - 1000209
    """

    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

    def __init__(
        self,
        root: str,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
    ):
        path = self.download_url(self.url, root)
        folder_path = osp.dirname(path)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        data_path = osp.join(folder_path, 'ml-1m')
        users = pd.read_csv(
            osp.join(data_path, 'users.dat'),
            header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zip'],
            sep='::',
            engine='python',
        )
        movies = pd.read_csv(
            osp.join(data_path, 'movies.dat'),
            header=None,
            names=['movie_id', 'title', 'genres'],
            sep='::',
            engine='python',
            encoding='ISO-8859-1',
        )
        ratings = pd.read_csv(
            osp.join(data_path, 'ratings.dat'),
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep='::',
            engine='python',
        )

        df = pd.merge(pd.merge(ratings, users), movies) \
               .sort_values(by='timestamp') \
               .reset_index().drop('index', axis=1)

        col_to_stype = {
            'user_id': torch_frame.categorical,
            'gender': torch_frame.categorical,
            'age': torch_frame.categorical,
            'occupation': torch_frame.categorical,
            'zip': torch_frame.categorical,
            'movie_id': torch_frame.categorical,
            'title': torch_frame.text_embedded,
            'genres': torch_frame.multicategorical,
            'rating': torch_frame.numerical,
            'timestamp': torch_frame.timestamp,
        }
        super().__init__(df, col_to_stype, target_col='rating', col_to_sep='|',
                         col_to_text_embedder_cfg=col_to_text_embedder_cfg)
