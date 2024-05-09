from __future__ import annotations

import os.path as osp
import zipfile

import pandas as pd

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.utils.split import SPLIT_TO_NUM

SPLIT_COL = 'split_col'


class Movielens1M(torch_frame.data.Dataset):
    r"""The MovieLens 1M rating dataset, assembled by GroupLens Research
    from the MovieLens web site, consisting of movies (3,883 nodes) and
    users (6,040 nodes) with approximately 1 million ratings between them.
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
        num_size = df.shape[0]
        train_size, val_size = int(num_size * 0.8), int(num_size * 0.1)
        df.loc[:train_size, SPLIT_COL] = SPLIT_TO_NUM['train']
        df.loc[train_size:train_size + val_size,
               SPLIT_COL] = SPLIT_TO_NUM['val']
        df.loc[train_size + val_size:, SPLIT_COL] = SPLIT_TO_NUM['test']
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
                         col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                         split_col=SPLIT_COL)


if __name__ == '__main__':
    import torch
    from sentence_transformers import SentenceTransformer
    from torch import Tensor

    class PretrainedTextEncoder:
        def __init__(self, device: torch.device) -> None:
            self.model = SentenceTransformer("all-distilroberta-v1",
                                             device=device)

        def __call__(self, sentences: list[str]) -> Tensor:
            # Inference on GPU (if available)
            embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                           convert_to_tensor=True)
            # Map back to CPU
            return embeddings.cpu()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    text_embedder = PretrainedTextEncoder(device=device)
    dataset = Movielens1M(
        './data',
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=text_embedder, batch_size=10240),
    )
    dataset.materialize(path='./data/ml-1m/data.pt')
    import pdb
    pdb.set_trace()
