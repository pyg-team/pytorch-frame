from __future__ import annotations

from typing import Callable, Tuple

import pandas as pd

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig


class QuoraQuestionPairs(torch_frame.data.Dataset):
    r"""The `Quora Question Pairs
    <https://www.kaggle.com/competitions/quora-question-pairs>`_
    dataset from Kaggle. Questions are extracted from Quora and the
    task is to identify duplicated questions.
    """

    url = ''  # noqa

    def __init__(
        self,
        root: str,
        text_stype: torch_frame.stype = torch_frame.text_embedded,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
        col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
        | TextTokenizerConfig | None = None,
        pre_transform: Callable[[pd.DataFrame], Tuple[pd.DataFrame,
                                                      dict[str,
                                                           torch_frame.stype]]]
        | None = None,
    ):
        # path = self.download_url(self.url, root)
        path = "/home/zecheng/code/pytorch-frame/train.csv.zip"
        cols = [
            'question1',
            'question2',
            'is_duplicate',
        ]
        df = pd.read_csv(path, usecols=cols)

        col_to_stype: dict[str, torch_frame.stype] = {
            'question1': text_stype,
            'question2': text_stype,
            'is_duplicate': torch_frame.categorical,
        }

        if pre_transform is not None:
            new_df, new_col_to_stype = pre_transform(df)
            df = pd.concat([df, new_df], axis=1)
            col_to_stype.update(new_col_to_stype)

        super().__init__(
            df,
            col_to_stype,
            target_col='is_duplicate',
            col_to_text_embedder_cfg=col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
        )
