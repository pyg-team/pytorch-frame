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
    ):
        # path = self.download_url(self.url, root)
        path = "/Users/zecheng/code/pytorch-frame/train.csv.zip"
        names = [
            'question1',
            'question2',
            'is_duplicate',
        ]
        df = pd.read_csv(path, names=names)

        col_to_stype = {
            'question1': text_stype,
            'question2': text_stype,
            'is_duplicate': torch_frame.categorical,
        }

        super().__init__(
            df,
            col_to_stype,
            target_col='is_duplicate',
            col_to_text_embedder_cfg=col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
        )
