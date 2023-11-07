from typing import Optional

import pandas as pd

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig


class AmazonFineFoodReviews(torch_frame.data.Dataset):
    r"""The `Amazon Fine Food Reviews <https://arxiv.org/abs/1303.4402>`_
    dataset. It consists of reviews of fine foods from amazon.

    Args:
        text_stype (torch_frame.stype): Text stype to use for text columns
            in the dataset. (default: :obj:`torch_frame.text_embedded`)
        text_embedder_cfg (TextEmbedderConfig, optional): Text embedder
            config to use if :obj:`text_stype` is specified as
            `torch_frame.text_embedded`.
        text_tokenizer_cfg (TextTokenizerConfig, optional): Text tokenizer
            config to use if :obj:`text_stype` is specified as
            `torch_frame.text_tokenized`.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #cols (text)
          - #classes
          - Task
          - Missing value ratio
        * - 568,454
          - 2
          - 3
          - 2
          - 5
          - multiclass_classification
          - 0.0%
    """

    url = "https://data.pyg.org/datasets/tables/amazon_fine_food_reviews.zip"

    def __init__(self, root: str,
                 text_stype: torch_frame.stype = torch_frame.text_embedded,
                 text_embedder_cfg: Optional[TextEmbedderConfig] = None,
                 text_tokenizer_cfg: Optional[TextTokenizerConfig] = None):
        self.root = root
        self.text_stype = text_stype
        path = self.download_url(self.url, root)

        col_to_stype = {
            'ProductId': torch_frame.categorical,
            'UserId': torch_frame.categorical,
            'HelpfulnessNumerator': torch_frame.numerical,
            'HelpfulnessDenominator': torch_frame.numerical,
            'Score': torch_frame.categorical,
            # 'Time': torch_frame.categorical,  # TODO: change to timestamp
            'Summary': text_stype,
            'Text': text_stype,
        }

        df = pd.read_csv(path)[list(col_to_stype.keys())]

        super().__init__(df, col_to_stype, target_col='Score',
                         text_embedder_cfg=text_embedder_cfg,
                         text_tokenizer_cfg=text_tokenizer_cfg)
