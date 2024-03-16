# flake8: noqa

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.typing import TaskType
from torch_frame.utils import generate_random_split
from torch_frame.utils.split import SPLIT_TO_NUM

SPLIT_COL = 'split'


class DataFrameTextBenchmark(torch_frame.data.Dataset):
    r"""A collection of datasets for tabular learning with text columns,
    covering categorical, numerical, multi-categorical and timestamp
    features. The datasets are categorized according to their task types
    and scales.

    Args:
        root (str): Root directory.
        task_type (TaskType): The task type. Either
            :obj:`TaskType.BINARY_CLASSIFICATION`,
            :obj:`TaskType.MULTICLASS_CLASSIFICATION`, or
            :obj:`TaskType.REGRESSION`
        scale (str): The scale of the dataset. :obj:`"small"` means 5K to 50K
            rows. :obj:`"medium"` means 50K to 500K rows. :obj:`"large"`
            means more than 500K rows.
        text_stype (torch_frame.stype): Text stype to use for text columns
            in the dataset. (default: :obj:`torch_frame.text_embedded`).
        idx (int): The index of the dataset within a category specified via
            :obj:`task_type` and :obj:`scale`.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10 10 10 20 20 10
        :header-rows: 1

        * - Task
          - Scale
          - Idx
          - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #cols (text)
          - #cols (other)
          - #classes
          - Class object
          - Missing value ratio
        * - binary_classification
          - small
          - 0
          - 15,907
          - 0
          - 3
          - 2
          - 0
          - 2
          - MultimodalTextBenchmark(name='fake_job_postings2')
          - 23.8%
        * - binary_classification
          - medium
          - 0
          - 125,000
          - 29
          - 0
          - 1
          - 0
          - 2
          - MultimodalTextBenchmark(name='jigsaw_unintended_bias100K')
          - 41.4%
        * - binary_classification
          - medium
          - 1
          - 108,128
          - 1
          - 3
          - 3
          - 2
          - 2
          - MultimodalTextBenchmark(name='kick_starter_funding')
          - 0.0%
        * - multiclass_classification
          - small
          - 0
          - 6,364
          - 0
          - 1
          - 1
          - 0
          - 4
          - MultimodalTextBenchmark(name='product_sentiment_machine_hack')
          - 0.0%
        * - multiclass_classification
          - small
          - 1
          - 25,355
          - 14
          - 0
          - 1
          - 0
          - 6
          - MultimodalTextBenchmark(name='news_channel')
          - 0.0%
        * - multiclass_classification
          - small
          - 2
          - 19,802
          - 0
          - 3
          - 2
          - 1
          - 6
          - MultimodalTextBenchmark(name='data_scientist_salary')
          - 12.3%
        * - multiclass_classification
          - small
          - 3
          - 22,895
          - 26
          - 47
          - 13
          - 3
          - 10
          - MultimodalTextBenchmark(name='melbourne_airbnb')
          - 9.6%
        * - multiclass_classification
          - medium
          - 0
          - 105,154
          - 2
          - 2
          - 1
          - 0
          - 30
          - MultimodalTextBenchmark(name='wine_reviews')
          - 1.0%
        * - multiclass_classification
          - medium
          - 1
          - 114,000
          - 11
          - 5
          - 3
          - 0
          - 114
          - HuggingFaceDatasetDict(path='maharshipandya/spotify-tracks-dataset', target_col='track_genre')
          - 0.0%
        * - multiclass_classification
          - large
          - 0
          - 568,454
          - 2
          - 3
          - 2
          - 0
          - 5
          - AmazonFineFoodReviews()
          - 0.0%
        * - regression
          - small
          - 0
          - 6,079
          - 0
          - 1
          - 3
          - 0
          - 1
          - MultimodalTextBenchmark(name='google_qa_answer_type_reason_explanation')
          - 0.0%
        * - regression
          - small
          - 1
          - 6,079
          - 0
          - 1
          - 3
          - 0
          - 1
          - MultimodalTextBenchmark(name='google_qa_question_type_reason_explanation')
          - 0.0%
        * - regression
          - small
          - 2
          - 6,237
          - 2
          - 3
          - 3
          - 0
          - 1
          - MultimodalTextBenchmark(name='bookprice_prediction')
          - 1.7%
        * - regression
          - small
          - 3
          - 13,575
          - 2
          - 1
          - 2
          - 0
          - 1
          - MultimodalTextBenchmark(name='jc_penney_products')
          - 13.7%
        * - regression
          - small
          - 4
          - 23,486
          - 1
          - 3
          - 2
          - 0
          - 1
          - MultimodalTextBenchmark(name='women_clothing_review')
          - 1.8%
        * - regression
          - small
          - 5
          - 30,009
          - 3
          - 0
          - 1
          - 0
          - 1
          - MultimodalTextBenchmark(name='news_popularity2')
          - 0.0%
        * - regression
          - small
          - 6
          - 28,328
          - 2
          - 5
          - 1
          - 3
          - 1
          - MultimodalTextBenchmark(name='ae_price_prediction')
          - 6.1%
        * - regression
          - small
          - 7
          - 47,439
          - 18
          - 8
          - 2
          - 11
          - 1
          - MultimodalTextBenchmark(name='california_house_price')
          - 13.8%
        * - regression
          - medium
          - 0
          - 125,000
          - 0
          - 6
          - 2
          - 1
          - 1
          - MultimodalTextBenchmark(name='mercari_price_suggestion100K')
          - 3.4%
        * - regression
          - large
          - 0
          - 1,482,535
          - 1
          - 4
          - 2
          - 1
          - 1
          - Mercari()
          - 0.0%
    """
    dataset_categorization_dict: dict[str, dict[str, list[tuple]]] = {
        'binary_classification': {
            'small': [
                ('MultimodalTextBenchmark', {
                    'name': 'fake_job_postings2'
                }),
            ],
            'medium': [
                ('MultimodalTextBenchmark', {
                    'name': 'jigsaw_unintended_bias100K'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'kick_starter_funding'
                }),
            ],
            'large': [],
        },
        'multiclass_classification': {
            'small': [
                ('MultimodalTextBenchmark', {
                    'name': 'product_sentiment_machine_hack'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'news_channel'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'data_scientist_salary'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'melbourne_airbnb'
                }),
            ],
            'medium': [
                ('MultimodalTextBenchmark', {
                    'name': 'wine_reviews'
                }),
                ('HuggingFaceDatasetDict', {
                    'path':
                    'maharshipandya/spotify-tracks-dataset',
                    'columns': [
                        'artists', 'album_name', 'track_name', 'popularity',
                        'duration_ms', 'explicit', 'danceability', 'energy',
                        'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo', 'time_signature', 'track_genre'
                    ],
                    'target_col':
                    'track_genre',
                }),
            ],
            'large': [
                ('AmazonFineFoodReviews', {}),
            ],
        },
        'regression': {
            'small': [
                ('MultimodalTextBenchmark', {
                    'name': 'google_qa_answer_type_reason_explanation'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'google_qa_question_type_reason_explanation'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'bookprice_prediction'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'jc_penney_products'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'women_clothing_review'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'news_popularity2'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'ae_price_prediction'
                }),
                ('MultimodalTextBenchmark', {
                    'name': 'california_house_price'
                }),
            ],
            'medium': [
                ('MultimodalTextBenchmark', {
                    'name': 'mercari_price_suggestion100K'
                }),
            ],
            'large': [
                ('Mercari', {}),
            ],
        }
    }

    @classmethod
    def datasets_available(
        cls,
        task_type: TaskType,
        scale: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        r"""List of datasets available for a given :obj:`task_type` and
        :obj:`scale`.
        """
        return cls.dataset_categorization_dict[task_type.value][scale]

    @classmethod
    def num_datasets_available(cls, task_type: TaskType, scale: str):
        r"""Number of datasets available for a given :obj:`task_type` and
        :obj:`scale`.
        """
        return len(cls.datasets_available(task_type, scale))

    def __init__(
        self,
        root: str,
        task_type: TaskType,
        scale: str,
        idx: int,
        text_stype: torch_frame.stype = torch_frame.text_embedded,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
        col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
        | TextTokenizerConfig | None = None,
        split_random_state: int = 42,
    ):
        self.root = root
        self._task_type = task_type
        self.scale = scale
        self.idx = idx

        datasets = self.datasets_available(task_type, scale)
        if idx >= len(datasets):
            raise ValueError(
                f"The idx needs to be smaller than {len(datasets)}, which is "
                f"the number of available datasets for task_type: "
                f"{task_type.value} and scale: {scale} (got idx: {idx}).")

        class_name, kwargs = self.datasets_available(task_type, scale)[idx]
        if class_name in {'MultimodalTextBenchmark', 'AmazonFineFoodReviews'}:
            text_args = dict(
                text_stype=text_stype,
                col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg)
        elif class_name == 'HuggingFaceDatasetDict':
            # TODO (zecheng): support text tokenized
            text_args = dict(col_to_text_embedder_cfg=col_to_text_embedder_cfg)
        else:
            text_args = dict(col_to_text_embedder_cfg=col_to_text_embedder_cfg)
        if class_name == 'HuggingFaceDatasetDict':
            # HuggingFace will handle the data download so don't need the root
            dataset = getattr(torch_frame.datasets, class_name)(**text_args,
                                                                **kwargs)
        else:
            dataset = getattr(torch_frame.datasets,
                              class_name)(root=root, **text_args, **kwargs)
        self.cls_str = str(dataset)

        # Add split col
        df = dataset.df
        # Follow default split for datasets of MultimodalTextBenchmark:
        if class_name == 'MultimodalTextBenchmark':
            df = df.sort_values(by=[SPLIT_COL])
            num_unique = df[SPLIT_COL].nunique()
            assert num_unique > 1
            # Manually split validation set from the train one:
            if num_unique == 2:
                ser = df[SPLIT_COL]
                train_ser = ser[ser == SPLIT_TO_NUM['train']]
                split_ser = generate_random_split(length=len(train_ser),
                                                  seed=split_random_state,
                                                  train_ratio=0.9,
                                                  val_ratio=0.1,
                                                  include_test=False)
                split_ser = np.concatenate([
                    split_ser,
                    np.full(len(df) - len(split_ser), SPLIT_TO_NUM['test'])
                ])
                df[SPLIT_COL] = split_ser
        else:
            if SPLIT_COL in df.columns:
                df.drop(columns=[SPLIT_COL], inplace=True)
            split_df = pd.DataFrame({
                SPLIT_COL:
                generate_random_split(length=len(df), seed=split_random_state,
                                      train_ratio=0.8, val_ratio=0.1)
            })
            df = pd.concat([df, split_df], axis=1)

        # For regression task, we normalize the target.
        if task_type == TaskType.REGRESSION:
            ser = df[dataset.target_col]
            df[dataset.target_col] = (ser - ser.mean()) / ser.std()

        # Check the scale
        if dataset.num_rows < 5000:
            assert False
        elif dataset.num_rows < 50000:
            assert scale == "small"
        elif dataset.num_rows < 500000:
            assert scale == "medium"
        else:
            assert scale == "large"

        super().__init__(
            df,
            dataset.col_to_stype,
            target_col=dataset.target_col,
            split_col=SPLIT_COL,
            col_to_sep=dataset.col_to_sep,
            col_to_text_embedder_cfg=dataset.col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=dataset.col_to_text_tokenizer_cfg,
        )
        del dataset

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  task_type={self._task_type.value},\n'
                f'  scale={self.scale},\n'
                f'  idx={self.idx},\n'
                f'  cls={self.cls_str}\n'
                f')')

    def materialize(self, *args, **kwargs) -> torch_frame.data.Dataset:
        super().materialize(*args, **kwargs)
        if self.task_type != self._task_type:
            raise RuntimeError(f"task type does not match. It should be "
                               f"{self.task_type.value} but specified as "
                               f"{self._task_type.value}.")
        return self
