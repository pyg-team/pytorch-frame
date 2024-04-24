from __future__ import annotations

import os
import os.path as osp
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from PIL import Image

import torch_frame
from torch_frame import stype
from torch_frame.config.image_embedder import ImageEmbedderConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.typing import TaskType
from torch_frame.utils.split import SPLIT_TO_NUM

TIME_FORMATS = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']


def _random_timestamp(start: datetime, end: datetime, format: str) -> str:
    r"""This function will return a random datetime converted to string with
    given format between the start and end datetime objects.
    """
    timestamp = start + timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )
    return timestamp.strftime(format)


def _generate_random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    random_string = ''.join(
        random.choice(string.ascii_letters) for _ in range(length))
    return random_string


class FakeDataset(torch_frame.data.Dataset):
    r"""A fake dataset for testing purpose.

    Args:
        num_rows (int): Number of rows.
        with_nan (bool): Whether include nan in the dataset.
        stypes (List[stype]): List of stype columns to include
                in the dataset. Particularly useful, when you want to
                create a dataset with only numerical or categorical
                feature columns. (default: [stype.categorical,
                stype.numerical])
        create_split (bool): Whether to create a train, val and test
                split for the fake dataset. (default: :obj:`False`)
        task_type (TaskType): Task type (default: :obj:`TaskType.REGRESSION`)
        tmp_path (str, optional): Temporary path to save created images.
    """
    def __init__(
        self,
        num_rows: int,
        with_nan: bool = False,
        stypes: list[stype] = [stype.categorical, stype.numerical],
        create_split: bool = False,
        task_type: TaskType = TaskType.REGRESSION,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
        col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
        | TextTokenizerConfig | None = None,
        col_to_image_embedder_cfg: dict[str, ImageEmbedderConfig]
        | ImageEmbedderConfig | None = None,
        tmp_path: str | None = None,
    ) -> None:
        assert len(stypes) > 0
        df_dict: dict[str, list | np.ndarray]
        arr: list | np.ndarray
        if task_type == TaskType.REGRESSION:
            arr = np.random.randn(num_rows)
            if with_nan:
                arr[0::2] = np.nan
            df_dict = {'target': np.random.randn(num_rows)}
            col_to_stype = {'target': stype.numerical}
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            labels = np.random.randint(0, 3, size=(num_rows, ))
            if num_rows < 3:
                raise ValueError("Number of rows needs to be at "
                                 "least 3 for multiclass classification")
            # make sure every label exists
            labels[0] = 0
            labels[1] = 1
            labels[2] = 2
            df_dict = {'target': labels}
            col_to_stype = {'target': stype.categorical}
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            labels = np.random.randint(0, 2, size=(num_rows, ))
            if num_rows < 2:
                raise ValueError("Number of rows needs to be at "
                                 "least 2 for binary classification")
            labels[0] = 0
            labels[1] = 1
            df_dict = {'target': labels}
            col_to_stype = {'target': stype.categorical}
        else:
            raise ValueError(
                "FakeDataset only support binary classification, "
                "multiclass classification or regression type, but"
                f" got {task_type}")
        if stype.numerical in stypes:
            for col_name in ['num_1', 'num_2', 'num_3']:
                arr = np.random.randn(num_rows)
                if with_nan:
                    arr[0::2] = np.nan
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.numerical
        if stype.categorical in stypes:
            for col_name in ['cat_1', 'cat_2']:
                arr = np.random.randint(0, 3, size=(num_rows, ))
                if with_nan:
                    arr = arr.astype(np.float32)
                    arr[1::2] = np.nan
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.categorical
        if stype.multicategorical in stypes:
            for col_name in [
                    'multicat_1', 'multicat_2', 'multicat_3', 'multicat_4'
            ]:
                vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
                arr = []
                for _ in range(num_rows):
                    sampled = random.sample(vocab, 3)
                    if col_name in ['multicat_1', 'multicat_2']:
                        arr.append(','.join(sampled))
                    else:
                        arr.append(sampled)
                if with_nan:
                    arr[0] = None
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.multicategorical
        if stype.sequence_numerical in stypes:
            for col_name in ['seq_num_1', 'seq_num_2']:
                arr = []
                for i in range(num_rows):
                    sequence_length = random.randint(1, 5)
                    sequence = [
                        random.random() for _ in range(sequence_length)
                    ]
                    nan_idx = random.randint(0, sequence_length - 1)
                    sequence[nan_idx] = np.nan
                    arr.append(sequence)
                df_dict[col_name] = arr
                if with_nan:
                    df_dict[col_name][0] = None
                col_to_stype[col_name] = stype.sequence_numerical
        if stype.text_embedded in stypes:
            for col_name in ['text_embedded_1', 'text_embedded_2']:
                arr = [
                    ' '.join([
                        _generate_random_string(5, 15),
                        _generate_random_string(5, 15)
                    ]) for _ in range(num_rows)
                ]
                if with_nan:
                    arr[0::2] = len(arr[0::2]) * [np.nan]
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.text_embedded
        if stype.text_tokenized in stypes:
            for col_name in ['text_tokenized_1', 'text_tokenized_2']:
                arr = [
                    ' '.join([
                        _generate_random_string(5, 15),
                        _generate_random_string(5, 15)
                    ]) for _ in range(num_rows)
                ]
                if with_nan:
                    arr[0::2] = len(arr[0::2]) * [np.nan]
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.text_tokenized
        if stype.embedding in stypes:
            for col_name in ['emb_1', 'emb_2']:
                emb_dim = random.randint(1, 5)
                emb = [random.random() for _ in range(emb_dim)]
                embs = [emb for _ in range(num_rows)]
                df_dict[col_name] = embs
                col_to_stype[col_name] = stype.embedding
        if stype.timestamp in stypes:
            start_date = datetime(2000, 1, 1)
            end_date = datetime(2023, 1, 1)
            for i in range(len(TIME_FORMATS)):
                col_name = f'timestamp_{i}'
                format = TIME_FORMATS[i]
                arr = [
                    _random_timestamp(start_date, end_date, format)
                    for _ in range(num_rows)
                ]
                if with_nan:
                    arr[0::2] = len(arr[0::2]) * [np.nan]
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.timestamp
        if stype.image_embedded in stypes:
            assert tmp_path is not None
            for col_name in ['image_embedded_1', 'image_embedded_2']:
                arr = []
                os.makedirs(osp.join(tmp_path, col_name), exist_ok=True)
                for i in range(num_rows):
                    img_path = osp.join(tmp_path, col_name, f'{i}.png')
                    img = Image.new('RGB', (24, 24))
                    img.save(img_path)
                    img.close()
                    arr.append(img_path)
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.image_embedded

        df = pd.DataFrame(df_dict)
        if create_split:
            # TODO: Instead of having a split column name with train, val and
            # test, we will implement `random_split` and `split_by_col`
            # function in the Dataset class. We will modify the following lines
            # when the functions are introduced.
            if num_rows < 3:
                raise ValueError("Dataframe needs at least 3 rows to include"
                                 " each of train, val and test split.")
            split = [SPLIT_TO_NUM['train']] * num_rows
            split[1] = SPLIT_TO_NUM['val']
            split[2] = SPLIT_TO_NUM['test']
            df['split'] = split

        super().__init__(
            df,
            col_to_stype,
            target_col='target',
            split_col='split' if create_split else None,
            col_to_sep={
                'multicat_1': ',',
                'multicat_2': ',',
            },
            col_to_text_embedder_cfg=col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
            col_to_time_format={
                f'timestamp_{i}': TIME_FORMATS[i]
                for i in range(len(TIME_FORMATS))
            },
            col_to_image_embedder_cfg=col_to_image_embedder_cfg,
        )
