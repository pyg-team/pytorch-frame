from typing import List, Optional

import numpy as np
import pandas as pd

import torch_frame
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.typing import TaskType
from torch_frame.utils.split import SPLIT_TO_NUM


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
        text_embedder_cfg (TextEmbedderConfig, optional): A text embedder
            config specifying :obj:`text_embedder` that maps sentences into
            PyTorch embeddings and :obj:`batch_size` that specifies the
            mini-batch size for :obj:`text_embedder` (default: :obj:`None`)
    """
    def __init__(
        self,
        num_rows: int,
        with_nan: bool = False,
        stypes: List[stype] = [stype.categorical, stype.numerical],
        create_split: bool = False,
        task_type: TaskType = TaskType.REGRESSION,
        text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    ):
        assert len(stypes) > 0
        if task_type == TaskType.REGRESSION:
            df_dict = {'target': np.random.randn(num_rows)}
            col_to_stype = {'target': stype.numerical}
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            df_dict = {'target': np.random.randint(0, 3, size=(num_rows, ))}
            col_to_stype = {'target': stype.categorical}
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            df_dict = {'target': np.random.randint(0, 2, size=(num_rows, ))}
            col_to_stype = {'target': stype.categorical}
        else:
            raise ValueError(
                "FakeDataset only support binary classification, "
                "multiclass classification or regression type, but"
                f" got {task_type}")
        if stype.numerical in stypes:
            for col_name in ['a', 'b', 'c']:
                arr = np.random.randn(num_rows)
                if with_nan:
                    arr[0::2] = np.nan
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.numerical
        if stype.categorical in stypes:
            for col_name in ['x', 'y']:
                arr = np.random.randint(0, 3, size=(num_rows, ))
                if with_nan:
                    arr = arr.astype(np.float32)
                    arr[1::2] = np.nan
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.categorical
        if stype.text_embedded in stypes:
            for col_name in ['text_1', 'text_2']:
                arr = ['Hello world!'] * num_rows
                if with_nan:
                    arr[0::2] = len(arr[0::2]) * [np.nan]
                df_dict[col_name] = arr
                col_to_stype[col_name] = stype.text_embedded
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
            text_embedder_cfg=text_embedder_cfg,
        )
