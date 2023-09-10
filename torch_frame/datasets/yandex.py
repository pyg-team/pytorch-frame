import os.path as osp
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch_frame
from torch_frame import TaskType


def load_numpy_dict(path: str) -> Dict[str, np.ndarray]:
    r"""Load numpy files from a ZIP file.

    Args:
        path (str): A path to the ZIP file containing .npy files.

    Returns:
        numpy_dict (Dict[str, np.ndarray]): A dictionary that maps the name of
            .npy file to the loaded numpy array.
    """
    numpy_dict: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.npy'):
                with zip_ref.open(file_name) as f:
                    array_name = osp.basename(file_name).replace('.npy', '')
                    numpy_dict[array_name] = np.load(f, allow_pickle=True)
    return numpy_dict


def get_df_and_col_to_stype(
        zip_file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""Get DataFrame and :obj:`col_to_stype` from a ZIP file.

    Args:
        zip_file_path (str): A path of the ZIP file containing .npy files.

    Returns:
        df (DataFrame): DataFrame containing train/val/test rows.
        col_to_stype (Dict[str, torch_frame.stype]). A dictionary mapping
            column names to their respective semantic types.
    """
    numpy_dict = load_numpy_dict(zip_file_path)
    dataframes: List[pd.DataFrame] = []
    col_to_stype: Dict[str, torch_frame.stype] = {}

    for split in ['train', 'val', 'test']:
        categorical_features = numpy_dict.get(f'C_{split}', None)
        numerical_features = numpy_dict.get(f'N_{split}', None)
        labels = numpy_dict[f'y_{split}']
        assert not ((categorical_features is None) and
                    (numerical_features is None))

        features: Optional[np.ndarray] = None
        if (categorical_features is not None
                and numerical_features is not None):
            features = np.concatenate(
                [categorical_features, numerical_features], axis=1)
            c_col_names = [
                f'C_feature_{i}' for i in range(categorical_features.shape[1])
            ]
            n_col_names = [
                f'N_feature_{i}' for i in range(numerical_features.shape[1])
            ]
            col_names = c_col_names + n_col_names

            for name in c_col_names:
                col_to_stype[name] = torch_frame.categorical
            for name in n_col_names:
                col_to_stype[name] = torch_frame.numerical
        else:
            # if the numpy_dict contains only categorical or numerical features
            features = (categorical_features if categorical_features
                        is not None else numerical_features)
            feature_type = 'C' if categorical_features is not None else 'N'
            col_names = [
                f'{feature_type}_feature_{i}' for i in range(features.shape[1])
            ]
            if feature_type == 'N':
                n_col_names = col_names
            for name in col_names:
                col_to_stype[name] = (torch_frame.categorical if feature_type
                                      == 'C' else torch_frame.numerical)
        assert features is not None
        df = pd.DataFrame(features, columns=col_names)
        # Explicitly set dtype for numerical features
        if numerical_features is not None:
            for n_col in n_col_names:
                df[n_col] = df[n_col].astype('float64')
        df['label'] = labels
        # Stores the split information in "split" column.
        df['split'] = split
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    return df, col_to_stype


class Yandex(torch_frame.data.Dataset):
    r"""The Yandex dataset collections used by https://arxiv.org/abs/2106.11959
    Originally downloaded from
    https://github.com/yandex-research/tabular-dl-revisiting-models
    """

    base_url = 'https://data.pyg.org/datasets/tables/revisiting_data/'
    classification_datasets = {
        'adult', 'aloi', 'covtype', 'helena', 'higgs_small', 'jannis'
    }
    regression_datasets = {'california_housing', 'microsoft', 'yahoo', 'year'}

    def __init__(self, root: str, name: str):
        assert name in self.classification_datasets | self.regression_datasets
        self.root = root
        self.name = name
        path = self.download_url(osp.join(self.base_url, self.name + '.zip'),
                                 root)
        df, col_to_stype = get_df_and_col_to_stype(path)
        if name in self.regression_datasets:
            col_to_stype['label'] = torch_frame.numerical
        else:
            col_to_stype['label'] = torch_frame.categorical
        super().__init__(df, col_to_stype, target_col='label',
                         split_col='split')

    @property
    def task_type(self) -> TaskType:
        r"""
        Returns:
            task_type (TaskType): The task type of the current dataset.
        """
        if self.name in self.regression_datasets:
            self.task_type = TaskType.REGRESSION
        else:
            if self.num_classes > 2:
                self.task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                self.task_type = TaskType.BINARY_CLASSIFICATION
        return self.task_type
