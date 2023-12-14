from __future__ import annotations

import os.path as osp
import zipfile
from typing import Any

import numpy as np
import pandas as pd

import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM

SPLIT_COL = 'split_col'
TARGET_COL = 'target_col'


def load_numpy_dict(path: str) -> dict[str, np.ndarray]:
    r"""Load numpy files from a ZIP file.

    Args:
        path (str): A path to the ZIP file containing .npy files.

    Returns:
        numpy_dict (Dict[str, np.ndarray]): A dictionary that maps the name of
            .npy file to the loaded numpy array.
    """
    numpy_dict: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.npy'):
                with zip_ref.open(file_name) as f:
                    array_name = osp.basename(file_name).replace('.npy', '')
                    numpy_dict[array_name] = np.load(f, allow_pickle=True)
    return numpy_dict


def get_df_and_col_to_stype(
        zip_file_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    r"""Get DataFrame and :obj:`col_to_stype` from a ZIP file.

    Args:
        zip_file_path (str): A path of the ZIP file containing .npy files.

    Returns:
        df (DataFrame): DataFrame containing train/val/test rows.
        col_to_stype (Dict[str, torch_frame.stype]). A dictionary mapping
            column names to their respective semantic types.
    """
    numpy_dict = load_numpy_dict(zip_file_path)
    dataframes: list[pd.DataFrame] = []
    col_to_stype: dict[str, torch_frame.stype] = {}

    for split in ['train', 'val', 'test']:
        categorical_features = numpy_dict.get(f'C_{split}', None)
        numerical_features = numpy_dict.get(f'N_{split}', None)
        labels = numpy_dict[f'y_{split}']
        assert not ((categorical_features is None) and
                    (numerical_features is None))

        features: np.ndarray | None = None
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
            assert features is not None
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
        label_split_df = pd.DataFrame({
            TARGET_COL:
            labels,
            SPLIT_COL:
            np.full((len(df), ), fill_value=SPLIT_TO_NUM[split])
        })
        df = pd.concat([df, label_split_df], axis=1)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    return df, col_to_stype


class Yandex(torch_frame.data.Dataset):
    r"""The Yandex dataset collections used by `"Revisiting Deep Learning
    Models for Tabular Data" <https://arxiv.org/abs/2106.11959>`_.
    Originally downloaded from
    `github.com/yandex-research/tabular-dl-revisiting-models
    <https://github.com/yandex-research/tabular-dl-revisiting-models>`_.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 20 10
        :header-rows: 1

        * - Name
          - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Task
          - Missing value ratio
        * - adult
          - 48,842
          - 6
          - 8
          - 2
          - binary_classification
          - 0.0%
        * - aloi
          - 108,000
          - 128
          - 0
          - 1,000
          - multiclass_classification
          - 0.0%
        * - covtype
          - 581,012
          - 54
          - 0
          - 7
          - multiclass_classification
          - 0.0%
        * - helena
          - 65,196
          - 27
          - 0
          - 100
          - multiclass_classification
          - 0.0%
        * - higgs_small
          - 98,050
          - 28
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - jannis
          - 83,733
          - 54
          - 0
          - 4
          - multiclass_classification
          - 0.0%
        * - california_housing
          - 20,640
          - 8
          - 0
          - 1
          - regression
          - 0.0%
        * - microsoft
          - 1,200,192
          - 136
          - 0
          - 1
          - regression
          - 0.0%
        * - yahoo
          - 709,877
          - 699
          - 0
          - 1
          - regression
          - 0.0%
        * - year
          - 515,345
          - 90
          - 0
          - 1
          - regression
          - 0.0%
    """

    base_url = 'https://data.pyg.org/datasets/tables/revisiting_data/'
    classification_datasets = {
        'adult', 'aloi', 'covtype', 'helena', 'higgs_small', 'jannis'
    }
    regression_datasets = {'california_housing', 'microsoft', 'yahoo', 'year'}
    name_list = sorted(
        list(classification_datasets) + list(regression_datasets))

    def __init__(self, root: str, name: str) -> None:
        assert name in self.classification_datasets | self.regression_datasets
        self.root = root
        self.name = name
        path = self.download_url(osp.join(self.base_url, self.name + '.zip'),
                                 root)
        df, col_to_stype = get_df_and_col_to_stype(path)
        if name in self.regression_datasets:
            col_to_stype[TARGET_COL] = torch_frame.numerical
        else:
            col_to_stype[TARGET_COL] = torch_frame.categorical
        super().__init__(df, col_to_stype, target_col=TARGET_COL,
                         split_col=SPLIT_COL)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}')")
