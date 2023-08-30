import os.path as osp
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch_frame


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    r"""Load a dataset from a ZIP file.

    Args:
        path (str): The file path to the ZIP file containing some .npy files
            that store the dataset.

    Returns:
        dataset (Dict[str, np.ndarray]): A dictionary that maps the name of
            .npy file to the loaded numpy array.
    """
    dataset: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.npy'):
                with zip_ref.open(file_name) as f:
                    array_name = osp.basename(file_name).replace('.npy', '')
                    dataset[array_name] = np.load(f, allow_pickle=True)
    return dataset


def get_df_and_col_to_stype(
        zip_file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""Get DataFrame and :obj:`col_to_stype` from a ZIP file.

    Args:
        zip_file_path (str): The file path of the ZIP file containing
            the dataset.

    Returns:
        df (DataFrame): DataFrame containing train/val/test rows.
        col_to_stype (Dict[str, torch_frame.stype]). A dictionary mapping
            column names to their respective semantic types.
    """
    dataset = load_dataset(zip_file_path)
    dataframes: List[pd.DataFrame] = []
    col_to_stype: Dict[str, torch_frame.stype] = {}

    for split in ['train', 'val', 'test']:
        categorical_features = dataset.get(f'C_{split}', None)
        numerical_features = dataset.get(f'N_{split}', None)
        labels = dataset[f'y_{split}']
        assert (categorical_features is None) and (numerical_features is None)

        merged_features: Optional[np.ndarray] = None
        if (categorical_features is not None) and (numerical_features
                                                   is not None):
            merged_features = np.concatenate(
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
            # if the dataset contains only categorical or numerical features
            merged_features = (categorical_features if categorical_features
                               is not None else numerical_features)
            feature_type = 'C' if categorical_features is not None else 'N'
            col_names = [
                f'{feature_type}_feature_{i}'
                for i in range(merged_features.shape[1])
            ]

            for name in col_names:
                col_to_stype[name] = (torch_frame.categorical if feature_type
                                      == 'C' else torch_frame.numerical)
        assert merged_features is not None
        df = pd.DataFrame(merged_features, columns=col_names)
        df['label'] = labels
        # Stores the split information in "split" column.
        df['split'] = split
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    return df, col_to_stype


class Yandex(torch_frame.data.Dataset):

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
        super().__init__(df, col_to_stype, target_col='label')
