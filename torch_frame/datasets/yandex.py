import os.path as osp
import zipfile
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import torch_frame


def load_dataset(path: str) -> Dict[str, Union[np.ndarray, Any]]:
    """
    Load a dataset from a ZIP file.

    Parameters:
    -----------
    path : str
        The file path to the ZIP file containing the dataset.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Any]]
        A dictionary where each key-value pair corresponds to either an array
        read from a .npy file within the ZIP file. Key is name of the file,
        e.g., 'train', 'test', etc.
    """
    dataset = {}
    with zipfile.ZipFile(path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.npy'):
                with zip_ref.open(file_name) as f:
                    array_name = osp.basename(file_name).replace('.npy', '')
                    dataset[array_name] = np.load(f, allow_pickle=True)
    return dataset


def load_and_merge_to_df(
        zip_file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and merge datasets from a ZIP file into a DataFrame.

    Parameters:
    - zip_file_path (str): The file path of the ZIP file containing
      the dataset.

    Returns:
    - Tuple[pd.DataFrame, Dict[str, Any]]: A tuple of two elements:
        1. A merged DataFrame containing the dataset.
        2. A dictionary mapping column names to their
            respective data types.

    """
    dataset = load_dataset(zip_file_path)
    dataframe_list: List[pd.DataFrame] = []
    col_to_stype: Dict[str, Any] = {}

    # Identify the available splits by examining the keys in dataset dictionary
    available_splits = [
        split for split in ['train', 'val', 'test', 'trainval']
        if f'C_{split}' in dataset or f'N_{split}' in dataset
    ]

    for split in available_splits:
        categorical_features = dataset.get(f'C_{split}', None)
        numerical_features = dataset.get(f'N_{split}', None)
        labels = dataset.get(f'y_{split}', None)

        if categorical_features is not None and numerical_features is not None:
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

        if merged_features is not None and labels is not None:
            df = pd.DataFrame(merged_features, columns=col_names)
            df['label'] = labels
            df['split'] = split
            dataframe_list.append(df)

    merged_df = pd.concat(dataframe_list, ignore_index=True)

    return merged_df, col_to_stype


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
        df, col_to_stype = load_and_merge_to_df(path)
        if name in self.regression_datasets:
            col_to_stype['label'] = torch_frame.numerical
        else:
            col_to_stype['label'] = torch_frame.categorical
        super().__init__(df, col_to_stype, target_col='label')
