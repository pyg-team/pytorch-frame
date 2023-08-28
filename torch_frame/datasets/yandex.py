import json
import os.path as osp
import zipfile

import numpy as np
import pandas as pd

import torch_frame


def load_dataset(path):
    dataset = {}
    with zipfile.ZipFile(path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('/'):
                continue
            if file_name.endswith('.npy'):
                with zip_ref.open(file_name) as f:
                    array_name = osp.basename(file_name).replace('.npy', '')
                    dataset[array_name] = np.load(f, allow_pickle=True)
            if file_name.endswith('.json'):
                with zip_ref.open(file_name) as f:
                    dataset['info'] = json.load(f)
    return dataset


def load_and_merge_to_df(zip_file_path):
    dataset = load_dataset(zip_file_path)
    dataframe_list = []
    col_to_stype = {}

    # Identify the available splits by examining the keys in dataset dictionary
    available_splits = [
        split for split in ['train', 'val', 'test', 'trainval']
        if f'C_{split}' in dataset or f'N_{split}' in dataset
    ]

    for split in available_splits:
        C_features = dataset.get(f'C_{split}', None)
        N_features = dataset.get(f'N_{split}', None)
        labels = dataset.get(f'y_{split}', None)

        if C_features is not None and N_features is not None:
            merged_features = np.concatenate([C_features, N_features], axis=1)
            c_col_names = [
                f'C_feature_{i}' for i in range(C_features.shape[1])
            ]
            n_col_names = [
                f'N_feature_{i}' for i in range(N_features.shape[1])
            ]
            col_names = c_col_names + n_col_names

            for name in c_col_names:
                col_to_stype[name] = torch_frame.categorical
            for name in n_col_names:
                col_to_stype[name] = torch_frame.numerical
        else:
            merged_features = (C_features
                               if C_features is not None else N_features)
            feature_type = 'C' if C_features is not None else 'N'
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

    def __init__(self, root: str, name: str):
        assert name in [
            'adult', 'aloi', 'california_housing', 'covtype', 'helena',
            'higgs_small', 'jannis', 'microsoft', 'yahoo', 'year'
        ]
        self.root = root
        self.name = name
        path = self.download_url(osp.join(self.base_url, self.name + '.zip'),
                                 root)
        df, col_to_stype = load_and_merge_to_df(path)
        if name in ['california_housing', 'microsoft', 'yahoo', 'year']:
            col_to_stype['label'] = torch_frame.numerical
        else:
            col_to_stype['label'] = torch_frame.categorical
        super().__init__(df, col_to_stype, target_col='label')
