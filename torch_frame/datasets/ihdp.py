import os.path as osp
import zipfile

import numpy as np
import pandas as pd

import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM


class IHDP(torch_frame.data.Dataset):
    r"""Counterfactual target is generated with knn."""
    train_url = 'https://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip'
    test_url = 'https://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip'

    def __init__(self, root: str, split_num: int = 0):
        train_path = self.download_url(self.train_url, root)
        test_path = self.download_url(self.test_url, root)
        self.split_num = split_num
        folder_path = osp.dirname(train_path)
        with zipfile.ZipFile(train_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        with zipfile.ZipFile(test_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        train_np = np.load(osp.join(folder_path, 'ihdp_npci_1-1000.train.npz'))
        test_np = np.load(osp.join(folder_path, 'ihdp_npci_1-1000.test.npz'))
        self.train_np = train_np
        self.test_np = test_np
        train_data = np.concatenate([
            train_np.f.t[:, split_num].reshape(-1, 1),
            train_np.f.x[:, :, split_num], train_np.f.yf[:, split_num].reshape(
                -1, 1), train_np.f.ycf[:, split_num].reshape(-1, 1)
        ], axis=1)
        test_data = np.concatenate([
            test_np.f.t[:, split_num].reshape(-1, 1),
            test_np.f.x[:, :, split_num], test_np.f.yf[:, split_num].reshape(
                -1, 1), test_np.f.ycf[:, split_num].reshape(-1, 1)
        ], axis=1)
        train_df = pd.DataFrame(
            train_data, columns=['treated'] +
            [f'Col_{i}' for i in range(train_np.f.x.shape[1])] + ['target'] +
            ['counterfactual_target'])
        train_df['split'] = SPLIT_TO_NUM['train']
        test_df = pd.DataFrame(
            test_data, columns=['treated'] +
            [f'Col_{i}' for i in range(train_np.f.x.shape[1])] + ['target'] +
            ['counterfactual_target'])
        test_df['split'] = SPLIT_TO_NUM['test']
        df = pd.concat([train_df, test_df], axis=0)
        col_to_stype = {
            'treated': torch_frame.categorical,
            'Col_0': torch_frame.numerical,
            'Col_1': torch_frame.numerical,
            'Col_2': torch_frame.numerical,
            'Col_3': torch_frame.numerical,
            'Col_4': torch_frame.numerical,
            'Col_5': torch_frame.numerical,
            'Col_6': torch_frame.categorical,
            'Col_7': torch_frame.categorical,
            'Col_8': torch_frame.categorical,
            'Col_9': torch_frame.categorical,
            'Col_10': torch_frame.categorical,
            'Col_11': torch_frame.categorical,
            'Col_12': torch_frame.categorical,
            'Col_13': torch_frame.categorical,
            'Col_14': torch_frame.categorical,
            'Col_15': torch_frame.categorical,
            'Col_16': torch_frame.categorical,
            'Col_17': torch_frame.categorical,
            'Col_18': torch_frame.categorical,
            'Col_19': torch_frame.categorical,
            'Col_20': torch_frame.categorical,
            'Col_21': torch_frame.categorical,
            'Col_22': torch_frame.categorical,
            'Col_23': torch_frame.categorical,
            'Col_24': torch_frame.categorical,
            'target': torch_frame.numerical,
        }
        super().__init__(df, col_to_stype=col_to_stype, target_col='target',
                         split_col='split')

    def get_att(self):
        r"""Obtain the ATT(true Average Treatment effect on Treated)).

        Returns:
            float: The ATT score from the original randomized experiments.
        """
        mu1 = np.concatenate([
            self.train_np.f.mu1[:, self.split_num],
            self.test_np.f.mu1[:, self.split_num]
        ], axis=0)
        mu0 = np.concatenate([
            self.train_np.f.mu0[:, self.split_num],
            self.test_np.f.mu0[:, self.split_num]
        ], axis=0)
        return np.mean(mu1) - np.mean(mu0)
