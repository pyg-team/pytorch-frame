import numpy as np
import pandas as pd

import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM


class Jobs(torch_frame.data.Dataset):
    r"""The `Jobs
    <https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz>`_
    dataset from Lalonde.
    treatment indicator (1 if treated, 0 if not treated), age,
    education, Black (1 if black, 0 otherwise), Hispanic
    (1 if Hispanic, 0 otherwise), married (1 if married, 0 otherwise),
    nodegree (1 if no degree, 0 otherwise), RE74 (earnings in 1974),
    RE75 (earnings in 1975), and RE78 (earnings in 1978).
    """
    lalonde_treated = 'https://users.nber.org/~rdehejia/data/nsw_treated.txt'
    lalonde_control = 'https://users.nber.org/~rdehejia/data/nsw_control.txt'
    psid = 'https://users.nber.org/~rdehejia/data/psid_controls.txt'
    train = 'https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz'
    test = 'https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz'

    def __init__(self, root: str, feature_engineering: bool = False):
        if feature_engineering:
            train = self.download_url(Jobs.train, root)
            test = self.download_url(Jobs.test, root)
            train_np = np.load(train)
            test_np = np.load(test)
            train_data = np.concatenate([
                train_np.f.t[:, 0].reshape(-1, 1), train_np.f.x[:, :, 0],
                train_np.f.e[:, 0].reshape(-1, 1), train_np.f.yf[:, 0].reshape(
                    -1, 1)
            ], axis=1)
            test_data = np.concatenate([
                test_np.f.t[:, 0].reshape(-1, 1), test_np.f.x[:, :, 0],
                test_np.f.e[:, 0].reshape(-1, 1), test_np.f.yf[:, 0].reshape(
                    -1, 1)
            ], axis=1)
            train_df = pd.DataFrame(
                train_data, columns=['treated'] +
                [f'Col_{i}'
                 for i in range(train_np.f.x.shape[1])] + ['source', 'target'])
            train_df['split'] = SPLIT_TO_NUM['train']
            test_df = pd.DataFrame(
                test_data, columns=['treated'] +
                [f'Col_{i}'
                 for i in range(train_np.f.x.shape[1])] + ['source', 'target'])
            test_df['split'] = SPLIT_TO_NUM['test']
            df = pd.concat([train_df, test_df], axis=0)
            col_to_stype = {
                'treated': torch_frame.categorical,
                'Col_0': torch_frame.numerical,
                'Col_1': torch_frame.numerical,
                'Col_2': torch_frame.categorical,
                'Col_3': torch_frame.categorical,
                'Col_4': torch_frame.categorical,
                'Col_5': torch_frame.categorical,
                'Col_6': torch_frame.numerical,
                'Col_7': torch_frame.numerical,
                'Col_8': torch_frame.numerical,
                'Col_9': torch_frame.numerical,
                'Col_10': torch_frame.numerical,
                'Col_11': torch_frame.numerical,
                'Col_12': torch_frame.numerical,
                'Col_13': torch_frame.categorical,
                'Col_14': torch_frame.categorical,
                'Col_15': torch_frame.numerical,
                'Col_16': torch_frame.categorical,
                'target': torch_frame.categorical
            }
            super().__init__(df, col_to_stype, target_col='target',
                             split_col='split')
        else:
            # National Supported Work Demonstration
            nsw_treated = self.download_url(Jobs.lalonde_treated, root)
            nsw_control = self.download_url(Jobs.lalonde_control, root)
            # Population Survey of Income Dynamics
            psid = self.download_url(Jobs.psid, root)
            names = [
                'treated', 'age', 'education', 'Black', 'Hispanic', 'married',
                'nodegree', 'RE75', 'RE78'
            ]

            nsw_treated_df = pd.read_csv(
                nsw_treated,
                sep='\s+',  # noqa
                names=names)
            assert (nsw_treated_df['treated'] == 1).all()
            nsw_treated_df['source'] = 'nsw'

            nsw_control_df = pd.read_csv(
                nsw_control,
                sep='\s+',  # noqa
                names=names)
            assert (nsw_control_df['treated'] == 0).all()
            nsw_control_df['source'] = 1

            names.insert(7, 'RE74')

            psid_df = pd.read_csv(psid, sep='\s+', names=names)  # noqa
            assert (psid_df['treated'] == 0).all()
            psid_df['source'] = 0
            psid_df = psid_df.drop('RE74', axis=1)

            df = pd.concat([nsw_treated_df, nsw_control_df, psid_df], axis=0)
            df['target'] = df['RE78'] != 0

            col_to_stype = {
                'treated': torch_frame.categorical,
                'age': torch_frame.numerical,
                'education': torch_frame.categorical,
                'Black': torch_frame.categorical,
                'Hispanic': torch_frame.categorical,
                'married': torch_frame.categorical,
                'nodegree': torch_frame.categorical,
                'RE75': torch_frame.numerical,
                'target': torch_frame.categorical,
            }

            super().__init__(df, col_to_stype, target_col='target')
        self.df = df

    def get_att(self):
        df = self.df[self.df['source'] == 1]
        treated = df[df['treated'] == 1]
        control = df[df['treated'] == 0]
        return sum(treated['target']) / len(treated) - sum(
            control['target']) / len(control)
