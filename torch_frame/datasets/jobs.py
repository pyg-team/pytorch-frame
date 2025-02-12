import numpy as np
import pandas as pd

import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM


class Jobs(torch_frame.data.Dataset):
    r"""The Jobs dataset from "Evaluating the Econometric
    Evaluations of Training Programs with Experimental Data"
    by Robert Lalonde. There are two versions of the data. One
    version is the `Dehejia subsampe.
    <https://users.nber.org/~rdehejia/data>`_. The version
    is a subsample of Lalonde's original dataset because it includes
    one more feature--RE74 (earnings in 1974, two years prior the
    treatment). The use of more than one year of pretreatment
    earnings is key in accurately estimating the treatment effect,
    because many people who volunteer for training programs experience
    a drop in their earnings just prior to entering the training program.

    Another version is a version containing additional columns obtained
    from feature engineering, from
    `Dr.Johansson's website <https://www.fredjo.com>_`.

    The target in the dataset is index to the target tensor. The target
    tensor is a :obj:`Tensor` of size (num_rows, 2), where the first
    column represents the target and the second column represents the
    treatment.
    """
    dehejia_treated_url = 'https://users.nber.org/~rdehejia/data/nswre74_treated.txt'  # noqa
    dehejia_control_url = 'https://users.nber.org/~rdehejia/data/nswre74_control.txt'  # noqa
    psid_url = 'https://users.nber.org/~rdehejia/data/psid_controls.txt'
    train_url = 'https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz'
    test_url = 'https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz'

    def __init__(self, root: str, feature_engineering: bool = False):
        if feature_engineering:
            split = 0
            train = self.download_url(self.train_url, root)
            test = self.download_url(self.test_url, root)
            train_np = np.load(train)
            test_np = np.load(test)
            train_data = np.concatenate([
                train_np.f.t[:, split].reshape(-1, 1),
                train_np.f.x[:, :, split], train_np.f.e[:, split].reshape(
                    -1, 1), train_np.f.yf[:, split].reshape(-1, 1)
            ], axis=1)
            test_data = np.concatenate([
                test_np.f.t[:, split].reshape(-1, 1), test_np.f.x[:, :, split],
                test_np.f.e[:, split].reshape(
                    -1, 1), test_np.f.yf[:, split].reshape(-1, 1)
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
            nsw_treated = self.download_url(self.dehejia_treated_url, root)
            nsw_control = self.download_url(self.dehejia_control_url, root)
            # Population Survey of Income Dynamics
            psid = self.download_url(self.psid_url, root)
            names = [
                'treated', 'age', 'education', 'Black', 'Hispanic', 'married',
                'nodegree', 'RE74', 'RE75', 'RE78'
            ]

            nsw_treated_df = pd.read_csv(
                nsw_treated,
                sep='\s+',  # noqa
                names=names)
            assert (nsw_treated_df['treated'] == 1).all()
            nsw_treated_df['source'] = 1

            nsw_control_df = pd.read_csv(
                nsw_control,
                sep='\s+',  # noqa
                names=names)
            assert (nsw_control_df['treated'] == 0).all()
            nsw_control_df['source'] = 1

            psid_df = pd.read_csv(psid, sep='\s+', names=names)  # noqa
            assert (psid_df['treated'] == 0).all()
            psid_df['source'] = 0

            df = pd.concat([nsw_treated_df, nsw_control_df, psid_df], axis=0)
            df['target'] = df['RE78'] != 0

            col_to_stype = {
                'treated': torch_frame.categorical,
                'age': torch_frame.numerical,
                'education': torch_frame.numerical,
                'Black': torch_frame.categorical,
                'Hispanic': torch_frame.categorical,
                'married': torch_frame.categorical,
                'nodegree': torch_frame.categorical,
                'RE74': torch_frame.numerical,
                'RE75': torch_frame.numerical,
                'target': torch_frame.categorical,
            }
            super().__init__(df, col_to_stype, target_col='target')
        self.df = df

    def get_att(self):
        r"""Obtain the ATT(true Average Treatment effect on Treated)).

        Returns:
            float: The ATT score from the original randomized experiments.
        """
        df = self.df[self.df['source'] == 1]
        treated = df[df['treated'] == 1]
        control = df[df['treated'] == 0]
        return sum(treated['target']) / len(treated) - sum(
            control['target']) / len(control)
