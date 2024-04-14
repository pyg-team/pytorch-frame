import pandas as pd

import torch_frame


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
    psid = 'https://users.nber.org/~rdehejia/data/psid_controls.txt'  # noqa

    def __init__(self, root: str):
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
        nsw_control_df['source'] = 'nsw'

        names.insert(7, 'RE74')

        psid_df = pd.read_csv(psid, sep='\s+', names=names)  # noqa
        assert (psid_df['treated'] == 0).all()
        psid_df['source'] = 'psid'
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
