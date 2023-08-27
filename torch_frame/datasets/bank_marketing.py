import pandas as pd

import torch_frame


class AdultCensusIncome(torch_frame.data.Dataset):
    r"""The `Bank Marketing
    <https://github.com/LeoGrin/tabular-benchmark>`_
    dataset. It's related with direct marketing campaigns of
    a Portuguese banking institution. The marketing campaigns
    were based on phone calls. Often, more than one contant to
    the same client was required, in order to access if the
    product(bank term deposit) would be(or not) subscribed.
    The classification goal is to predict if the client will
    subscribe a term deposit. """

    url = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'  # noqa

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        names = [
            'age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome', 'y'
        ]
        df = pd.read_csv(path, names=names)

        col_to_stype = {
            'age': torch_frame.numerical,
            'job': torch_frame.categorical,
            'marital': torch_frame.categorical,
            'education': torch_frame.categorical,
            'default': torch_frame.categorical,
            'balance': torch_frame.numerical,
            'housing': torch_frame.categorical,
            'loan': torch_frame.categorical,
            'contact': torch_frame.categorical,
            'day': torch_frame.numerical,
            'month': torch_frame.categorical,
            'duration': torch_frame.numerical,
            'campaign': torch_frame.numerical,
            'pdays': torch_frame.numerical,
            'previous': torch_frame.numerical,
            'poutcome': torch_frame.categorical,
            'y': torch_frame.categorical,
        }

        super().__init__(df, col_to_stype, target_col='y')
