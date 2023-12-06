import os.path as osp
import zipfile

import pandas as pd

import torch_frame


class BankMarketing(torch_frame.data.Dataset):
    r"""The `Bank Marketing
    <https://github.com/LeoGrin/tabular-benchmark>`_
    dataset. It's related with direct marketing campaigns of
    a Portuguese banking institution. The marketing campaigns
    were based on phone calls. Often, more than one contant to
    the same client was required, in order to access if the
    product (bank term deposit) would be (or not) subscribed.
    The classification goal is to predict if the client will
    subscribe a term deposit.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Task
          - Missing value ratio
        * - 45,211
          - 7
          - 9
          - 2
          - binary_classification
          - 0.0%
    """

    url = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'  # noqa

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        folder_path = osp.dirname(path)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        data_path = osp.join(folder_path, 'bank.zip')
        data_subfolder_path = osp.join(folder_path, 'bank')
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall(data_subfolder_path)
        df = pd.read_csv(osp.join(data_subfolder_path, 'bank-full.csv'),
                         sep=';')

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
