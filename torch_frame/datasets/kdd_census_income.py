from __future__ import annotations

import os
import os.path as osp
import tarfile
import zipfile

import pandas as pd

import torch_frame
from torch_frame import stype


class KDDCensusIncome(torch_frame.data.Dataset):
    r"""The `KDD Census Income
    <https://archive.ics.uci.edu/dataset/117/census+income+kdd>`_
    dataset. It's a task of forest cover type classification
    based on attributes such as elevation, slop and soil type etc.

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
        * - 199,523
          - 7
          - 34
          - 2
          - binary_classification
          - 0.0%
    """

    url = 'https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip'

    def __init__(self, root: str):
        data_dir = osp.join(root, 'census')
        filename = osp.join(data_dir, 'census-income.data')
        if not osp.exists(filename):
            path = self.download_url(self.url, root)
            tar_gz_path = osp.join(root, 'census.tar.gz')
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(root)
            with tarfile.open(tar_gz_path, 'r:gz') as tar_ref:
                tar_ref.extractall(data_dir)
            os.remove(tar_gz_path)
            os.remove(path)

        names = [
            'age',
            'class of worker',
            'industry code',
            'occupation code',
            'education',
            'wage per hour',
            'enrolled in edu inst last wk',
            'marital status',
            'major industry code',
            'major occupation code',
            'race',
            'hispanic Origin',
            'sex',
            'member of a labor union',
            'reason for unemployment',
            'full or part time employment stat',
            'capital gains',
            'capital losses',
            'divdends from stocks',
            'tax filer status',
            'region of previous residence',
            'state of previous residence',
            'detailed household and family stat',
            'detailed household summary in household',
            'migration code-change in msa',
            'migration code-change in reg',
            'migration code-move within reg',
            'live in this house 1 year ago',
            'migration prev res in sunbelt',
            'family members under 18',
            'num persons worked for employer',
            'country of birth father',
            'country of birth mother',
            'country of birth self',
            'citizenship',
            'total person income',
            'own business or self employed',
            "fill inc questionnaire for veteran's admin",
            'veterans benefits',
            'weeks worked in year',
            'year',
            'income above 50000',
        ]

        continuous_cols = {
            'age',
            'wage per hour',
            'capital gains',
            'capital losses',
            'divdends from stocks',
            'num persons worked for employer',
            'weeks worked in year',
        }

        col_to_stype: dict[str, stype] = {}
        for name in names:
            if name in continuous_cols:
                col_to_stype[name] = torch_frame.numerical
            else:
                col_to_stype[name] = torch_frame.categorical

        df = pd.read_csv(filename, names=names)

        super().__init__(df, col_to_stype, target_col='income above 50000')
