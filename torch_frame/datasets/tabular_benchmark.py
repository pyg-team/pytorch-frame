from __future__ import annotations

import os

import pandas as pd
from pandas.api.types import is_numeric_dtype

import torch_frame


class TabularBenchmark(torch_frame.data.Dataset):
    r"""A collection of Tabular benchmark datasets introduced in
    `"Why do tree-based models still outperform deep learning on tabular data?"
    <https://arxiv.org/abs/2207.08815>`_.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 20 10
        :header-rows: 1

        * - Name
          - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Task
          - Missing value ratio
        * - albert
          - 58,252
          - 23
          - 8
          - 2
          - binary_classification
          - 0.0%
        * - compas-two-years
          - 4,966
          - 2
          - 9
          - 2
          - binary_classification
          - 0.0%
        * - covertype
          - 423,680
          - 10
          - 44
          - 2
          - binary_classification
          - 0.0%
        * - default-of-credit-card-clients
          - 13,272
          - 20
          - 1
          - 2
          - binary_classification
          - 0.0%
        * - electricity
          - 38,474
          - 7
          - 1
          - 2
          - binary_classification
          - 0.0%
        * - eye_movements
          - 7,608
          - 18
          - 5
          - 2
          - binary_classification
          - 0.0%
        * - road-safety
          - 111,762
          - 24
          - 8
          - 2
          - binary_classification
          - 0.0%
        * - Bioresponse
          - 3,434
          - 419
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - Diabetes130US
          - 71,090
          - 7
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - Higgs
          - 940,160
          - 24
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - MagicTelescope
          - 13,376
          - 10
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - MiniBooNE
          - 72,998
          - 50
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - bank-marketing
          - 10,578
          - 7
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - california
          - 20,634
          - 8
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - credit
          - 16,714
          - 10
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - heloc
          - 10,000
          - 22
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - house_16H
          - 13,488
          - 16
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - jannis
          - 57,580
          - 54
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - pol
          - 10,082
          - 26
          - 0
          - 2
          - binary_classification
          - 0.0%
        * - analcatdata_supreme
          - 4,052
          - 1
          - 6
          - 1
          - regression
          - 0.0%
        * - Airlines_DepDelay_1M
          - 1,000,000
          - 5
          - 0
          - 1
          - regression
          - 0.0%
        * - Allstate_Claims_Severity
          - 188,318
          - 25
          - 99
          - 1
          - regression
          - 0.0%
        * - Bike_Sharing_Demand
          - 17,379
          - 6
          - 5
          - 1
          - regression
          - 0.0%
        * - Brazilian_houses
          - 10,692
          - 7
          - 4
          - 1
          - regression
          - 0.0%
        * - Mercedes_Benz_Greener_Manufacturing
          - 4,209
          - 1
          - 358
          - 1
          - regression
          - 0.0%
        * - SGEMM_GPU_kernel_performance
          - 241,600
          - 3
          - 6
          - 1
          - regression
          - 0.0%
        * - diamonds
          - 53,940
          - 6
          - 3
          - 1
          - regression
          - 0.0%
        * - house_sales
          - 21,613
          - 15
          - 2
          - 1
          - regression
          - 0.0%
        * - medical_charges
          - 163,065
          - 3
          - 0
          - 1
          - regression
          - 0.0%
        * - particulate-matter-ukair-2017
          - 394,299
          - 4
          - 2
          - 1
          - regression
          - 0.0%
        * - seattlecrime6
          - 52,031
          - 3
          - 1
          - 1
          - regression
          - 0.0%
        * - topo_2_1
          - 8,885
          - 252
          - 3
          - 1
          - regression
          - 0.0%
        * - visualizing_soil
          - 8,641
          - 3
          - 1
          - 1
          - regression
          - 0.0%
        * - cpu_act
          - 8,192
          - 21
          - 0
          - 1
          - regression
          - 0.0%
        * - elevators
          - 16,599
          - 16
          - 0
          - 1
          - regression
          - 0.0%
        * - houses
          - 20,640
          - 8
          - 0
          - 1
          - regression
          - 0.0%
        * - delays_zurich_transport
          - 5,465,575
          - 8
          - 0
          - 1
          - regression
          - 0.0%
        * - nyc-taxi-green-dec-2016
          - 581,835
          - 9
          - 0
          - 1
          - regression
          - 0.0%
        * - sulfur
          - 10,081
          - 6
          - 0
          - 1
          - regression
          - 0.0%
        * - superconduct
          - 21,263
          - 79
          - 0
          - 1
          - regression
          - 0.0%
        * - wine_quality
          - 6,497
          - 11
          - 0
          - 1
          - regression
          - 0.0%
        * - yprop_4_1
          - 8,885
          - 42
          - 0
          - 1
          - regression
          - 0.0%
    """

    name_to_task_category = {
        'albert': 'clf_cat',
        'compas-two-years': 'clf_cat',
        'covertype': 'clf_cat',
        'default-of-credit-card-clients': 'clf_cat',
        'electricity': 'clf_cat',
        'eye_movements': 'clf_cat',
        'road-safety': 'clf_cat',
        'Bioresponse': 'clf_num',
        'Diabetes130US': 'clf_num',
        'Higgs': 'clf_num',
        'MagicTelescope': 'clf_num',
        'MiniBooNE': 'clf_num',
        'bank-marketing': 'clf_num',
        'california': 'clf_num',
        'credit': 'clf_num',
        'heloc': 'clf_num',
        'house_16H': 'clf_num',
        'jannis': 'clf_num',
        'pol': 'clf_num',
        'analcatdata_supreme': 'reg_cat',
        'Airlines_DepDelay_1M': 'reg_cat',
        'Allstate_Claims_Severity': 'reg_cat',
        'Bike_Sharing_Demand': 'reg_cat',
        'Brazilian_houses': 'reg_cat',
        'Mercedes_Benz_Greener_Manufacturing': 'reg_cat',
        'SGEMM_GPU_kernel_performance': 'reg_cat',
        'diamonds': 'reg_cat',
        'house_sales': 'reg_cat',
        'medical_charges': 'reg_cat',
        'particulate-matter-ukair-2017': 'reg_cat',
        'seattlecrime6': 'reg_cat',
        'topo_2_1': 'reg_cat',
        'visualizing_soil': 'reg_cat',
        'elevators': 'reg_num',
        'houses': 'reg_num',
        'cpu_act': 'reg_num',
        'delays_zurich_transport': 'reg_num',
        'nyc-taxi-green-dec-2016': 'reg_num',
        'sulfur': 'reg_num',
        'superconduct': 'reg_num',
        'wine_quality': 'reg_num',
        'yprop_4_1': 'reg_num',
    }

    large_datasets = {
        'covertype',
        'road-safety',
        'Higgs',
        'MiniBooNE',
        'jannis',
        'delays_zurich_transport',
        'particulate-matter-ukair-2017',
        'nyc-taxi-green-dec-2016',
        'SGEMM_GPU_kernel_performance',
        'Airlines_DepDelay_1M',
        'Allstate_Claims_Severity',
        'topo_2_1',
        'superconduct',
    }

    base_url = 'https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/main/'  # noqa
    # Dedicated URLs for large datasets
    base_url_large = 'https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/'  # noqa
    name_list = sorted(list(name_to_task_category.keys()))

    def __init__(self, root: str, name: str) -> None:
        self.root = root
        self.name = name
        if name not in self.name_to_task_category:
            raise ValueError(
                f"The given dataset name ('{name}') is not available. It "
                f"needs to be chosen from "
                f"{list(self.name_to_task_category.keys())}.")
        base_url = (self.base_url_large
                    if name in self.large_datasets else self.base_url)
        task_category = self.name_to_task_category[name]
        url = os.path.join(
            base_url,
            task_category,
            f'{name}.csv',
        )
        path = self.download_url(url, root)
        df = pd.read_csv(path)
        # The last column is the target column
        col_to_stype = {}
        target_col = df.columns[-1]
        if "clf" in task_category:
            col_to_stype[target_col] = torch_frame.categorical
        else:
            col_to_stype[target_col] = torch_frame.numerical

        for col in df.columns[:-1]:
            if "num" in task_category:
                # "num" implies all features are numerical.
                col_to_stype[col] = torch_frame.numerical
            elif df[col].dtype == float:
                col_to_stype[col] = torch_frame.numerical
            else:
                # Heuristics to decide stype
                if is_numeric_dtype(df[col].dtype) and df[col].nunique() > 10:
                    col_to_stype[col] = torch_frame.numerical
                else:
                    col_to_stype[col] = torch_frame.categorical
        super().__init__(df, col_to_stype, target_col=target_col)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
