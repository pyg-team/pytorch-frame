from __future__ import annotations

from typing import Any

import pandas as pd

import torch_frame
from torch_frame.typing import TaskType
from torch_frame.utils import generate_random_split

SPLIT_COL = 'split'


class DataFrameBenchmark(torch_frame.data.Dataset):
    r"""A collection of standardized datasets for tabular learning, covering
    categorical and numerical features. The datasets are categorized according
    to their task types and scales.

    Args:
        root (str): Root directory.
        task_type (TaskType): The task type. Either
            :obj:`TaskType.BINARY_CLASSIFICATION`,
            :obj:`TaskType.MULTICLASS_CLASSIFICATION`, or
            :obj:`TaskType.REGRESSION`
        scale (str): The scale of the dataset. :obj:`"small"` means 5K to 50K
            rows. :obj:`"medium"` means 50K to 500K rows. :obj:`"large"`
            means more than 500K rows.
        idx (int): The index of the dataset within a category specified via
            :obj:`task_type` and :obj:`scale`.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10 20 20 10
        :header-rows: 1

        * - Task
          - Scale
          - Idx
          - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Class object
          - Missing value ratio
        * - binary_classification
          - small
          - 0
          - 32,561
          - 4
          - 8
          - 2
          - AdultCensusIncome()
          - 0.0%
        * - binary_classification
          - small
          - 1
          - 8,124
          - 0
          - 22
          - 2
          - Mushroom()
          - 0.0%
        * - binary_classification
          - small
          - 2
          - 45,211
          - 7
          - 9
          - 2
          - BankMarketing()
          - 0.0%
        * - binary_classification
          - small
          - 3
          - 13,376
          - 10
          - 0
          - 2
          - TabularBenchmark(name='MagicTelescope')
          - 0.0%
        * - binary_classification
          - small
          - 4
          - 10,578
          - 7
          - 0
          - 2
          - TabularBenchmark(name='bank-marketing')
          - 0.0%
        * - binary_classification
          - small
          - 5
          - 20,634
          - 8
          - 0
          - 2
          - TabularBenchmark(name='california')
          - 0.0%
        * - binary_classification
          - small
          - 6
          - 16,714
          - 10
          - 0
          - 2
          - TabularBenchmark(name='credit')
          - 0.0%
        * - binary_classification
          - small
          - 7
          - 13,272
          - 20
          - 1
          - 2
          - TabularBenchmark(name='default-of-credit-card-clients')
          - 0.0%
        * - binary_classification
          - small
          - 8
          - 38,474
          - 7
          - 1
          - 2
          - TabularBenchmark(name='electricity')
          - 0.0%
        * - binary_classification
          - small
          - 9
          - 7,608
          - 18
          - 5
          - 2
          - TabularBenchmark(name='eye_movements')
          - 0.0%
        * - binary_classification
          - small
          - 10
          - 10,000
          - 22
          - 0
          - 2
          - TabularBenchmark(name='heloc')
          - 0.0%
        * - binary_classification
          - small
          - 11
          - 13,488
          - 16
          - 0
          - 2
          - TabularBenchmark(name='house_16H')
          - 0.0%
        * - binary_classification
          - small
          - 12
          - 10,082
          - 26
          - 0
          - 2
          - TabularBenchmark(name='pol')
          - 0.0%
        * - binary_classification
          - small
          - 13
          - 48,842
          - 6
          - 8
          - 2
          - Yandex(name='adult')
          - 0.0%
        * - binary_classification
          - medium
          - 0
          - 92,650
          - 0
          - 116
          - 2
          - Dota2()
          - 0.0%
        * - binary_classification
          - medium
          - 1
          - 199,523
          - 7
          - 34
          - 2
          - KDDCensusIncome()
          - 0.0%
        * - binary_classification
          - medium
          - 2
          - 71,090
          - 7
          - 0
          - 2
          - TabularBenchmark(name='Diabetes130US')
          - 0.0%
        * - binary_classification
          - medium
          - 3
          - 72,998
          - 50
          - 0
          - 2
          - TabularBenchmark(name='MiniBooNE')
          - 0.0%
        * - binary_classification
          - medium
          - 4
          - 58,252
          - 23
          - 8
          - 2
          - TabularBenchmark(name='albert')
          - 0.0%
        * - binary_classification
          - medium
          - 5
          - 423,680
          - 10
          - 44
          - 2
          - TabularBenchmark(name='covertype')
          - 0.0%
        * - binary_classification
          - medium
          - 6
          - 57,580
          - 54
          - 0
          - 2
          - TabularBenchmark(name='jannis')
          - 0.0%
        * - binary_classification
          - medium
          - 7
          - 111,762
          - 24
          - 8
          - 2
          - TabularBenchmark(name='road-safety')
          - 0.0%
        * - binary_classification
          - medium
          - 8
          - 98,050
          - 28
          - 0
          - 2
          - Yandex(name='higgs_small')
          - 0.0%
        * - binary_classification
          - large
          - 0
          - 940,160
          - 24
          - 0
          - 2
          - TabularBenchmark(name='Higgs')
          - 0.0%
        * - multiclass_classification
          - medium
          - 0
          - 108,000
          - 128
          - 0
          - 1,000
          - Yandex(name='aloi')
          - 0.0%
        * - multiclass_classification
          - medium
          - 1
          - 65,196
          - 27
          - 0
          - 100
          - Yandex(name='helena')
          - 0.0%
        * - multiclass_classification
          - medium
          - 2
          - 83,733
          - 54
          - 0
          - 4
          - Yandex(name='jannis')
          - 0.0%
        * - multiclass_classification
          - large
          - 0
          - 581,012
          - 10
          - 44
          - 7
          - ForestCoverType()
          - 0.0%
        * - multiclass_classification
          - large
          - 1
          - 1,025,010
          - 5
          - 5
          - 10
          - PokerHand()
          - 0.0%
        * - multiclass_classification
          - large
          - 2
          - 581,012
          - 54
          - 0
          - 7
          - Yandex(name='covtype')
          - 0.0%
        * - regression
          - small
          - 0
          - 17,379
          - 6
          - 5
          - 1
          - TabularBenchmark(name='Bike_Sharing_Demand')
          - 0.0%
        * - regression
          - small
          - 1
          - 10,692
          - 7
          - 4
          - 1
          - TabularBenchmark(name='Brazilian_houses')
          - 0.0%
        * - regression
          - small
          - 2
          - 8,192
          - 21
          - 0
          - 1
          - TabularBenchmark(name='cpu_act')
          - 0.0%
        * - regression
          - small
          - 3
          - 16,599
          - 16
          - 0
          - 1
          - TabularBenchmark(name='elevators')
          - 0.0%
        * - regression
          - small
          - 4
          - 21,613
          - 15
          - 2
          - 1
          - TabularBenchmark(name='house_sales')
          - 0.0%
        * - regression
          - small
          - 5
          - 20,640
          - 8
          - 0
          - 1
          - TabularBenchmark(name='houses')
          - 0.0%
        * - regression
          - small
          - 6
          - 10,081
          - 6
          - 0
          - 1
          - TabularBenchmark(name='sulfur')
          - 0.0%
        * - regression
          - small
          - 7
          - 21,263
          - 79
          - 0
          - 1
          - TabularBenchmark(name='superconduct')
          - 0.0%
        * - regression
          - small
          - 8
          - 8,885
          - 252
          - 3
          - 1
          - TabularBenchmark(name='topo_2_1')
          - 0.0%
        * - regression
          - small
          - 9
          - 8,641
          - 3
          - 1
          - 1
          - TabularBenchmark(name='visualizing_soil')
          - 0.0%
        * - regression
          - small
          - 10
          - 6,497
          - 11
          - 0
          - 1
          - TabularBenchmark(name='wine_quality')
          - 0.0%
        * - regression
          - small
          - 11
          - 8,885
          - 42
          - 0
          - 1
          - TabularBenchmark(name='yprop_4_1')
          - 0.0%
        * - regression
          - small
          - 12
          - 20,640
          - 8
          - 0
          - 1
          - Yandex(name='california_housing')
          - 0.0%
        * - regression
          - medium
          - 0
          - 188,318
          - 25
          - 99
          - 1
          - TabularBenchmark(name='Allstate_Claims_Severity')
          - 0.0%
        * - regression
          - medium
          - 1
          - 241,600
          - 3
          - 6
          - 1
          - TabularBenchmark(name='SGEMM_GPU_kernel_performance')
          - 0.0%
        * - regression
          - medium
          - 2
          - 53,940
          - 6
          - 3
          - 1
          - TabularBenchmark(name='diamonds')
          - 0.0%
        * - regression
          - medium
          - 3
          - 163,065
          - 3
          - 0
          - 1
          - TabularBenchmark(name='medical_charges')
          - 0.0%
        * - regression
          - medium
          - 4
          - 394,299
          - 4
          - 2
          - 1
          - TabularBenchmark(name='particulate-matter-ukair-2017')
          - 0.0%
        * - regression
          - medium
          - 5
          - 52,031
          - 3
          - 1
          - 1
          - TabularBenchmark(name='seattlecrime6')
          - 0.0%
        * - regression
          - large
          - 0
          - 1,000,000
          - 5
          - 0
          - 1
          - TabularBenchmark(name='Airlines_DepDelay_1M')
          - 0.0%
        * - regression
          - large
          - 1
          - 5,465,575
          - 8
          - 0
          - 1
          - TabularBenchmark(name='delays_zurich_transport')
          - 0.0%
        * - regression
          - large
          - 2
          - 581,835
          - 9
          - 0
          - 1
          - TabularBenchmark(name='nyc-taxi-green-dec-2016')
          - 0.0%
        * - regression
          - large
          - 3
          - 1,200,192
          - 136
          - 0
          - 1
          - Yandex(name='microsoft')
          - 0.0%
        * - regression
          - large
          - 4
          - 709,877
          - 699
          - 0
          - 1
          - Yandex(name='yahoo')
          - 0.0%
        * - regression
          - large
          - 5
          - 515,345
          - 90
          - 0
          - 1
          - Yandex(name='year')
          - 0.0%
    """
    dataset_categorization_dict: dict[str, dict[str, list[tuple]]] = {
        'binary_classification': {
            'small': [
                ('AdultCensusIncome', {}),
                ('Mushroom', {}),
                ('BankMarketing', {}),
                ('TabularBenchmark', {
                    'name': 'MagicTelescope'
                }),
                ('TabularBenchmark', {
                    'name': 'bank-marketing'
                }),
                ('TabularBenchmark', {
                    'name': 'california'
                }),
                ('TabularBenchmark', {
                    'name': 'credit'
                }),
                ('TabularBenchmark', {
                    'name': 'default-of-credit-card-clients'
                }),
                ('TabularBenchmark', {
                    'name': 'electricity'
                }),
                ('TabularBenchmark', {
                    'name': 'eye_movements'
                }),
                ('TabularBenchmark', {
                    'name': 'heloc'
                }),
                ('TabularBenchmark', {
                    'name': 'house_16H'
                }),
                ('TabularBenchmark', {
                    'name': 'pol'
                }),
                ('Yandex', {
                    'name': 'adult'
                }),
            ],
            'medium': [
                ('Dota2', {}),
                ('KDDCensusIncome', {}),
                ('TabularBenchmark', {
                    'name': 'Diabetes130US'
                }),
                ('TabularBenchmark', {
                    'name': 'MiniBooNE'
                }),
                ('TabularBenchmark', {
                    'name': 'albert'
                }),
                ('TabularBenchmark', {
                    'name': 'covertype'
                }),
                ('TabularBenchmark', {
                    'name': 'jannis'
                }),
                ('TabularBenchmark', {
                    'name': 'road-safety'
                }),
                ('Yandex', {
                    'name': 'higgs_small'
                }),
            ],
            'large': [
                ('TabularBenchmark', {
                    'name': 'Higgs'
                }),
            ]
        },
        'multiclass_classification': {
            'small': [],
            'medium': [
                ('Yandex', {
                    'name': 'aloi'
                }),
                ('Yandex', {
                    'name': 'helena'
                }),
                ('Yandex', {
                    'name': 'jannis'
                }),
            ],
            'large': [
                ('ForestCoverType', {}),
                ('PokerHand', {}),
                ('Yandex', {
                    'name': 'covtype'
                }),
            ]
        },
        'regression': {
            'small': [
                ('TabularBenchmark', {
                    'name': 'Bike_Sharing_Demand'
                }),
                ('TabularBenchmark', {
                    'name': 'Brazilian_houses'
                }),
                ('TabularBenchmark', {
                    'name': 'cpu_act'
                }),
                ('TabularBenchmark', {
                    'name': 'elevators'
                }),
                ('TabularBenchmark', {
                    'name': 'house_sales'
                }),
                ('TabularBenchmark', {
                    'name': 'houses'
                }),
                ('TabularBenchmark', {
                    'name': 'sulfur'
                }),
                ('TabularBenchmark', {
                    'name': 'superconduct'
                }),
                ('TabularBenchmark', {
                    'name': 'topo_2_1'
                }),
                ('TabularBenchmark', {
                    'name': 'visualizing_soil'
                }),
                ('TabularBenchmark', {
                    'name': 'wine_quality'
                }),
                ('TabularBenchmark', {
                    'name': 'yprop_4_1'
                }),
                ('Yandex', {
                    'name': 'california_housing'
                }),
            ],
            'medium': [
                ('TabularBenchmark', {
                    'name': 'Allstate_Claims_Severity'
                }),
                ('TabularBenchmark', {
                    'name': 'SGEMM_GPU_kernel_performance'
                }),
                ('TabularBenchmark', {
                    'name': 'diamonds'
                }),
                ('TabularBenchmark', {
                    'name': 'medical_charges'
                }),
                ('TabularBenchmark', {
                    'name': 'particulate-matter-ukair-2017'
                }),
                ('TabularBenchmark', {
                    'name': 'seattlecrime6'
                }),
            ],
            'large': [
                ('TabularBenchmark', {
                    'name': 'Airlines_DepDelay_1M'
                }),
                ('TabularBenchmark', {
                    'name': 'delays_zurich_transport'
                }),
                ('TabularBenchmark', {
                    'name': 'nyc-taxi-green-dec-2016'
                }),
                ('Yandex', {
                    'name': 'microsoft'
                }),
                ('Yandex', {
                    'name': 'yahoo'
                }),
                ('Yandex', {
                    'name': 'year'
                }),
            ]
        }
    }

    @classmethod
    def datasets_available(
        cls,
        task_type: TaskType,
        scale: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        r"""List of datasets available for a given :obj:`task_type` and
        :obj:`scale`.
        """
        return cls.dataset_categorization_dict[task_type.value][scale]

    @classmethod
    def num_datasets_available(cls, task_type: TaskType, scale: str):
        r"""Number of datasets available for a given :obj:`task_type` and
        :obj:`scale`.
        """
        return len(cls.datasets_available(task_type, scale))

    def __init__(
        self,
        root: str,
        task_type: TaskType,
        scale: str,
        idx: int,
        split_random_state: int = 42,
    ):
        self.root = root
        self._task_type = task_type
        self.scale = scale
        self.idx = idx

        datasets = self.datasets_available(task_type, scale)
        if idx >= len(datasets):
            raise ValueError(
                f"The idx needs to be smaller than {len(datasets)}, which is "
                f"the number of available datasets for task_type: "
                f"{task_type.value} and scale: {scale} (got idx: {idx}).")

        class_name, kwargs = self.datasets_available(task_type, scale)[idx]
        dataset = getattr(torch_frame.datasets, class_name)(root=root,
                                                            **kwargs)
        self.cls_str = str(dataset)

        # Add split col
        df = dataset.df
        if SPLIT_COL in df.columns:
            df.drop(columns=[SPLIT_COL], inplace=True)
        split_df = pd.DataFrame({
            SPLIT_COL:
            generate_random_split(length=len(df), seed=split_random_state,
                                  train_ratio=0.8, val_ratio=0.1)
        })
        df = pd.concat([df, split_df], axis=1)

        # For regression task, we normalize the target.
        if task_type == TaskType.REGRESSION:
            ser = df[dataset.target_col]
            df[dataset.target_col] = (ser - ser.mean()) / ser.std()

        # Check the scale
        if dataset.num_rows < 5000:
            assert False
        elif dataset.num_rows < 50000:
            assert scale == "small"
        elif dataset.num_rows < 500000:
            assert scale == "medium"
        else:
            assert scale == "large"

        super().__init__(df=df, col_to_stype=dataset.col_to_stype,
                         target_col=dataset.target_col, split_col=SPLIT_COL)
        del dataset

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  task_type={self._task_type.value},\n'
                f'  scale={self.scale},\n'
                f'  idx={self.idx},\n'
                f'  cls={self.cls_str}\n'
                f')')

    def materialize(self, *args, **kwargs) -> torch_frame.data.Dataset:
        super().materialize(*args, **kwargs)
        if self.task_type != self._task_type:
            raise RuntimeError(f"task type does not match. It should be "
                               f"{self.task_type.value} but specified as "
                               f"{self._task_type.value}.")
        return self
