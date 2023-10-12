from typing import Any, Dict, List, Tuple

import torch_frame
from torch_frame.typing import TaskType


class DataFrameBenchmark(torch_frame.data.Dataset):
    r"""A collection of standardized datasets for tabular learning covering
    categorical and numerical features.

    Args:
        root (str): Root directory.
        task_type (TaskType): The task type.
            - :obj:`TaskType.BINARY_CLASSIFICATION`
            - :obj:`TaskType.MULTICLASS_CLASSIFICATION`
            - :obj:`TaskType.REGRESSION`
        scale (str): The scale of the dataset.
            - :obj:`small`: 5K to 50K rows.
            - :obj:`medium`: 50K to 500K rows.
            - :obj:`large`: More than 500K rows.
        idx (int): The integer
    """
    dataset_categorization_dict = {
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
    def datasets_available(cls, task_type: TaskType,
                           scale: str) -> List[Tuple[str, Dict[str, Any]]]:
        return cls.dataset_categorization_dict[task_type.value][scale]

    @classmethod
    def num_datasets_available(cls, task_type: TaskType, scale: str):
        return len(cls.datasets_available(task_type, scale))

    def __init__(self, root: str, task_type: TaskType, scale: str, idx: int):
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

        # check the scale
        if dataset.num_rows < 5000:
            assert False
        elif dataset.num_rows < 50000:
            assert scale == "small"
        elif dataset.num_rows < 500000:
            assert scale == "medium"
        else:
            assert scale == "large"

        super().__init__(df=dataset.df, col_to_stype=dataset.col_to_stype,
                         target_col=dataset.target_col)
        del dataset

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  task_type={self._task_type.value},\n'
                f'  scale={self.scale},\n'
                f'  idx={self.idx},\n'
                f'  cls={self.cls_str}\n'
                f')')

    def materialize(self):
        super().materialize()
        if self.task_type != self._task_type:
            raise RuntimeError(f"task type does not match. It should be "
                               f"{self.task_type.value} but specified as "
                               f"{self._task_type.value}.")
