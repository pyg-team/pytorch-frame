# flake8: noqa

from .titanic import Titanic
from .adult_census_income import AdultCensusIncome
from .fake import FakeDataset
from .forest_cover_type import ForestCoverType
from .mushroom import Mushroom
from .poker_hand import PokerHand
from .bank_marketing import BankMarketing
from .tabular_benchmark import TabularBenchmark
from .yandex import Yandex
from .dota2 import Dota2
from .kdd_census_income import KDDCensusIncome
from .multimodal_text_benchmark import MultimodalTextBenchmark
from .data_frame_benchmark import DataFrameBenchmark
from .mercari import Mercari

real_world_datasets = [
    'Titanic',
    'AdultCensusIncome',
    'ForestCoverType',
    'Dota2',
    'Mushroom',
    'PokerHand',
    'BankMarketing',
    'TabularBenchmark',
    'Yandex',
    'KDDCensusIncome',
    'MultimodalTextBenchmark',
    'DataFrameBenchmark',
    'Mercari',
]

synthetic_datasets = [
    'FakeDataset',
]

__all__ = real_world_datasets + synthetic_datasets
