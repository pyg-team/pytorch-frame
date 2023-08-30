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

real_world_datasets = [
    'Titanic',
    'AdultCensusIncome',
    'ForestCoverType',
    'Mushroom',
    'PokerHand',
    'BankMarketing',
    'TabularBenchmark',
    'Yandex',
]

synthetic_datasets = [
    'FakeDataset',
]

__all__ = real_world_datasets + synthetic_datasets
