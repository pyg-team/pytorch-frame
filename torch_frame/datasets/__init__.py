# flake8: noqa

from .titanic import Titanic
from .adult_census_income import AdultCensusIncome
from .fake import FakeDataset
from .forest_cover_type import ForestCoverType
from .mushroom import Mushroom
from .poker_hand import PokerHand

real_world_datasets = [
    'Titanic',
    'AdultCensusIncome',
    'FakeDataset',
    'ForestCoverType',
    'Mushroom',
    "PokerHand",
]

synthetic_datasets = [
    'FakeDataset',
]

__all__ = real_world_datasets + synthetic_datasets
