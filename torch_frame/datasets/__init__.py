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
from .data_frame_text_benchmark import DataFrameTextBenchmark
from .mercari import Mercari
from .movielens_1m import Movielens1M
from .amazon_fine_food_reviews import AmazonFineFoodReviews
from .diamond_images import DiamondImages
from .huggingface_dataset import HuggingFaceDatasetDict

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
    'DataFrameTextBenchmark',
    'Mercari',
    'Movielens1M',
    'AmazonFineFoodReviews',
    'DiamondImages',
]

synthetic_datasets = [
    'FakeDataset',
]

other_datasets = [
    'HuggingFaceDatasetDict',
]

__all__ = real_world_datasets + synthetic_datasets + other_datasets
