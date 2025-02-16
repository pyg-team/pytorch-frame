# flake8: noqa
from .adult_census_income import AdultCensusIncome
from .amazon_fine_food_reviews import AmazonFineFoodReviews
from .amphibians import Amphibians
from .bank_marketing import BankMarketing
from .data_frame_benchmark import DataFrameBenchmark
from .data_frame_text_benchmark import DataFrameTextBenchmark
from .diamond_images import DiamondImages
from .dota2 import Dota2
from .fake import FakeDataset
from .forest_cover_type import ForestCoverType
from .huggingface_dataset import HuggingFaceDatasetDict
from .kdd_census_income import KDDCensusIncome
from .mercari import Mercari
from .movielens_1m import Movielens1M
from .multimodal_text_benchmark import MultimodalTextBenchmark
from .mushroom import Mushroom
from .poker_hand import PokerHand
from .tabular_benchmark import TabularBenchmark
from .titanic import Titanic
from .yandex import Yandex

real_world_datasets = [
    'AdultCensusIncome',
    'AmazonFineFoodReviews',
    'Amphibians',
    'BankMarketing',
    'DataFrameBenchmark',
    'DataFrameTextBenchmark',
    'Dota2',
    'Titanic',
    'ForestCoverType',
    'HuggingFaceDatasetDict',
    'KDDCensusIncome',
    'Mercari',
    'Movielens1M',
    'MultimodalTextBenchmark',
    'Mushroom',
    'PokerHand',
    'TabularBenchmark',
    'Yandex',
    'DiamondImages',
]

synthetic_datasets = [
    'FakeDataset',
]

other_datasets = [
    'HuggingFaceDatasetDict',
]

__all__ = real_world_datasets + synthetic_datasets + other_datasets
