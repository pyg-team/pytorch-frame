# flake8: noqa

from .titanic import Titanic
from .adult_census_income import AdultCensusIncome

real_world_datasets = [
    'Titanic',
    'AdultCensusIncome',
]

synthetic_datasets = []

__all__ = real_world_datasets + synthetic_datasets
