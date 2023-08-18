# flake8: noqa

from .titanic import Titanic

real_world_datasets = [
    'Titanic',
]

synthetic_datasets = []

__all__ = real_world_datasets + synthetic_datasets
