r"""Encoder package."""
from .encoder import FeatureEncoder
from .stypewise_encoder import StypeWiseFeatureEncoder
from .stype_encoder import (
    StypeEncoder,
    EmbeddingEncoder,
    MultiCategoricalEmbeddingEncoder,
    LinearEncoder,
    LinearBucketEncoder,
    LinearPeriodicEncoder,
    ExcelFormerEncoder,
    LinearEmbeddingEncoder,
    LinearMultiEmbeddingEncoder,
    LinearModelEncoder,
    StackEncoder,
)

__all__ = classes = [
    'FeatureEncoder',
    'StypeWiseFeatureEncoder',
    'StypeEncoder',
    'EmbeddingEncoder',
    'MultiCategoricalEmbeddingEncoder',
    'LinearEncoder',
    'LinearBucketEncoder',
    'LinearPeriodicEncoder',
    'ExcelFormerEncoder',
    'LinearEmbeddingEncoder',
    'LinearMultiEmbeddingEncoder',
    'LinearModelEncoder',
    'StackEncoder',
]
