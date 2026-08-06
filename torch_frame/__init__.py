r"""Utility package."""
from ._stype import (
    stype,
    numerical,
    categorical,
    text_embedded,
    text_tokenized,
    multicategorical,
    sequence_numerical,
    timestamp,
    image_embedded,
    embedding,
)
from .data import TensorFrame
from .typing import (
    TaskType,
    Metric,
    DataFrame,
    NAStrategy,
    WITH_PT24,
)
from torch_frame.utils import save, load, cat  # noqa
import torch_frame.data  # noqa
import torch_frame.datasets  # noqa
import torch_frame.nn  # noqa
import torch_frame.gbdt  # noqa

if WITH_PT24:
    import torch
    import numpy as np
    import _codecs

    torch.serialization.add_safe_globals([
        stype,
        torch_frame.data.stats.StatType,
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.int32)),
        _codecs.encode,
    ])

# https://peps.python.org/pep-0440/
__version__ = '0.4.0.dev0'

__all__ = [
    'DataFrame',
    'stype',
    'numerical',
    'categorical',
    'text_embedded',
    'text_tokenized',
    'multicategorical',
    'sequence_numerical',
    'timestamp',
    'image_embedded',
    'embedding',
    'TaskType',
    'Metric',
    'NAStrategy',
    'TensorFrame',
    'save',
    'load',
    'cat',
    'torch_frame',
    '__version__',
]
