r"""Utility package."""
from enum import Enum

from .stype import stype
from .data import TensorFrame
from .typing import TaskType, DataFrame, NAStrategy
from torch_frame.utils import save, load, cat  # noqa

__version__ = '0.1.0'

__all__ = [
    'DataFrame',
    'stype',
    'set_stype',
    'TaskType',
    'NAStrategy',
    'TensorFrame',
    'save',
    'load',
    'cat',
    '__version__',
]


# TODO: maybe the arg should be a dict:
# {
#     torch_frame.numerical: my_package.numerical,
#     torch_frame.categorical: my_package.cat,
#     ...
# }
def set_stype(custom_stype: Enum) -> None:
    """Make stype members available.

    * torch_frame.numerical
    * torch_frame.categorical
    * torch_frame.text_embedded
    * torch_frame.text_tokenized
    * torch_frame.multicategorical
    * torch_frame.sequence_numerical
    * torch_frame.timestamp
    * torch_frame.embedding
    """
    # TODO: Check if custom_stype has necessary methods, e.g.
    # use_multi_embedding_tensor
    for name, value in custom_stype.__members__.items():
        globals()[name] = value
        if name not in __all__:
            __all__.append(name)

        globals()["stype"] = custom_stype


set_stype(stype)

# Call this from my_package
import my_package
set_stype(my_package.typing.Stype)
