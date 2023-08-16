from enum import Enum


class stype(Enum):
    r"""The semantic type of a column."""
    numerical = 'numerical'
    categorical = 'categorical'
    unsupported = 'unsupported'


numerical = stype('numerical')
categorical = stype('categorical')
unsupported = stype('unsupported')
