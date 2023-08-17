from enum import Enum


class stype(Enum):
    r"""The semantic type of a column."""
    numerical = 'numerical'
    categorical = 'categorical'


numerical = stype('numerical')
categorical = stype('categorical')
