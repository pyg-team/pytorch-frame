from enum import Enum


class stype(Enum):
    r"""The semantic type of a column."""
    numerical = 'numerical'
    categorical = 'categorical'
    text_encoded = 'text_encoded'


numerical = stype('numerical')
categorical = stype('categorical')
text_encoded = stype('text_encoded')
