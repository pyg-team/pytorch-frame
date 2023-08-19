from enum import Enum


class Stype(Enum):
    r"""The semantic type of a column."""
    numerical = 'numerical'
    categorical = 'categorical'


numerical = Stype('numerical')
categorical = Stype('categorical')

all_stype_list = [numerical, categorical]
