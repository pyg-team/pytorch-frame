from enum import Enum


class stype(Enum):
    numerical = 'numerical'
    categorical = 'categorical'
    unsupported = 'unsupported'


numerical = stype('numerical')
categorical = stype('categorical')
unsupported = stype('unsupported')
