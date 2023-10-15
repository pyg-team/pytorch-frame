from enum import Enum


class stype(Enum):
    r"""The semantic type of a column.

    Attributes:
        numerical: Numerical features.
        categorical: Categorical features.
        text_embedded: Pre-embedding of text.
    """
    numerical = 'numerical'
    categorical = 'categorical'
    text_embedded = 'text_embedded'

    @property
    def is_text_stype(self) -> bool:
        return self in [stype.text_embedded]


numerical = stype('numerical')
categorical = stype('categorical')
text_embedded = stype('text_embedded')
