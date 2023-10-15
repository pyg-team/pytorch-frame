from enum import Enum


class stype(Enum):
    r"""The semantic type of a column."""
    numerical = 'numerical'
    categorical = 'categorical'
    text_embedded = 'text_embedded'
    text_tokenized = 'text_tokenized'

    @property
    def is_text_stype(self) -> bool:
        return self in [stype.text_embedded]


numerical = stype('numerical')
categorical = stype('categorical')
text_embedded = stype('text_embedded')
text_tokenized = stype('text_tokenized')
