from enum import Enum


class stype(Enum):
    r"""The semantic type of a column.
    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into an embedding space within tabular deep learning
    models:

    .. code-block:: python

        import torch_frame

        stype = torch_frame.numerical  # Numerical columns
        stype = torch_frame.categorical  # Categorical columns
        ...

    Attributes:
        numerical: Numerical columns.
        categorical: Categorical columns.
        multicategorical: Multicategorical columns.
        text_embedded: Pre-computed embeddings of text columns.
    """
    numerical = 'numerical'
    categorical = 'categorical'
    text_embedded = 'text_embedded'
    text_tokenized = 'text_tokenized'
    multicategorical = 'multicategorical'
    sequence_numerical = 'sequence_numerical'
    embedding = 'embedding'

    @property
    def is_text_stype(self) -> bool:
        return self in [stype.text_embedded, stype.text_tokenized]

    @property
    def use_multi_nested_tensor(self) -> bool:
        r""" This property indicates if the data of an stype is stored in
        :class:`torch_frame.data.MultiNestedTensor`.
        """
        return self in [stype.multicategorical, self.sequence_numerical]

    @property
    def use_dict_multi_nested_tensor(self) -> bool:
        r""" This property indicates if the data of an stype is stored in
        a dictionary of :class:`torch_frame.data.MultiNestedTensor`.
        """
        return self in [stype.text_tokenized]

    @property
    def use_multi_nested_tensor(self) -> bool:
        r""" This property indicates if the data of an stype is stored in
        :class:`torch_frame.data.MultiEmbeddingTensor`.
        """
        return self in [stype.embedding]


numerical = stype('numerical')
categorical = stype('categorical')
text_embedded = stype('text_embedded')
text_tokenized = stype('text_tokenized')
multicategorical = stype('multicategorical')
sequence_numerical = stype('sequence_numerical')
embedding = stype('embedding')
