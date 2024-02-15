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
        text_embedded: Pre-computed embeddings of text columns.
        text_tokenized: Tokenized text columns for finetuning.
        multicategorical: Multicategorical columns.
        sequence_numerical: Sequence of numerical values.
        embedding: Embedding columns.
        timestamp: Timestamp columns.
        image_embedded: Pre-computed embeddings of image columns.
    """
    numerical = 'numerical'
    categorical = 'categorical'
    text_embedded = 'text_embedded'
    text_tokenized = 'text_tokenized'
    multicategorical = 'multicategorical'
    sequence_numerical = 'sequence_numerical'
    timestamp = 'timestamp'
    image_embedded = 'image_embedded'
    embedding = 'embedding'

    @property
    def is_text_stype(self) -> bool:
        return self in [stype.text_embedded, stype.text_tokenized]

    @property
    def is_image_stype(self) -> bool:
        return self in [stype.image_embedded]

    @property
    def use_multi_nested_tensor(self) -> bool:
        r"""This property indicates if the data of an stype is stored in
        :class:`torch_frame.data.MultiNestedTensor`.
        """
        return self in [stype.multicategorical, self.sequence_numerical]

    @property
    def use_multi_embedding_tensor(self) -> bool:
        r"""This property indicates if the data of an stype is stored in
        :class:`torch_frame.data.MultiNestedTensor`.
        """
        return self in [
            stype.text_embedded, stype.image_embedded, stype.embedding
        ]

    @property
    def use_dict_multi_nested_tensor(self) -> bool:
        r"""This property indicates if the data of an stype is stored in
        a dictionary of :class:`torch_frame.data.MultiNestedTensor`.
        """
        return self in [stype.text_tokenized]

    @property
    def use_multi_tensor(self) -> bool:
        r"""This property indicates if the data of an
        :class:`~torch_frame.stype` is stored in
        :class:`torch_frame.data._MultiTensor`.
        """
        return self.use_multi_nested_tensor or self.use_multi_embedding_tensor

    @property
    def parent(self):
        r"""This property indicates if an :class:`~torch_frame.stype` is
        user-facing column :obj:`stype` or internal :obj:`stype` for grouping
        columns in :obj:`TensorFrame`. User-facing :class:`~torch_frame.stype`
        will be mapped to its parent during materialization. For
        :class:`<stypes>~torch_frame.stype` that are both internal and
        user-facing, the parent maps to itself.
        """
        if self == stype.text_embedded:
            return stype.embedding
        elif self == stype.image_embedded:
            return stype.embedding
        else:
            return self

    def __str__(self) -> str:
        return f'{self.name}'


numerical = stype('numerical')
categorical = stype('categorical')
text_embedded = stype('text_embedded')
text_tokenized = stype('text_tokenized')
multicategorical = stype('multicategorical')
sequence_numerical = stype('sequence_numerical')
timestamp = stype('timestamp')
image_embedded = stype('image_embedded')
embedding = stype('embedding')
