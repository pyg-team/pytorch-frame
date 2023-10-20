Handling Datasets with Text Data
================================

:pyf:`PyTorch Frame` enhances the learning on text columns in tabular datasets
through the use of advanced text encoders, such as BERT etc. By featurizing
text columns with well trained text models, it can boost the overall performance
of tabular learning.

.. contents::
    :local:

Handling Text Data on Common Benchmark Dataset
----------------------------------------------

In a table with several text columns, :pyf:`PyTorch Frame` offers the
support of :obj:`~torch_frame.stype.text_embedded` :class:`~torch_frame.stype`,
enabling retrieval of embeddings from text models during the materialization phase.

:pyf:`PyTorch Frame` additionally provides a collection of benchmark tabular datasets
that include text columns, such as those in the
`Benchmarking Multimodal AutoML for Tabular Data with Text Fields <https://arxiv.org/abs/2111.02705>`_.

The processes of initializing and materializing datasets are similar to those introduced in
:doc:`/get_started/introduction`. Below is an example demonstrating how to embed a standard
tabular text dataset using a specified text encoder and incorporating caching to speed up
the :meth:`~torch_frame.data.Dataset.materialize`.

.. code-block:: python

    from typing import List
    from sentence_transformers import SentenceTransformer
    from torch import Tensor
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_frame.datasets import MultimodalTextBenchmark

    # A simple example text encoder passed to the dataset
    class ExampleTextEncoder:
        def __init__(self):
            self.model = SentenceTransformer('all-distilroberta-v1')

        def __call__(self, sentences: List[str]) -> Tensor:
            # Encode list of sentences each time
            embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                           convert_to_tensor=True)
            return embeddings

    text_encoder = ExampleTextEncoder()
    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=text_encoder,
            batch_size=5),  # Text encoder will encode 5 text rows each time
    )

    len(dataset)
    >>> 105154

    dataset.feat_cols  # This dataset contains one text column `description`
    >>> ['description', 'country', 'province', 'points', 'price']

    # Materialize will call predefined encoder to encode text columns
    dataset.materialize(path='/tmp/multimodal_text_benchmark/wine_reviews/data.pt')

It's strongly recommended to cache the :class:`~torch_frame.TensorFrame`
by specifying the `path` during :meth:`~torch_frame.data.Dataset.materialize`,
as text encoding typically takes a considerable amount of time.
Cached :class:`~torch_frame.TensorFrame`, including encoded text embeddings, can be efficiently reused
for subsequent :meth:`~torch_frame.data.Dataset.materialize` calls.

Fusing Text Embeddings into Tabular Learning
--------------------------------------------

:pyf:`PyTorch Frame` offers :class:`~torch_frame.nn.encoder.LinearEmbeddingEncoder`, designed
to encode pre-computed embeddings. This encoder applies linear layer on each embedding feature
and concatenate the output embeddings.
This encoder can easily handle :obj:`~torch_frame.stype.text_embedded` case.

.. code-block:: python

    from torch_frame.nn.encoder import (
        EmbeddingEncoder,
        LinearEmbeddingEncoder,
        LinearEncoder,
    )

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.text_embedded: LinearEmbeddingEncoder(in_channels=768) # With text embedding size 768
    }

In the example above, `stype_encoder_dict` can be directly fed into
:class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder` directly and seamlessly
fusing text embeddings to the tabular learning.

Please refer to the
`pytorch-frame/examples/fttransformer_text.py <https://github.com/pyg-team/pytorch-frame/blob/master/examples/fttransformer_text.py>`_
for more information.
