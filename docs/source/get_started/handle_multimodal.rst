Handling Datasets with Text Data
================================

:pyf:`PyTorch Frame` allows handling tablular dataset text columns by using
advanced text encoder. Featurizing text columns with well trained text model
can improve the overall tabular learning performance.

.. contents::
    :local:

Data Handling of Text on Common Benchmark Dataset
-------------------------------------------------

A table can contain multiple text columns and currently :pyf:`PyTorch Frame`
supports :obj:`~torch_frame.stype.text_embedded` :class:`~torch_frame.stype` to allow getting embeddings from text
models during the data materialization stage.

:pyf:`PyTorch Frame` also provides a list of tabular benchmark datasets with text
columns, *e.g.*, datasets from `Benchmarking Multimodal AutoML for Tabular Data with Text Fields <https://arxiv.org/abs/2111.02705>`_ .

Initializing and materializing the datasets are similar to those introduced in :doc:`/get_started/introduction`.
Following is an example to embed a common tabular text dataset with specified text encoder and caching during materialization.

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_frame.datasets import MultimodalTextBenchmark

    # Define an example text encoder to get text embeddings
    class ExampleTextEncoder:
        def __init__(self):
            self.model = SentenceTransformer('all-distilroberta-v1')

        def __call__(self, sentences: List[str]) -> Tensor:
            embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                           convert_to_tensor=True)
            return embeddings

    text_encoder = ExampleTextEncoder()

    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=text_encoder,
            batch_size=5), # Text encoder will encode 5 text rows each time
    )

    len(dataset)
    >>> 105154

    dataset.feat_cols # This dataset contains one text column `description`
    >>> ['description', 'country', 'province', 'points', 'price']

    # During materialization text encoding defined before will be called
    # Highly recommend caching the materialized dataset as text encoding
    # usually takes a long time
    dataset.materialize(path='/tmp/multimodal_text_benchmark/wine_reviews/data.pt')
