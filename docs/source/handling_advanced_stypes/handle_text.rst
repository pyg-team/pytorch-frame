Handling Text Columns
================================

:pyf:`PyTorch Frame` supports text columns by utilizing external pre-trained
text embedding models, such as language models. Currently, we support
:class:`stype.text_embedded<torch_frame.stype>` for text columns,
which use a text embedding model to pre-encode text columns into embeddings
(frozen during model training). We plan to support fine-tuning of the text embedding
models soon.


.. contents::
    :local:

Handling Text Columns in a Benchmark Dataset
----------------------------------------------

:pyf:`PyTorch Frame` provides a collection of tabular benchmark datasets
with text columns, such as :obj:`~torch_frame.datasets.MultimodalTextBenchmark`.

In :pyf:`PyTorch Frame`, you can specify text columns as
:class:`stype.text_embedded<torch_frame.stype>`. This will
encode text columns using a user-specified text embedding model during the
dataset materialization stage.

The processes of initializing and materializing datasets are similar
to :doc:`/get_started/introduction`. Below we highlight the difference.

First you need to specify your text embedding model. Here, we use the
`SentenceTransformer <https://www.sbert.net/>`_ package.

.. code-block:: none

    pip install -U sentence-transformers

Next we create a text encoder class that encodes a list of strings into text embeddings.

.. code-block:: python

    from typing import List
    import torch
    from torch import Tensor
    from sentence_transformers import SentenceTransformer

    class TextEmbeddingModel:
    def __init__(self, device: torch.device):

        self.model = SentenceTransformer('all-distilroberta-v1', device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        # Encode a list of batch_size sentences into a PyTorch Tensor of
        # size [batch_size, emb_dim]
        embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                        convert_to_tensor=True)
        return embeddings.cpu()

Then we instantiate :obj:`~torch_frame.config.TextEmbedderConfig` for our text embedding model as follows.

.. code-block:: python

    from torch_frame.config.text_embedder import TextEmbedderConfig

    device = (torch.device('cuda')
          if torch.cuda.is_available() else torch.device('cpu'))

    text_embedder_cfg = TextEmbedderConfig(text_embedder=text_encoder,
                                       batch_size=5)

Here :obj:`text_embedder` maps a list of sentences into PyTorch Tensor embeddings
in mini-batch, where :obj:`batch_size` represents the batch size.

.. code-block:: python

    import torch_frame
    from torch_frame.datasets import MultimodalTextBenchmark


    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        col_to_text_embedder_cfg=text_embedder_cfg,
    )

    len(dataset)
    >>> 105154

    dataset.feat_cols  # This dataset contains one text column `description`
    >>> ['description', 'country', 'province', 'points', 'price']

    dataset.col_to_stype['description']
    >>> <stype.text_embedded: 'text_embedded'>

    # Materialize will call pre-defined encoding for text columns
    dataset.materialize(path='/tmp/multimodal_text_benchmark/wine_reviews/data.pt')

    # Text embedding of shape [num_rows, num_text_cols, emb_dim]
    dataset.tensor_frame.feat_dict[torch_frame.embedding].shape
    >>> torch.Size([105154, 1, 768])

.. note::
    Internally, :class:`~torch_frame.stype.text_embedded` is grouped together in the parent stype :class:`~torch_frame.stype.embedding` within :obj:`TensorFrame`.

It is strongly recommended to cache :class:`~torch_frame.TensorFrame`
by specifying the :obj:`path` during :meth:`~torch_frame.data.Dataset.materialize`,
as embedding texts in every materialization run can be quite time-consuming.
Once cached, :class:`~torch_frame.TensorFrame` can be reused for
subsequent :meth:`~torch_frame.data.Dataset.materialize` calls.

Fusing Text Embeddings into Tabular Learning
--------------------------------------------

:pyf:`PyTorch Frame` offers :class:`~torch_frame.nn.encoder.LinearEmbeddingEncoder` designed
to encode pre-computed embeddings. This encoder applies linear function over the
pre-computed embeddings, which can easily handle :obj:`~torch_frame.stype.text_embedded`.

As mentioned earlier, :class:`~torch_frame.stype.text_embedded` is stored together with other :obj:`embeddings` in :obj:`TensorFrame`, so we only need to specify the encoder for parent :obj:`~torch_frame.stype`, i.e. :class:`~torch_frame.stype.embedding`, in the :obj:`stype_encoder_dict`.

.. code-block:: python

    from torch_frame.nn.encoder import (
        EmbeddingEncoder,
        LinearEmbeddingEncoder,
        LinearEncoder,
    )

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.embedding: LinearEmbeddingEncoder()
    }

Then, :obj:`stype_encoder_dict` can be directly fed into
:class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder` to handle text columns.

Please refer to the
`pytorch-frame/examples/transformers_text.py <https://github.com/pyg-team/pytorch-frame/blob/master/examples/transformers_text.py>`_
for more information.
